# ============================================================
# DÉTECTEUR DE GAZ TOXIQUES — RF baseline + n_estimators=600
# Variation MINIMALE vs best (0.14273) : seul n_estimators change
# ============================================================

import os
import time
import warnings
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split

warnings.filterwarnings('ignore')

N_CPUS = os.cpu_count() or 4

FEATURE_COLS = ['Humidity', 'M12', 'M13', 'M14', 'M15',
                'M4', 'M5', 'M6', 'M7', 'R', 'S1', 'S2', 'S3']

TARGET_COLS = [f'c{i:02d}' for i in range(1, 24)]

# ============================================================
# 1. MÉTRIQUE OFFICIELLE
# ============================================================

def weighted_rmse(y_true, y_pred):
    f = np.where(y_true >= 0.5, 1.2, 1.0)
    per_sample = np.mean(f * (y_pred - y_true) ** 2, axis=1)
    return np.sqrt(np.mean(per_sample))

# ============================================================
# 2. CHARGEMENT PARALLÈLE
# ============================================================

def load_csv_parallel(file_map: dict) -> dict:
    results = {}

    def _load(name, path):
        t0 = time.time()
        df = pd.read_csv(path)
        return name, df, time.time() - t0

    with ThreadPoolExecutor(max_workers=N_CPUS) as exe:
        futures = {exe.submit(_load, name, path): name
                   for name, path in file_map.items()}
        for fut in as_completed(futures):
            name, df, elapsed = fut.result()
            results[name] = df
            print(f"  ✔ {name:<25} {str(df.shape):<14} ({elapsed:.2f}s)")

    return results

# ============================================================
# 3. RÉÉQUILIBRAGE HUMIDITÉ
# ============================================================

def balance_humidity(dfX: pd.DataFrame, dfY: pd.DataFrame,
                     dftest: pd.DataFrame) -> tuple:
    dfX = dfX.reset_index(drop=True)
    dfY = dfY.reset_index(drop=True)

    n_test_h0  = (dftest['Humidity'] <= 0.1).sum()
    n_train_h0 = (dfX['Humidity']    <= 0.1).sum()

    print(f"  Humidity≈0 → train : {n_train_h0} | test : {n_test_h0}")

    if n_train_h0 > n_test_h0:
        mask_h0  = dfX['Humidity'] <= 0.173
        h0_idx   = dfX[mask_h0].index
        not_h0_idx = dfX[~mask_h0].index

        h0_down_idx = pd.Index(
            np.random.RandomState(42).choice(h0_idx, size=n_test_h0, replace=False)
        )

        keep_idx = h0_down_idx.append(not_h0_idx)
        keep_idx = keep_idx[np.random.RandomState(42).permutation(len(keep_idx))]

        dfX_bal = dfX.loc[keep_idx].reset_index(drop=True)
        dfY_bal = dfY.loc[keep_idx].reset_index(drop=True)
        print(f"  Après rééquilibrage → {len(dfX_bal)} lignes")
    else:
        dfX_bal = dfX.reset_index(drop=True)
        dfY_bal = dfY.reset_index(drop=True)
        print("  Pas de rééquilibrage nécessaire")

    return dfX_bal, dfY_bal

# ============================================================
# 4. FEATURE ENGINEERING PARALLÈLE
# ============================================================

def prepare_features_parallel(df: pd.DataFrame) -> pd.DataFrame:
    available = [c for c in FEATURE_COLS if c in df.columns]
    base = df[available].copy().reset_index(drop=True)

    def _feat_poly(df_):
        out = {}
        for col in available[1:]:
            out[f'{col}_sq']   = df_[col] ** 2
            out[f'{col}_sqrt'] = np.sqrt(np.abs(df_[col]))
            out[f'{col}_log']  = np.log1p(np.abs(df_[col]))
        return pd.DataFrame(out, index=df_.index)

    def _feat_ratios(df_):
        out = {}
        sensor_cols = available[1:]
        for i in range(len(sensor_cols) - 1):
            denom = df_[sensor_cols[i+1]].replace(0, 1e-9)
            out[f'{sensor_cols[i]}_{sensor_cols[i+1]}_ratio'] = df_[sensor_cols[i]] / denom
        return pd.DataFrame(out, index=df_.index)

    def _feat_diffs(df_):
        out = {}
        sensor_cols = available[1:]
        for i in range(len(sensor_cols) - 1):
            out[f'{sensor_cols[i]}_{sensor_cols[i+1]}_diff'] = \
                df_[sensor_cols[i]] - df_[sensor_cols[i+1]]
        return pd.DataFrame(out, index=df_.index)

    def _feat_stats(df_):
        m_cols = [c for c in ['M4', 'M5', 'M6', 'M7', 'M12', 'M13', 'M14', 'M15'] if c in df_.columns]
        s_cols = [c for c in ['S1', 'S2', 'S3'] if c in df_.columns]
        out = {}
        if m_cols:
            out['M_mean'] = df_[m_cols].mean(axis=1)
            out['M_std']  = df_[m_cols].std(axis=1)
            out['M_max']  = df_[m_cols].max(axis=1)
            out['M_min']  = df_[m_cols].min(axis=1)
        if s_cols:
            out['S_mean'] = df_[s_cols].mean(axis=1)
            out['S_std']  = df_[s_cols].std(axis=1)
        if m_cols and s_cols:
            out['M_S_diff'] = df_[m_cols].mean(axis=1) - df_[s_cols].mean(axis=1)
        if 'R' in df_.columns and 'Humidity' in df_.columns:
            out['R_Humidity_ratio'] = df_['R'] / df_['Humidity'].replace(0, 1e-9)
        return pd.DataFrame(out, index=df_.index)

    def _feat_cross(df_):
        out = {}
        pairs = [('S1','M4'), ('S2','M5'), ('S3','M6'), ('R','S1'), ('R','M4')]
        for a, b in pairs:
            if a in df_.columns and b in df_.columns:
                out[f'{a}_x_{b}'] = df_[a] * df_[b]
        return pd.DataFrame(out, index=df_.index)

    blocs  = [_feat_poly, _feat_ratios, _feat_diffs, _feat_stats, _feat_cross]
    extras = []

    with ThreadPoolExecutor(max_workers=min(5, N_CPUS)) as exe:
        futures = {exe.submit(fn, base): fn.__name__ for fn in blocs}
        for fut in as_completed(futures):
            try:
                extras.append(fut.result())
            except Exception as e:
                print(f"  ⚠ Feature bloc erreur : {e}")

    result = pd.concat([base] + extras, axis=1)
    result = result.loc[:, ~result.columns.duplicated()]
    return result

# ============================================================
# 5. MODÈLE RF (BASELINE — IDENTIQUE AU BEST 0.14273)
# ============================================================

def build_rf(n_estimators=600,
             max_features='sqrt',
             min_samples_leaf=2,
             random_state=42):
    base = RandomForestRegressor(
        n_estimators      = n_estimators,
        max_depth         = 20,
        min_samples_split = 5,
        min_samples_leaf  = min_samples_leaf,
        max_features      = max_features,
        n_jobs            = -1,
        random_state      = random_state,
        bootstrap         = True,
    )
    return MultiOutputRegressor(base, n_jobs=1)

# ============================================================
# 6. PIPELINE PRINCIPAL
# ============================================================

def run_pipeline(train_input_path, train_output_path, test_input_path,
                 output_path='submission.csv',
                 n_estimators=600,
                 max_features='sqrt',
                 min_samples_leaf=2):

    print("=" * 65)
    print("  RF baseline (MultiOutput) — variation minimale vs best")
    print(f"  n_estimators={n_estimators} | max_features={max_features} "
          f"| leaf={min_samples_leaf}")
    print(f"  CPUs disponibles : {N_CPUS}")
    print("=" * 65)

    total_t0 = time.time()

    # ── [1/5] Chargement ────────────────────────────────────────────
    print("\n[1/5] Chargement parallèle des données...")
    data = load_csv_parallel({
        'train_input':  train_input_path,
        'train_output': train_output_path,
        'test_input':   test_input_path,
    })

    dfX    = data['train_input'].drop(columns=['ID'], errors='ignore')
    dfY    = data['train_output'].drop(columns=['ID'], errors='ignore')
    dftest = data['test_input']
    dftest_clean = dftest.drop(columns=['ID'], errors='ignore')

    dfY = dfY[[c for c in TARGET_COLS if c in dfY.columns]]
    target_cols_used = list(dfY.columns)

    # ── [2/5] Rééquilibrage humidité ────────────────────────────────
    print("\n[2/5] Rééquilibrage humidité...")
    dfX_bal, dfY_bal = balance_humidity(dfX, dfY, dftest_clean)

    # ── [3/5] Feature engineering ───────────────────────────────────
    print("\n[3/5] Feature engineering parallèle...")
    t0 = time.time()
    X_all  = prepare_features_parallel(dfX_bal)
    X_test = prepare_features_parallel(dftest_clean)

    for col in X_all.columns:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[X_all.columns]

    X_all  = X_all.replace([np.inf, -np.inf], 0).fillna(0)
    X_test = X_test.replace([np.inf, -np.inf], 0).fillna(0)
    y_mat  = dfY_bal.values.astype(float)

    print(f"  Features finales : {X_all.shape[1]} | ({time.time()-t0:.2f}s)")

    # ── [4/5] Validation ────────────────────────────────────────────
    print("\n[4/5] Split validation 85/15 (mesure)...")
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_all, y_mat, test_size=0.15, random_state=42
    )
    print(f"  Train : {len(X_tr)} | Validation : {len(X_val)}")

    print("\n  Entraînement validation (RF MultiOutput)...")
    t0 = time.time()
    val_model = build_rf(n_estimators, max_features, min_samples_leaf)
    val_model.fit(X_tr.values, y_tr)
    print(f"  ✔ Modèle validation entraîné en {time.time()-t0:.1f}s")

    val_pred = np.clip(val_model.predict(X_val.values), 0, 1)
    val_score = weighted_rmse(y_val, val_pred)
    print(f"\n  ====> Weighted RMSE validation : {val_score:.6f}")

    # Top features
    try:
        importances = np.mean(
            [est.feature_importances_ for est in val_model.estimators_], axis=0
        )
        feat_imp = pd.Series(importances, index=X_all.columns)\
                     .sort_values(ascending=False)
        print("\n  Top 10 features importantes :")
        for fname, val in feat_imp.head(10).items():
            bar = '█' * int(val * 300)
            print(f"    {fname:<30} {val:.4f} {bar}")
    except Exception as e:
        print(f"  (importances indisponibles : {e})")

    # ── [5/5] Ré-entraînement final 100% ────────────────────────────
    print("\n[5/5] Ré-entraînement final sur 100% des données...")
    t0 = time.time()
    final_model = build_rf(n_estimators, max_features, min_samples_leaf)
    final_model.fit(X_all.values, y_mat)
    print(f"  ✔ Modèle final entraîné en {time.time()-t0:.1f}s")

    print("\n  Génération des prédictions sur le test set...")
    test_preds = np.clip(final_model.predict(X_test.values), 0, 1)

    pred_df = pd.DataFrame(0.0, index=range(len(test_preds)), columns=TARGET_COLS)
    for j, col in enumerate(target_cols_used):
        pred_df[col] = test_preds[:, j]

    if 'ID' in dftest.columns:
        pred_df.insert(0, 'ID', dftest['ID'].values)

    pred_df.to_csv(output_path, index=False)
    print(f"  ✔ Fichier sauvegardé : '{output_path}'  {pred_df.shape}")
    print(f"\n  ⏱  Temps total : {time.time()-total_t0:.2f}s")
    print("\n" + "=" * 65)
    print("  PIPELINE TERMINÉ ✔")
    print("=" * 65)

    return final_model, pred_df

# ============================================================
# 7. POINT D'ENTRÉE
# ============================================================

if __name__ == '__main__':
    base = r'c:/Users/Utilisateur/Documents/YNOV/M1/Machine learning/gaz toxique'
    model, predictions = run_pipeline(
        train_input_path  = f'{base}/x_train.csv',
        train_output_path = f'{base}/y_train.csv',
        test_input_path   = f'{base}/x_test.csv',
        output_path       = f'{base}/submission.csv',
        n_estimators      = 600,   # SEUL changement vs best (était 300)
        max_features      = 'sqrt',
        min_samples_leaf  = 2,
    )
