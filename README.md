# 🧪 Détecteur de Gaz Toxiques — Machine Learning Pipeline

## 📋 Description

Ce projet propose un pipeline de Machine Learning pour la **prédiction de concentrations de gaz toxiques** à partir de données de capteurs environnementaux. Deux scripts sont disponibles, chacun représentant une itération d'optimisation du modèle.

---

## 📁 Structure des fichiers

```
projet/
├── test meilleur.py       # Pipeline RF optimisé (n_estimators=600, best score)
├── train9.py              # Pipeline RF + HGB avec sélection automatique du meilleur modèle
├── x_train.csv            # Données d'entrainement (features)
├── y_train.csv            # Données d'entrainement (targets)
├── x_test.csv             # Données de test
└── submission.csv         # Fichier de soumission généré
```

---

## 🔧 Prérequis

### Python 3.8+

### Librairies requises

```bash
pip install numpy pandas scikit-learn
```

---

## 🗂️ Données

### Features (`x_train.csv` / `x_test.csv`)

| Colonne    | Description                        |
|------------|------------------------------------|
| `ID`       | Identifiant unique de l'échantillon|
| `Humidity` | Taux d'humidité ambiant            |
| `M4`–`M7` | Capteurs de type M (groupe 1)      |
| `M12`–`M15`| Capteurs de type M (groupe 2)     |
| `R`        | Capteur de résistance              |
| `S1`–`S3`  | Capteurs de type S                 |

### Targets (`y_train.csv`)

- **23 colonnes** : `c01` à `c23`
- Valeurs comprises entre **0 et 1** représentant les concentrations de chaque gaz

---

## 📊 Métrique d'évaluation

Le projet utilise une **Weighted RMSE** personnalisée :

```
f = 1.2  si y_true >= 0.5  (pénalise davantage les fortes concentrations)
f = 1.0  sinon

Weighted RMSE = sqrt( mean( mean( f * (y_pred - y_true)² ) ) )
```

> 🏆 Meilleur score obtenu : **0.14273**

---

## ⚙️ Feature Engineering

Le pipeline génère automatiquement des features enrichies à partir des capteurs bruts :

| Bloc            | Features générées                                          |
|-----------------|------------------------------------------------------------|
| **Polynomial**  | `col²`, `√col`, `log(1+col)` pour chaque capteur          |
| **Ratios**      | Ratios entre capteurs consécutifs                          |
| **Différences** | Différences entre capteurs consécutifs                     |
| **Statistiques**| Moyenne, écart-type, min, max des groupes M et S           |
| **Croisements** | Produits entre paires de capteurs clés                     |

> ⚡ Tous les blocs de feature engineering sont exécutés **en parallèle** via `ThreadPoolExecutor`.

---

## 🌲 Modèles

### `test meilleur.py` — Random Forest Optimisé

```python
RandomForestRegressor(
    n_estimators      = 600,    # Augmenté vs baseline (+300)
    max_depth         = 20,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    max_features      = 'sqrt',
    bootstrap         = True,
)
```

- Enveloppé dans `MultiOutputRegressor`
- Ré-entraînement final sur **100% des données**
- Split validation : **85% / 15%**

### `train9.py` — Random Forest + HistGradientBoosting

```python
# Deux modèles entraînés en parallèle :
RandomForestRegressor(n_estimators=300, ...)
HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, ...)
```

- Sélection automatique du **meilleur modèle** selon la Weighted RMSE
- Split validation : **80% / 20%**

---

## 🔄 Pipeline d'exécution

```
[1] Chargement parallèle des CSV
        ↓
[2] Rééquilibrage de la distribution Humidité (train ↔ test)
        ↓
[3] Feature Engineering parallèle (~60+ features)
        ↓
[4] Split Train / Validation
        ↓
[5] Entraînement + Évaluation Weighted RMSE
        ↓
[6] Ré-entraînement final 100% + Génération submission.csv
```

---

## 🚀 Utilisation

### Script optimisé (`test meilleur.py`)

```python
base = r'chemin/vers/vos/données'
model, predictions = run_pipeline(
    train_input_path  = f'{base}/x_train.csv',
    train_output_path = f'{base}/y_train.csv',
    test_input_path   = f'{base}/x_test.csv',
    output_path       = f'{base}/submission.csv',
    n_estimators      = 600,
    max_features      = 'sqrt',
    min_samples_leaf  = 2,
)
```

### Script multi-modèles (`train9.py`)

```python
base = r'chemin/vers/vos/données'
model, predictions = run_pipeline(
    train_input_path  = f'{base}/x_train.csv',
    train_output_path = f'{base}/y_train.csv',
    test_input_path   = f'{base}/x_test.csv',
    output_path       = f'{base}/submission.csv',
)
```

---

## 📈 Résultats

| Script              | Modèle          | n_estimators |
|---------------------|-----------------|:------------:|
| `train9.py`         | Random Forest   | 300          |
| `test meilleur.py`  | Random Forest   | 600          |

---

## ⚖️ Rééquilibrage de l'humidité

Le pipeline détecte et corrige automatiquement le déséquilibre de la distribution d'humidité entre le jeu d'entraînement et le jeu de test :

- Si le train contient **trop d'échantillons à humidité ≈ 0**, un sous-échantillonnage aléatoire est appliqué
- Seuil de détection : `Humidity ≤ 0.173`
- Reproductibilité garantie via `random_state=42`

---

## 👤 Auteur

Projet réalisé dans le cadre du cours de **Machine Learning — M1 YNOV**
