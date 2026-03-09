# ============================================================
# Script : generate_all_figures.py
# Objectif : Génération automatique des figures et tableaux
# Auteur : ChatGPT
# ============================================================

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline

from statsmodels.tsa.statespace.sarimax import SARIMAX

sns.set(style="whitegrid")

# ============================================================
# 0. Préparation des dossiers
# ============================================================
os.makedirs("figures", exist_ok=True)
os.makedirs("tables", exist_ok=True)
os.makedirs("models", exist_ok=True)

print("📁 Dossiers prêts : figures/, tables/, models/")


# ============================================================
# 1. Chargement des données
# ============================================================
print("📥 Chargement du fichier Excel...")

df = pd.read_excel("données_anjara_gid_ANALYSE.xlsx")

print("Données chargées :", df.shape)


# ============================================================
# 2. 🔍 Figure 7.1 — Matrice des valeurs manquantes
# ============================================================
plt.figure(figsize=(12, 6))
sns.heatmap(df.isna(), cbar=False)
plt.title("Figure 7.1 — Matrice des valeurs manquantes")
plt.savefig("figures/fig_7_1_missing_matrix.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 3. 📊 Figure 7.2 — Histogramme FONDS (log1p)
# ============================================================
plt.figure(figsize=(8, 5))
sns.histplot(np.log1p(df["FONDS"].fillna(0)), bins=50, kde=True)
plt.title("Figure 7.2 — Distribution de log(FONDS+1)")
plt.savefig("figures/fig_7_2_hist_fonds.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 4. 🧱 Figure 7.3 — Top 20 ACTPR
# ============================================================
plt.figure(figsize=(10, 6))
df["ACTPR"].value_counts().head(20).plot(kind="bar")
plt.title("Figure 7.3 — Top 20 des valeurs ACTPR")
plt.savefig("figures/fig_7_3_top20_actpr.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 5. 🌍 Figure 7.4 — Répartition par FIVON
# ============================================================
plt.figure(figsize=(10, 5))
df["FIVON"].value_counts().head(15).plot(kind="bar")
plt.title("Figure 7.4 — Répartition des entreprises par FIVON")
plt.savefig("figures/fig_7_4_fivon_repartition.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 6. 📈 Figure 7.5 — Créations annuelles
# ============================================================
df["annee"] = pd.to_datetime(df["DTCRE"]).dt.year
creations = df.groupby("annee").size()

plt.figure(figsize=(10, 5))
creations.plot()
plt.title("Figure 7.5 — Créations d'entreprises par année")
plt.savefig("figures/fig_7_5_creations_par_annee.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 7. 🔥 Figure 7.6 — Corrélations
# ============================================================
plt.figure(figsize=(10, 8))
sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=False, cmap='coolwarm')
plt.title("Figure 7.6 — Heatmap des corrélations")
plt.savefig("figures/fig_7_6_corr_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 8. 🧼 Préparation pour les modèles
# ============================================================

# Variables simples pour démonstration
df_model = df[['FONDS', 'QUALITE', 'FIVON', 'LFJ', 'annee']].copy()

# Nettoyage minimal
df_model = df_model.dropna()

# Encodeur pour FIVON et LFJ
df_model = pd.get_dummies(df_model, columns=["FIVON", "LFJ"], drop_first=True)

X = df_model.drop("FONDS", axis=1)
y = np.log1p(df_model["FONDS"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# ============================================================
# 9. ⚙️ Figure 7.7 — Distribution après normalisation
# ============================================================
scaler = StandardScaler()
scaled = scaler.fit_transform(X)

plt.figure(figsize=(10, 5))
sns.boxplot(data=pd.DataFrame(scaled, columns=X.columns))
plt.xticks(rotation=90)
plt.title("Figure 7.7 — Distribution des variables normalisées")
plt.savefig("figures/fig_7_7_scaled_boxplots.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 10. 🌳 Random Forest — Modèle + importance
# ============================================================

rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)

joblib.dump(rf, "models/rf_model.pkl")

importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

plt.figure(figsize=(10, 6))
importances.tail(20).plot(kind="barh")
plt.title("Figure 7.9 — Importance des variables (RF)")
plt.savefig("figures/fig_7_9_feature_importance.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 11. 📌 Figure 7.8 — Prédictions vs réel
# ============================================================

y_pred = rf.predict(X_test)

plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()], 'r--')
plt.title("Figure 7.8 — Prédictions vs Réel")
plt.xlabel("Réel")
plt.ylabel("Prédit")
plt.savefig("figures/fig_7_8_pred_vs_actual.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 12. 🧪 Figure 7.10 — Résidus
# ============================================================

residuals = y_test - y_pred

plt.figure(figsize=(8, 5))
sns.histplot(residuals, kde=True)
plt.title("Figure 7.10 — Distribution des résidus (Random Forest)")
plt.savefig("figures/fig_7_10_residuals.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 13. ⏳ Figure 7.11 — Prévision SARIMA
# ============================================================

ts = df.groupby("annee").size()

model = SARIMAX(ts, order=(1, 1, 1), seasonal_order=(1, 1, 1, 1))
res = model.fit(disp=False)

forecast = res.get_forecast(steps=3)
frame = forecast.summary_frame()
frame.to_csv("tables/table_7_3_sarima_forecast.csv")

plt.figure(figsize=(10, 5))
ts.plot(label="Réel")
forecast.predicted_mean.plot(label="Prévision")
plt.fill_between(
    frame.index,
    frame["mean_ci_lower"],
    frame["mean_ci_upper"],
    alpha=0.3
)
plt.legend()
plt.title("Figure 7.11 — Prévisions SARIMA")
plt.savefig("figures/fig_7_11_sarima_forecast.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 14. 🔍 Figure 7.12 — Courbes d'apprentissage (GridSearch)
# ============================================================

param_grid = {
    "n_estimators": [100, 200],
    "max_depth": [None, 10],
}

gscv = GridSearchCV(
    RandomForestRegressor(),
    param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=-1
)

gscv.fit(X, y)

results = pd.DataFrame(gscv.cv_results_)
results.to_csv("tables/table_7_1_gridsearch_results.csv", index=False)

plt.figure(figsize=(10, 5))
plt.plot(results["mean_test_score"])
plt.title("Figure 7.12 — Performance GridSearch")
plt.savefig("figures/fig_7_12_learning_curve.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# 15. 🧩 Figure 7.13 — Cas d'étude
# ============================================================

sample = pd.DataFrame({
    "Reel": y_test[:5],
    "Pred": y_pred[:5]
})
sample.to_csv("tables/table_7_2_case_studies.csv")

plt.figure(figsize=(8, 5))
plt.plot(sample["Reel"].values, label="Réel")
plt.plot(sample["Pred"].values, label="Prédit")
plt.title("Figure 7.13 — Cas d'étude Réel vs Prédit")
plt.legend()
plt.savefig("figures/fig_7_13_case_studies.png", dpi=150, bbox_inches='tight')
plt.close()


# ============================================================
# FIN
# ============================================================
print("🎉 Toutes les figures et tableaux ont été générés avec succès !")
