# -*- coding: utf-8 -*-
"""
PCA + Clustering + Évaluations sur Campagne_Market.csv

Exécuter:
    python pca_clustering_market.py

Sorties dans ./outputs :
    - pca_3d.png : projection 3D PCA
    - clusters_3d.png : clusters (meilleur modèle) sur PCA 3D
    - silhouette_scores.csv : comparaison KMeans/Agglo (K=2..8)
    - cluster_sizes.csv : volume et % par cluster
    - spend_income_by_cluster.csv : dépense, revenu, ratio
    - campaigns_by_cluster.csv : taux d'acceptation par campagne et cluster
    - rf_top20_importances.csv : top 20 déterminants de la dépense
    - cluster_profiles.csv : médianes des variables clés par cluster
    - SUMMARY.txt : réponses simples aux questions 6.i — 6.iv
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from itertools import product

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor

plt.switch_backend("Agg")  # pour sauvegarder sans interface graphique

# === 1) Chargement du CSV ===
from pathlib import Path
import os

HERE = Path(__file__).resolve().parent     # <-- chemin du dossier du script
DATA = HERE / "Camp_Market.csv"            # <-- ton vrai fichier CSV
OUT = HERE / "outputs"                     # dossier de sortie
OUT.mkdir(exist_ok=True, parents=True)

# === 1) Chargement robuste du CSV ===
try_opts = [
    {"sep": ";", "encoding": "utf-8"},
    {"sep": ";", "encoding": "latin-1"},
    {"sep": ",", "encoding": "utf-8"},
]

df = None
for opt in try_opts:
    try:
        tmp = pd.read_csv(DATA, **opt)
        if tmp.shape[1] >= 10:
            df = tmp
            used_opt = opt
            break
    except Exception:
        continue

if df is None:
    raise RuntimeError("Impossible de lire le CSV. Vérifie le chemin et le séparateur.")

print(f"CSV chargé avec {used_opt} — shape = {df.shape}")

# Dates / Tenure (optionnel)
if "Dt_Customer" in df.columns:
    df["Dt_Customer"] = pd.to_datetime(df["Dt_Customer"], errors="coerce", dayfirst=True)
    df["Tenure_days"] = (df["Dt_Customer"].max() - df["Dt_Customer"]).dt.days

# === 2) Features numériques pour PCA (exclusion d'IDs et flags de campagnes) ===
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
exclude_patterns = ["ID", "Id", "id", "Customer", "Cust", "Dt_", "Year", "Month",
                    "AcceptedCmp", "Response", "Complain", "Z_CostContact", "Z_Revenue"]
exclude_cols = set()
for c in num_cols:
    for pat in exclude_patterns:
        if pat in c:
            exclude_cols.add(c)
            break

feature_cols = [c for c in num_cols if c not in exclude_cols]
base_feature_cols = feature_cols.copy()

# Sécurité: si peu de features numériques -> tenter coercition de colonnes en nombre
if len(feature_cols) < 3:
    for c in df.columns:
        if c not in num_cols:
            s = df[c].astype(str).str.replace("\u00A0", "", regex=False)
            s = s.str.replace(",", ".", regex=False)
            df[c] = pd.to_numeric(s, errors="ignore")
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in num_cols if all(p not in c for p in exclude_patterns)]
    base_feature_cols = feature_cols.copy()

if len(feature_cols) < 3:
    raise RuntimeError("Pas assez de colonnes numériques pertinentes pour faire une PCA à 3 composantes.")

# Prétraitement
X_raw = df[feature_cols].copy()
X_raw = X_raw.fillna(X_raw.median(numeric_only=True))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# === 3) PCA (3 composantes) ===
pca = PCA(n_components=3, random_state=42)
X_pca = pca.fit_transform(X_scaled)
expl = pca.explained_variance_ratio_
print(f"Variance expliquée PCA (PC1..PC3) = {expl} — cumul={expl.sum():.4f}")

# Figure PCA 3D
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X_pca[:,0], X_pca[:,1], X_pca[:,2], s=6)
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title("Projection PCA (3 composantes)")
fig.tight_layout()
fig.savefig(OUT / "pca_3d.png", dpi=160)
plt.close(fig)

# === 4) Sélection du modèle de clustering (KMeans vs Agglo, K=2..8) ===
rows = []
for method, K in product(["kmeans", "agglo"], range(2, 9)):
    try:
        if method == "kmeans":
            model = KMeans(n_clusters=K, n_init="auto", random_state=42)
            labels = model.fit_predict(X_pca)
        else:
            model = AgglomerativeClustering(n_clusters=K, linkage="ward")
            labels = model.fit_predict(X_pca)
        sil = silhouette_score(X_pca, labels)
        rows.append({"method": method, "K": K, "silhouette": sil})
    except Exception as e:
        rows.append({"method": method, "K": K, "silhouette": np.nan})

scores = pd.DataFrame(rows).sort_values("silhouette", ascending=False)
scores.to_csv(OUT / "silhouette_scores.csv", index=False)

best = scores.iloc[0]
best_method, best_K = best["method"], int(best["K"])
print(f"Meilleur clustering: {best_method} avec K={best_K} (silhouette={best['silhouette']:.3f})")

if best_method == "kmeans":
    best_model = KMeans(n_clusters=best_K, n_init="auto", random_state=42)
else:
    best_model = AgglomerativeClustering(n_clusters=best_K, linkage="ward")

labels = best_model.fit_predict(X_pca)
df["cluster"] = labels

# Plot clusters 3D
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')
for k in sorted(np.unique(labels)):
    mask = labels == k
    ax.scatter(X_pca[mask,0], X_pca[mask,1], X_pca[mask,2], s=10, label=f"Cluster {k}")
ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")
ax.set_title(f"Clustering '{best_method}' sur PCA (K={best_K})")
ax.legend()
fig.tight_layout()
fig.savefig(OUT / "clusters_3d.png", dpi=160)
plt.close(fig)

# === 5) 6.i — Répartition des clients par cluster ===
size = df["cluster"].value_counts().sort_index()
size_pct = (size / len(df) * 100).round(2)
cluster_sizes = pd.DataFrame({"cluster": size.index, "count": size.values, "percent": size_pct.values})
cluster_sizes.to_csv(OUT / "cluster_sizes.csv", index=False)

# === 6) 6.ii — Dépense / Revenu par cluster ===
mnt_cols = [c for c in df.columns if str(c).startswith("Mnt")]
if mnt_cols:
    df["TotalSpend"] = df[mnt_cols].sum(axis=1)
else:
    # fallback générique
    maybe = [c for c in df.columns if any(k in str(c) for k in ["Spend","Amount","Mnt"])]
    df["TotalSpend"] = df[maybe].select_dtypes(include=[np.number]).sum(axis=1)

income_col = None
for name in ["Income","income","INCOME","Yearly_Income","Salary","Revenue"]:
    if name in df.columns:
        income_col = name
        break

# --- forcer en numérique les colonnes utilisées pour l'agg ---
cols = ["cluster", "TotalSpend"] + ([income_col] if income_col else [])
# Convertit toutes les colonnes sauf 'cluster' en numérique
for c in cols:
    if c != "cluster":
        df[c] = pd.to_numeric(df[c], errors="coerce")

# Agrégations (médiane / moyenne) uniquement sur des numériques
spend_income = df[cols].groupby("cluster").agg(["median", "mean"])

# Ratio dépense / revenu (médiane des ratios par cluster), si revenu dispo
if income_col:
    ratio = (
        df.groupby("cluster")
          .apply(lambda x: (x["TotalSpend"] / pd.to_numeric(x[income_col], errors="coerce").replace(0, np.nan)).median())
    )
    spend_income[("TotalSpend_to_Income", "median_ratio")] = ratio

# Aplatir les colonnes MultiIndex pour l’export
spend_income.columns = ["_".join([c for c in col if c]) for col in spend_income.columns]
spend_income = spend_income.reset_index()
spend_income.to_csv(OUT / "spend_income_by_cluster.csv", index=False)
 

# === 7) 6.iii — Répartition des clusters sur les campagnes ===
camp_cols = [c for c in df.columns if str(c).startswith("AcceptedCmp")]
if "Response" in df.columns:
    camp_cols += ["Response"]
camp_table = None
if camp_cols:
    for c in camp_cols:
        tab = df.groupby("cluster")[c].mean().rename(c+"_rate").reset_index()
        camp_table = tab if camp_table is None else camp_table.merge(tab, on="cluster", how="outer")

if camp_table is not None:
    camp_table.to_csv(OUT / "campaigns_by_cluster.csv", index=False)

# === 8) 6.iv — Déterminants de la dépense + Profils ===
# Pour éviter la fuite, on enlève les colonnes Mnt* des features explicatives
exp_cols = [c for c in base_feature_cols if not str(c).startswith("Mnt")]
X_exp_raw = df[exp_cols].copy().fillna(df[exp_cols].median(numeric_only=True))

# Ajouter Tenure si dispo et absente
if "Tenure_days" in df.columns and "Tenure_days" not in X_exp_raw.columns:
    X_exp_raw["Tenure_days"] = df["Tenure_days"].fillna(df["Tenure_days"].median())

# Supprimer les colonnes constantes
X_exp_raw = X_exp_raw.loc[:, X_exp_raw.nunique(dropna=False) > 1]

y = df["TotalSpend"].fillna(0).values

sc = StandardScaler()
X_exp = sc.fit_transform(X_exp_raw)

rf = RandomForestRegressor(n_estimators=400, random_state=42, n_jobs=-1)
rf.fit(X_exp, y)
imp = pd.Series(rf.feature_importances_, index=X_exp_raw.columns).sort_values(ascending=False)
top20 = imp.head(20).reset_index()
top20.columns = ["feature","importance"]
top20.to_csv(OUT / "rf_top20_importances.csv", index=False)

top_vars = top20["feature"].tolist()[:8]
cluster_profiles = df.groupby("cluster")[top_vars + ["TotalSpend"]].median().reset_index()
cluster_profiles.to_csv(OUT / "cluster_profiles.csv", index=False)

# === 9) SUMMARY: réponses simples 6.i — 6.iv ===
lines = []
lines.append("=== SUMMARY (réponses simples) ===")
lines.append(f"Meilleur clustering: {best_method} avec K={best_K} (silhouette={best['silhouette']:.3f})")
lines.append("")
lines.append("6.i — Répartition des clients par cluster:")
for _, row in cluster_sizes.iterrows():
    lines.append(f"  - Cluster {int(row['cluster'])}: {int(row['count'])} clients ({row['percent']}%)")
lines.append("")
lines.append("6.ii — Dépense/Revenu par cluster (voir spend_income_by_cluster.csv pour détails):")
if income_col:
    lines.append("  Indicateurs: TotalSpend_median, TotalSpend_mean, Income_median, Income_mean, TotalSpend_to_Income_median_ratio")
else:
    lines.append("  Indicateurs: TotalSpend_median, TotalSpend_mean (Income non détecté)")
lines.append("")
lines.append("6.iii — Campagnes: taux moyens par cluster (campaigns_by_cluster.csv)")
if camp_table is not None:
    lines.append("  Colonnes: " + ", ".join([c for c in camp_table.columns if c != "cluster"]))
else:
    lines.append("  Aucune colonne de campagne détectée.")
lines.append("")
lines.append("6.iv — Déterminants de la dépense (rf_top20_importances.csv), Profils (cluster_profiles.csv)")

(OUT / "SUMMARY.txt").write_text("\n".join(lines), encoding="utf-8")

print("\n=== Terminé ===")
print(f"Fichiers créés dans: {OUT.resolve()}")
