import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.statespace.sarimax import SARIMAX

# =====================================================
# CONFIGURATION GÉNÉRALE
# =====================================================
st.set_page_config(page_title="Analyse des données INSTAT ", layout="wide")

# =====================================================
# CHARGEMENT DES DONNÉES
# =====================================================
@st.cache_data
def load_data():
    fichier = "/home/anjara/INSTAT/données_anjara_gid_ANALYSE.xlsx"
    return pd.read_excel(fichier)

try:
    df = load_data()
    st.session_state["df"] = df
except Exception as e:
    st.error(f"❌ Erreur chargement données : {e}")
    st.stop()



# =====================================================
# MENU DE NAVIGATION
# =====================================================
st.sidebar.title("📌 Menu Principale")
menu = st.sidebar.radio(
    "Menu",
    (
        "🏠 Accueil",
        "📂 Données",
        "📈 Prédiction annuelle",
        "📉 Prévision mensuelle SARIMA",
        "📊 Visualisations"
    )
)

# =====================================================
# 🏠 ACCUEIL MODERNE
# =====================================================
if menu == "🏠 Accueil":
    st.title("📊 TABLEAU DE BORD INSTAT – 2013-2025")
    
    # --- Statistiques globales ---
    total_entreprises = len(df)
    total_fonds = df["FONDS"].sum() if "FONDS" in df.columns else 0
    total_villes = df["FIVON"].nunique() if "FIVON" in df.columns else 0
    last_year = pd.to_datetime(df["DTCRE"], errors="coerce").dt.year.max()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📈 Total entreprises", f"{total_entreprises:,}")
    col2.metric("💰 Fonds total (MGF)", f"{total_fonds:,.0f}")
    col3.metric("🏙 Fivondronana actives", f"{total_villes}")
    col4.metric("📅 Dernière année", f"{int(last_year)}")
    
    st.markdown("---")


     # --- Navigation rapide ---
    st.subheader("ETAPE A SUIVRE")
    col1, col2, col3, col4 = st.columns(4)
    if col1.button("Voir données brutes"):
        st.session_state["menu"] = "📂 Données"
    if col2.button("Prédiction annuelle"):
        st.session_state["menu"] = "📈 Prédiction annuelle"
    if col3.button("Prévision mensuelle"):
        st.session_state["menu"] = "📉 Prévision mensuelle SARIMA"
    if col4.button("Voir Visualisations"):
        st.session_state["menu"] = "📉 Visualisations"



    
    # =====================================================
# 📈 ÉVOLUTION ANNUELLE – BARRES + CIRCULAIRE
# =====================================================
    st.subheader("📈 Créations d'entreprises par année")

# Préparation des données (UNE SEULE FOIS)
    df["DTCRE"] = pd.to_datetime(df["DTCRE"], errors="coerce")

    df_year = (
        df.dropna(subset=["DTCRE"])
            .assign(annee=lambda x: x["DTCRE"].dt.year)
            .groupby("annee")
            .size()
            .reset_index(name="nb_entreprises")
)

# Création de deux colonnes horizontales
    col1, col2 = st.columns(2)

# =====================================================
# 📊 GRAPHIQUE EN BARRES (GAUCHE)
# =====================================================
    with col1:
        st.markdown("### 📊 Évolution annuelle (Barres)")
    
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        sns.barplot(
            data=df_year,
            x="annee",
            y="nb_entreprises",
            palette="viridis",
            ax=ax1
    )
        ax1.set_xlabel("Année")
        ax1.set_ylabel("Nombre d'entreprises")
        ax1.set_title("Créations par année")
        ax1.grid(axis="y", linestyle="--", alpha=0.6)
        st.pyplot(fig1)

# =====================================================
# 🟢 GRAPHIQUE CIRCULAIRE (DROITE)
# =====================================================
    with col2:
        st.markdown("### 🟢 Répartition annuelle (Circulaire)")
    
        fig2, ax2 = plt.subplots(figsize=(6, 6))
        ax2.pie(
            df_year["nb_entreprises"],
            labels=df_year["annee"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": 0.4}  # Donut moderne
    )
        ax2.set_title("Répartition des créations")
        st.pyplot(fig2)


   
    

# =====================================================
# 📂 DONNÉES
# =====================================================
elif menu == "📂 Données":
    st.title("📂 Nouvelle Données")
    st.dataframe(df)

# =====================================================
# 📈 PRÉDICTION ANNUELLE — RÉGRESSION LINÉAIRE
# =====================================================
elif menu == "📈 Prédiction annuelle":
    st.title("📈 Prédiction annuelle — Régression linéaire")
    
    df["DTCRE"] = pd.to_datetime(df["DTCRE"], errors="coerce")
    df_year = (
        df.dropna(subset=["DTCRE"])
          .assign(annee=lambda x: x["DTCRE"].dt.year)
          .groupby("annee")
          .size()
          .reset_index(name="nb_entreprises")
    )
    
    # Ajouter toutes les années pour la continuité
    all_years = pd.DataFrame({'annee': np.arange(df_year['annee'].min(), df_year['annee'].max()+1)})
    df_year = all_years.merge(df_year, on='annee', how='left').fillna(0)
    
    # Régression linéaire
    X = (df_year[["annee"]] - df_year["annee"].min())
    y = df_year["nb_entreprises"]
    model = LinearRegression()
    model.fit(X, y)
    df_year["prediction"] = model.predict(X)
    
    # Prévisions futures
    future_years = np.arange(df_year["annee"].max()+1, df_year["annee"].max()+6)
    X_future = (future_years - df_year["annee"].min()).reshape(-1,1)
    future_pred = model.predict(X_future)
    df_future = pd.DataFrame({"annee": future_years, "prediction": future_pred})
    
    # Performance
    r2 = r2_score(y, df_year["prediction"])
    mae = mean_absolute_error(y, df_year["prediction"])
    rmse = np.sqrt(mean_squared_error(y, df_year["prediction"]))
    
    # Graphique
    st.subheader("📊 Réel vs Régression + Prévisions")
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_year["annee"], df_year["nb_entreprises"], marker="o", label="Réel")
    ax.plot(df_year["annee"], df_year["prediction"], linestyle="--", label="Régression")
    ax.plot(df_future["annee"], df_future["prediction"], marker="x", color="red", label="Prévision")
    ax.set_xlabel("Année")
    ax.set_ylabel("Nombre d'entreprises")
    ax.set_title("Réel vs Prédiction")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    

    
    st.subheader("📊 Prévisions futures")
    st.dataframe(df_future)

# =====================================================
# 📉 PRÉVISION MENSUELLE — SARIMA
# =====================================================
elif menu == "📉 Prévision mensuelle SARIMA":
    st.title("📉 Prévision mensuelle — SARIMA")
    
    df["DTCRE"] = pd.to_datetime(df["DTCRE"], errors="coerce")
    df_month = df.dropna(subset=["DTCRE"]).set_index("DTCRE").resample("M").size()
    
    model = SARIMAX(
        df_month,
        order=(1,1,1),
        seasonal_order=(1,1,1,12),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    results = model.fit(disp=False)
    
    forecast = results.get_forecast(steps=12)
    forecast_mean = forecast.predicted_mean
    
    fitted = results.fittedvalues
    mae = mean_absolute_error(df_month[1:], fitted[1:])
    rmse = np.sqrt(mean_squared_error(df_month[1:], fitted[1:]))
    
    
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(df_month, label="Observé")
    ax.plot(forecast_mean, color="red", label="Prévision 12 mois")
    ax.set_title("Série réelle et prévision SARIMA")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)
    
    st.subheader("📊 Prévisions mensuelles")
    st.dataframe(forecast_mean.rename("Prévision"))

# =====================================================
# 📊 VISUALISATIONS
# =====================================================
elif menu == "📊 Visualisations":
    st.title("📊 Visualisations")
    
    vis = st.selectbox(
        "Choisir une analyse :",
        (
            "Activités dominantes / moins dominantes",
            "Répartition géographique",
            "Analyse des événements",
            "Analyse des fonds",
            "Nationalité MG vs Étranger",
            "Forme juridique (LFJ)",
            "Capital moyen ACTPR × LFJ",
            "Localisation inconnue",
            "Répartition des entreprises par DFOKON",
            "Top 10 activités par Fivondronana (Heatmap)",
            "Top 10 activités par Fokontany (Heatmap)",
            "Activités moins fréquentes par Fivondronana (Heatmap)"

        )
    )
    
    df = st.session_state["df"]
    
    # =================================================
    # 🏭 ACTIVITÉS DOMINANTES & MOINS DOMINANTES
    # =================================================
    if vis == "Activités dominantes / moins dominantes":

        st.subheader("🏭 Répartition des activités")

        top_n = st.slider("Nombre d'activités à afficher", 5, 15, 10)

        df_act = (
            df["ACTPR"]
            .fillna("INCONNU")
            .value_counts()
            .reset_index()
        )
        df_act.columns = ["Activité", "Nombre"]

        col1, col2 = st.columns(2)

        # 🔝 Dominantes
        with col1:
            st.markdown("### 🔝 Activités dominantes")
            df_top = df_act.head(top_n)
            autres = df_act["Nombre"].sum() - df_top["Nombre"].sum()
            df_pie = pd.concat(
                [df_top, pd.DataFrame({"Activité": ["Autres"], "Nombre": [autres]})]
            )

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(
                df_pie["Nombre"],
                labels=df_pie["Activité"],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"width": 0.4},
            )
            ax.set_title("Activités dominantes")
            st.pyplot(fig)

        # 🔻 Moins dominantes
        with col2:
            st.markdown("### 🔻 Activités moins dominantes")
            df_bottom = df_act.tail(top_n)

            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie(
                df_bottom["Nombre"],
                labels=df_bottom["Activité"],
                autopct="%1.1f%%",
                startangle=90,
                wedgeprops={"width": 0.4},
            )
            ax.set_title("Activités moins dominantes")
            st.pyplot(fig)

        with st.expander("📋 Tableau détaillé des activités"):
            st.dataframe(df_act)

    # =================================================
    # 📍 DFOKON — DONUT
    # =================================================
    elif vis == "Répartition des entreprises par DFOKON":

        st.subheader("📍 Répartition des entreprises par Fokontany")

        top_n = st.slider("Top Fokontany", 5, 20, 10)

        df_dfokon = (
            df["DFOKON"]
            .fillna("INCONNU")
            .value_counts()
            .reset_index()
        )
        df_dfokon.columns = ["DFOKON", "Nombre"]

        df_top = df_dfokon.head(top_n)
        autres = df_dfokon["Nombre"].sum() - df_top["Nombre"].sum()

        df_pie = pd.concat(
            [df_top, pd.DataFrame({"DFOKON": ["Autres"], "Nombre": [autres]})]
        )

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.pie(
            df_pie["Nombre"],
            labels=df_pie["DFOKON"],
            autopct="%1.1f%%",
            startangle=90,
            wedgeprops={"width": 0.45},
        )
        ax.set_title("Répartition des entreprises par DFOKON")
        st.pyplot(fig)

        with st.expander("📋 Détails par DFOKON"):
            st.dataframe(df_dfokon)

    
    elif vis == "Répartition géographique":
        with st.expander("🏙 Par ville (FIVON)"):
            rep = df["FIVON"].value_counts().reset_index()
            rep.columns = ["FIVON","Nombre"]
            fig, ax = plt.subplots(figsize=(12,5))
            sns.barplot(data=rep, x="FIVON", y="Nombre", palette="coolwarm", ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
            st.pyplot(fig)


            # 📌 ANALYSE DES ÉVÉNEMENTS
    # =================================================
    if vis == "Analyse des événements":

        st.subheader("📌 Analyse des événements (TPMAJ par année)")

        df["DTMAJ"] = pd.to_datetime(df["DTMAJ"], errors="coerce")
        df["annee_maj"] = df["DTMAJ"].dt.year

        tab = pd.crosstab(df["annee_maj"], df["TPMAJ"])
        st.dataframe(tab)

        fig, ax = plt.subplots(figsize=(12, 6))
        tab.plot(kind="bar", ax=ax)
        ax.set_xlabel("Année")
        ax.set_ylabel("Nombre d'événements")
        ax.set_title("Événements par année")
        ax.grid(True)
        st.pyplot(fig)

    # =================================================
    # 💰 ANALYSE DES FONDS
    # =================================================
    elif vis == "Analyse des fonds":

        st.subheader("💰 Évolution du capital moyen par année")

        df["FONDS"] = pd.to_numeric(df["FONDS"], errors="coerce").fillna(0)
        df["DTCRE"] = pd.to_datetime(df["DTCRE"], errors="coerce")
        df["annee"] = df["DTCRE"].dt.year

        capital = df.groupby("annee")["FONDS"].mean().reset_index()

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(data=capital, x="annee", y="FONDS", marker="o", ax=ax)
        ax.set_xlabel("Année")
        ax.set_ylabel("Capital moyen")
        ax.set_title("Capital moyen des entreprises par année")
        ax.grid(True)
        st.pyplot(fig)

        st.dataframe(capital)

    # =================================================
    # 🌍 NATIONALITÉ
    # =================================================
    elif vis == "Nationalité MG vs Étranger":

        st.subheader("🌍 Répartition par nationalité")

        df["NAT"] = df["LNAT"].apply(
            lambda x: "MG" if str(x).upper() == "MG" else "Étranger"
        )

        counts = df["NAT"].value_counts().reset_index()
        counts.columns = ["Nationalité", "Nombre"]

        fig, ax = plt.subplots(figsize=(6, 4))
        sns.barplot(data=counts, x="Nationalité", y="Nombre", ax=ax)
        ax.set_title("Nationalité des entreprises")
        ax.grid(True)
        st.pyplot(fig)

        st.dataframe(counts)

    # =================================================
    # 🏢 FORME JURIDIQUE
    # =================================================
    elif vis == "Forme juridique (LFJ)":

        st.subheader("🏢 Répartition par forme juridique")

        lfj = df["LFJ"].value_counts().reset_index()
        lfj.columns = ["Forme juridique", "Nombre"]

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(data=lfj, x="Forme juridique", y="Nombre", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Répartition des entreprises par forme juridique")
        ax.grid(True)
        st.pyplot(fig)

        st.dataframe(lfj)

    # =================================================
    # 💼 CAPITAL MOYEN ACTPR × LFJ
    # =================================================
    elif vis == "Capital moyen ACTPR × LFJ":

        st.subheader("💼 Capital moyen par activité et forme juridique")

        df["FONDS"] = pd.to_numeric(df["FONDS"], errors="coerce").fillna(0)

        pivot = df.pivot_table(
            values="FONDS",
            index="ACTPR",
            columns="LFJ",
            aggfunc="mean"
        )

        fig, ax = plt.subplots(figsize=(18, 8))
        sns.heatmap(pivot, cmap="YlGnBu", linewidths=0.5)
        ax.set_title("Capital moyen (ACTPR × LFJ)")
        st.pyplot(fig)

        st.dataframe(pivot)

    # =================================================
    # ❓ LOCALISATION INCONNUE
    # =================================================
    elif vis == "Localisation inconnue":

        st.subheader("❓ Entreprises avec localisation inconnue")

        inconnu = (
            df[df["DFOKON"] == "INCONNU"]["LFJ"]
            .value_counts()
            .reset_index()
        )
        inconnu.columns = ["Forme juridique", "Nombre"]

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(data=inconnu, x="Forme juridique", y="Nombre", ax=ax)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
        ax.set_title("Forme juridique – Localisation inconnue")
        ax.grid(True)
        st.pyplot(fig)

        st.dataframe(inconnu)


       
            # =================================================
# 🔥 TOP 10 ACTIVITÉS PAR FIVONDRONANA (HEATMAP)
# =================================================
    elif vis == "Top 10 activités par Fivondronana (Heatmap)":

        st.subheader("🔥 Répartition des 10 principales activités par Fivondronana")

        # Vérification des colonnes nécessaires
        required_cols = {"ACTPR", "FIVON"}
        if not required_cols.issubset(df.columns):
            st.error("Colonnes ACTPR ou FIVON manquantes dans les données.")
            st.stop()

        # 1️⃣ Identifier les 10 activités les plus fréquentes
        top_10_actpr = df["ACTPR"].value_counts().nlargest(10).index

        # 2️⃣ Créer la table croisée FIVON × ACTPR
        activite_par_fivon = pd.crosstab(
            df["FIVON"],
            df["ACTPR"]
        )

        # 3️⃣ Garder uniquement les 10 activités dominantes
        activite_heatmap = activite_par_fivon[top_10_actpr]

        # 4️⃣ Graphique Heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            activite_heatmap,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            linewidths=0.5,
            cbar_kws={"label": "Nombre d'entreprises"},
            ax=ax
        )

        ax.set_title(
            "Répartition des 10 principales activités par Fivondronana",
            fontsize=16
        )
        ax.set_xlabel("Activité principale (ACTPR)")
        ax.set_ylabel("Fivondronana")

        st.pyplot(fig)

        # 5️⃣ Tableau des données
        with st.expander("📋 Données détaillées (FIVON × ACTPR)"):
            st.dataframe(activite_heatmap)

                # =================================================
    # 🔥 TOP 10 ACTIVITÉS × TOP 15 FOKONTANY (HEATMAP)
    # =================================================
    elif vis == "Top 10 activités par Fokontany (Heatmap)":

        st.subheader("🔥 Répartition des 10 principales activités par les 15 Fokontany les plus actifs")

        # Vérification des colonnes
        required_cols = {"DFOKON", "ACTPR"}
        if not required_cols.issubset(df.columns):
            st.error("Colonnes DFOKON ou ACTPR manquantes.")
            st.stop()

        # Top 10 activités
        top_10_actpr = df["ACTPR"].value_counts().nlargest(10).index

        # Top 15 Fokontany
        top_15_dfokon = df["DFOKON"].value_counts().nlargest(15).index

        # Table croisée DFOKON × ACTPR
        activite_par_dfokon = pd.crosstab(
            df["DFOKON"],
            df["ACTPR"]
        )

        heatmap_data_dfokon = activite_par_dfokon.loc[top_15_dfokon, top_10_actpr]

        # Heatmap
        fig, ax = plt.subplots(figsize=(16, 10))
        sns.heatmap(
            heatmap_data_dfokon,
            annot=True,
            fmt="d",
            cmap="YlOrRd",
            linewidths=0.5,
            cbar_kws={"label": "Nombre d'entreprises"},
            ax=ax
        )

        ax.set_title("Top 10 activités par les 15 Fokontany les plus actifs")
        ax.set_xlabel("Activité principale (ACTPR)")
        ax.set_ylabel("Fokontany (DFOKON)")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")

        st.pyplot(fig)

        with st.expander("📋 Données détaillées"):
            st.dataframe(heatmap_data_dfokon)


                # =================================================
    # 🔻 ACTIVITÉS MOINS FRÉQUENTES × FIVONDRONANA (HEATMAP)
    # =================================================
    elif vis == "Activités moins fréquentes par Fivondronana (Heatmap)":

        st.subheader("🔻 Répartition des 10 activités les moins fréquentes par Fivondronana")

        # Vérification des colonnes
        required_cols = {"FIVON", "ACTPR"}
        if not required_cols.issubset(df.columns):
            st.error("Colonnes FIVON ou ACTPR manquantes.")
            st.stop()

        # Identifier les 10 activités les moins fréquentes
        bottom_10_actpr = df["ACTPR"].value_counts().nsmallest(10).index

        # Table croisée FIVON × ACTPR
        activite_par_fivon = pd.crosstab(
            df["FIVON"],
            df["ACTPR"]
        )

        activite_heatmap_bottom10 = activite_par_fivon[bottom_10_actpr]

        # Heatmap
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.heatmap(
            activite_heatmap_bottom10,
            annot=True,
            fmt="d",
            cmap="Reds",
            linewidths=0.5,
            cbar_kws={"label": "Nombre d'entreprises (faible)"},
            ax=ax
        )

        ax.set_title("10 activités les moins fréquentes par Fivondronana")
        ax.set_xlabel("Activité principale (ACTPR – moins fréquentes)")
        ax.set_ylabel("Fivondronana")

        st.pyplot(fig)

        with st.expander("📋 Données détaillées"):
            st.dataframe(activite_heatmap_bottom10)









