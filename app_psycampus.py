import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
import numpy as np

# ============================================================
# 1. CONFIGURATION DE LA PAGE
# ============================================================
st.set_page_config(
    page_title="PsyCampus Pro",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# 2. DESIGN — Thème psychologie : violet profond + lavande
# ============================================================
st.markdown("""
<style>
/* ===== FOND GÉNÉRAL ===== */
.stApp {
    background: linear-gradient(160deg, #0d0d1a 0%, #1a0a2e 40%, #120820 100%);
    color: #E8E0FF;
}

/* ===== SIDEBAR ===== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1a0a2e 0%, #0d0d1a 100%);
    border-right: 1px solid rgba(147, 51, 234, 0.3);
}
[data-testid="stSidebar"] * {
    color: #E8E0FF !important;
}

/* ===== TITRES ===== */
h1, h2, h3, h4, h5, h6 {
    color: #C084FC !important;
    font-family: 'Georgia', serif !important;
    letter-spacing: 0.5px;
}

/* ===== TEXTE GÉNÉRAL ===== */
p, span, label, div {
    color: #D4C5F9 !important;
}

/* ===== MÉTRIQUES ===== */
[data-testid="metric-container"] {
    background: rgba(147, 51, 234, 0.12) !important;
    border: 1px solid rgba(192, 132, 252, 0.35) !important;
    border-radius: 14px !important;
    padding: 18px !important;
    box-shadow: 0 4px 20px rgba(147, 51, 234, 0.2) !important;
}
[data-testid="metric-container"] label {
    color: #A78BFA !important;
    font-size: 13px !important;
    letter-spacing: 0.8px !important;
    text-transform: uppercase !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #F0ABFC !important;
    font-size: 28px !important;
    font-weight: 700 !important;
}

/* ===== BOUTONS ===== */
.stButton > button {
    background: linear-gradient(135deg, #7C3AED, #9333EA) !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 25px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    letter-spacing: 0.5px !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(124, 58, 237, 0.4) !important;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #6D28D9, #7C3AED) !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(124, 58, 237, 0.6) !important;
}

/* ===== SLIDERS ===== */
.stSlider > div > div > div {
    background: linear-gradient(90deg, #7C3AED, #C084FC) !important;
}

/* ===== TABS ===== */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(147, 51, 234, 0.1) !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #A78BFA !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #7C3AED, #9333EA) !important;
    color: #FFFFFF !important;
}

/* ===== RADIO SIDEBAR ===== */
.stRadio > label {
    color: #A78BFA !important;
    font-weight: 600 !important;
}
.stRadio > div > label {
    color: #D4C5F9 !important;
}

/* ===== SÉPARATEUR ===== */
hr {
    border-color: rgba(192, 132, 252, 0.25) !important;
}

/* ===== ALERTES ===== */
.stWarning, .stError, .stInfo, .stSuccess {
    border-radius: 10px !important;
}

/* ===== CARD PERSONNALISÉE ===== */
.psycard {
    background: rgba(147, 51, 234, 0.1);
    border: 1px solid rgba(192, 132, 252, 0.3);
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 16px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.3);
}
.psycard h3 {
    margin-bottom: 8px;
}

/* ===== BADGE ===== */
.badge {
    display: inline-block;
    background: rgba(192, 132, 252, 0.2);
    border: 1px solid rgba(192, 132, 252, 0.5);
    color: #C084FC !important;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 13px;
    font-weight: 600;
    margin: 4px;
}
</style>
""", unsafe_allow_html=True)

# ============================================================
# 3. COULEURS PLOTLY (palette psychologie)
# ============================================================
COLORS = ["#C084FC", "#A855F7", "#7C3AED", "#9333EA",
          "#E879F9", "#D946EF", "#F0ABFC", "#6D28D9"]

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(13,13,26,0)",
    plot_bgcolor="rgba(26,10,46,0.4)",
    font=dict(color="#D4C5F9", family="Georgia, serif"),
    title_font=dict(color="#C084FC", size=18),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color="#D4C5F9")),
    xaxis=dict(gridcolor="rgba(192,132,252,0.15)", color="#A78BFA"),
    yaxis=dict(gridcolor="rgba(192,132,252,0.15)", color="#A78BFA"),
)

# ============================================================
# 4. URL DONNÉES
# ============================================================
URL_CSV = "https://docs.google.com/spreadsheets/d/e/2PACX-1vSY4fmMvLl3ktOK73KA_cqm2agYil3OxZkREM6GgnwUOU6-i5IYE3yGA8WWGVQps_5kGmoRjkYWHcQG/pub?gid=1441813994&single=true&output=csv"

# ============================================================
# 5. CHARGEMENT DES DONNÉES (robuste)
# ============================================================
@st.cache_data(ttl=60)
def load_data():
    try:
        data = pd.read_csv(URL_CSV)
        data.columns = [c.strip() for c in data.columns]

        # Mapping des colonnes du formulaire Google
        col_map = {
            "Combien d'heures de sommeil par nuit ?": "Sommeil",
            "Combien de fois faites-vous du sport par semaine ?": "Sport",
            "Niveau de stress (1 à 5)": "Stress",
            "Filière": "Filiere",
            "Filiere": "Filiere",
            "Genre": "Genre",
            "Année d'étude": "Annee",
        }
        data = data.rename(columns={k: v for k, v in col_map.items() if k in data.columns})

        # Colonnes numériques obligatoires
        for col in ["Sommeil", "Sport", "Stress"]:
            if col not in data.columns:
                data[col] = np.nan
            else:
                data[col] = data[col].astype(str).str.extract(r'(\d+\.?\d*)').astype(float)

        # Bien-être global (moyenne des colonnes contenant "Bien")
        cols_be = [c for c in data.columns if "Bien" in c or "bien" in c]
        if cols_be:
            for c in cols_be:
                data[c] = pd.to_numeric(data[c], errors="coerce")
            data["BienEtre_Global"] = data[cols_be].mean(axis=1)
        else:
            # Simulation si colonne absente : dérivé du stress inversé
            data["BienEtre_Global"] = (6 - data["Stress"]).clip(1, 5)

        # Stress catégoriel pour visualisations
        bins   = [0, 1.5, 2.5, 3.5, 4.5, 5.1]
        labels = ["Très faible", "Faible", "Modéré", "Élevé", "Très élevé"]
        data["Stress_Cat"] = pd.cut(
            data["Stress"], bins=bins, labels=labels, include_lowest=True
        )

        # Sommeil catégoriel
        data["Sommeil_Cat"] = pd.cut(
            data["Sommeil"],
            bins=[0, 5, 6, 7, 8, 24],
            labels=["< 5h", "5-6h", "6-7h", "7-8h", "> 8h"],
            include_lowest=True
        )

        return data

    except Exception as e:
        return None, str(e)


result = load_data()
if isinstance(result, tuple):
    df, error_msg = None, result[1]
else:
    df, error_msg = result, None

# ============================================================
# 6. EN-TÊTE PRINCIPAL
# ============================================================
st.markdown("""
<div style='text-align:center; padding: 20px 0 10px 0;'>
    <h1 style='font-size:42px; color:#C084FC; margin-bottom:4px;'>🧠 PsyCampus Analytics</h1>
    <p style='color:#A78BFA; font-size:16px; letter-spacing:1px;'>
        Plateforme d'analyse du bien-être étudiant
    </p>
</div>
""", unsafe_allow_html=True)
st.markdown("---")

# ============================================================
# 7. SIDEBAR NAVIGATION
# ============================================================
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding:16px 0;'>
        <div style='font-size:40px;'>🧠</div>
        <h2 style='color:#C084FC; font-size:20px; margin:8px 0 4px;'>PsyCampus</h2>
        <p style='color:#A78BFA; font-size:12px;'>Bien-être Étudiant</p>
    </div>
    <hr style='border-color:rgba(192,132,252,0.3);'>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "🏠  Tableau de bord",
        "📊  Analyses détaillées",
        "🔮  Prédictions & IA",
        "📝  Participer à l'étude"
    ], label_visibility="collapsed")

    st.markdown("<hr style='border-color:rgba(192,132,252,0.2);'>", unsafe_allow_html=True)

    if df is not None:
        st.markdown(f"""
        <div style='text-align:center;'>
            <span class='badge'>📋 {len(df)} réponses</span>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# 8. GESTION ERREUR CHARGEMENT
# ============================================================
if df is None:
    st.error(f"⚠️ Impossible de charger les données depuis Google Sheets.")
    st.markdown("""
    **Causes possibles :**
    - Le fichier Google Sheets n'est pas publié en CSV
    - L'URL a expiré ou les permissions ont changé
    - Problème de connexion réseau

    **Solution :** Vérifiez que votre Google Sheet est publié via :
    *Fichier → Partager → Publier sur le web → Format CSV*
    """)
    st.stop()

# ============================================================
# 9. PAGE : TABLEAU DE BORD
# ============================================================
if page == "🏠  Tableau de bord":

    st.subheader("Vue d'ensemble du bien-être étudiant")
    st.markdown(" ")

    # --- Métriques ---
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("👥 Participants",    f"{len(df)}")
    c2.metric("😴 Sommeil moyen",   f"{df['Sommeil'].mean():.1f} h")
    c3.metric("😰 Stress moyen",    f"{df['Stress'].mean():.1f} / 5")
    c4.metric("💚 Bien-être moyen", f"{df['BienEtre_Global'].mean():.1f} / 5")

    st.markdown(" ")
    st.markdown("---")

    # --- Graphique combiné résumé ---
    st.markdown("### 📈 Vue synthétique")

    col_l, col_r = st.columns(2)

    with col_l:
        # Scatter : Sommeil vs Stress
        fig_s = px.scatter(
            df.dropna(subset=["Sommeil", "Stress"]),
            x="Sommeil", y="Stress",
            color="Stress",
            color_continuous_scale=["#6D28D9", "#C084FC", "#F0ABFC"],
            size_max=14,
            title="Sommeil vs Niveau de stress",
            labels={"Sommeil": "Heures de sommeil", "Stress": "Stress (1-5)"},
            trendline="ols"
        )
        fig_s.update_layout(**PLOTLY_LAYOUT)
        fig_s.update_traces(marker=dict(size=8, opacity=0.75))
        st.plotly_chart(fig_s, use_container_width=True)

    with col_r:
        # Distribution bien-être
        fig_be = px.histogram(
            df.dropna(subset=["BienEtre_Global"]),
            x="BienEtre_Global",
            nbins=10,
            title="Distribution du Bien-être Global",
            labels={"BienEtre_Global": "Score de bien-être"},
            color_discrete_sequence=["#A855F7"]
        )
        fig_be.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_be, use_container_width=True)

# ============================================================
# 10. PAGE : ANALYSES DÉTAILLÉES
# ============================================================
elif page == "📊  Analyses détaillées":

    st.subheader("Analyses détaillées des réponses")
    st.markdown(" ")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🥧  Répartitions",
        "📊  Histogrammes",
        "📉  Courbes",
        "▬  Diagrammes en bandes",
        "🔗  Corrélations"
    ])

    # ---- TAB 1 : DIAGRAMMES CIRCULAIRES ----
    with tab1:
        st.markdown("### 🥧 Répartitions — Diagrammes circulaires")

        cols = st.columns(2)

        # Pie 1 : Répartition par filière
        with cols[0]:
            if "Filiere" in df.columns and df["Filiere"].notna().sum() > 0:
                fig_pie1 = px.pie(
                    df.dropna(subset=["Filiere"]),
                    names="Filiere",
                    title="Répartition par Filière",
                    color_discrete_sequence=COLORS,
                    hole=0.45
                )
                fig_pie1.update_layout(**PLOTLY_LAYOUT)
                fig_pie1.update_traces(
                    textposition="outside",
                    textinfo="percent+label",
                    pull=[0.04] * len(df["Filiere"].dropna().unique())
                )
                st.plotly_chart(fig_pie1, use_container_width=True)
            else:
                st.info("Colonne 'Filière' non disponible dans les données.")

        # Pie 2 : Catégories de stress
        with cols[1]:
            stress_counts = df["Stress_Cat"].value_counts().reset_index()
            stress_counts.columns = ["Niveau", "Nombre"]
            fig_pie2 = px.pie(
                stress_counts,
                names="Niveau",
                values="Nombre",
                title="Répartition des Niveaux de Stress",
                color_discrete_sequence=COLORS,
                hole=0.45
            )
            fig_pie2.update_layout(**PLOTLY_LAYOUT)
            fig_pie2.update_traces(
                textposition="outside",
                textinfo="percent+label"
            )
            st.plotly_chart(fig_pie2, use_container_width=True)

        # Pie 3 : Catégories de sommeil
        sommeil_counts = df["Sommeil_Cat"].value_counts().reset_index()
        sommeil_counts.columns = ["Plage", "Nombre"]
        fig_pie3 = px.pie(
            sommeil_counts,
            names="Plage",
            values="Nombre",
            title="Répartition des Plages de Sommeil",
            color_discrete_sequence=COLORS,
            hole=0.45
        )
        fig_pie3.update_layout(**PLOTLY_LAYOUT)
        fig_pie3.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(fig_pie3, use_container_width=True)

    # ---- TAB 2 : HISTOGRAMMES ----
    with tab2:
        st.markdown("### 📊 Histogrammes de distribution")

        col1, col2 = st.columns(2)

        with col1:
            fig_h1 = px.histogram(
                df.dropna(subset=["Stress"]),
                x="Stress",
                nbins=10,
                title="Distribution du Stress",
                labels={"Stress": "Niveau de stress (1-5)"},
                color_discrete_sequence=["#C084FC"]
            )
            fig_h1.update_layout(**PLOTLY_LAYOUT)
            fig_h1.update_traces(marker_line_color="#7C3AED", marker_line_width=1.5)
            st.plotly_chart(fig_h1, use_container_width=True)

            fig_h3 = px.histogram(
                df.dropna(subset=["BienEtre_Global"]),
                x="BienEtre_Global",
                nbins=10,
                title="Distribution du Bien-être Global",
                labels={"BienEtre_Global": "Score bien-être (1-5)"},
                color_discrete_sequence=["#A855F7"]
            )
            fig_h3.update_layout(**PLOTLY_LAYOUT)
            fig_h3.update_traces(marker_line_color="#6D28D9", marker_line_width=1.5)
            st.plotly_chart(fig_h3, use_container_width=True)

        with col2:
            fig_h2 = px.histogram(
                df.dropna(subset=["Sommeil"]),
                x="Sommeil",
                nbins=10,
                title="Distribution du Sommeil",
                labels={"Sommeil": "Heures de sommeil"},
                color_discrete_sequence=["#E879F9"]
            )
            fig_h2.update_layout(**PLOTLY_LAYOUT)
            fig_h2.update_traces(marker_line_color="#9333EA", marker_line_width=1.5)
            st.plotly_chart(fig_h2, use_container_width=True)

            fig_h4 = px.histogram(
                df.dropna(subset=["Sport"]),
                x="Sport",
                nbins=8,
                title="Distribution de la Pratique Sportive",
                labels={"Sport": "Séances de sport / semaine"},
                color_discrete_sequence=["#F0ABFC"]
            )
            fig_h4.update_layout(**PLOTLY_LAYOUT)
            fig_h4.update_traces(marker_line_color="#C084FC", marker_line_width=1.5)
            st.plotly_chart(fig_h4, use_container_width=True)

    # ---- TAB 3 : COURBES ----
    with tab3:
        st.markdown("### 📉 Courbes & tendances")

        # Courbe 1 : Évolution du stress par index (chronologique)
        df_sorted = df.copy().reset_index()
        fig_c1 = go.Figure()
        fig_c1.add_trace(go.Scatter(
            x=df_sorted.index,
            y=df_sorted["Stress"].rolling(window=5, min_periods=1).mean(),
            mode="lines",
            name="Stress (moyenne mobile)",
            line=dict(color="#C084FC", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(192,132,252,0.12)"
        ))
        fig_c1.add_trace(go.Scatter(
            x=df_sorted.index,
            y=df_sorted["BienEtre_Global"].rolling(window=5, min_periods=1).mean(),
            mode="lines",
            name="Bien-être (moyenne mobile)",
            line=dict(color="#4ADE80", width=2.5),
            fill="tozeroy",
            fillcolor="rgba(74,222,128,0.08)"
        ))
        fig_c1.update_layout(
            **PLOTLY_LAYOUT,
            title="Évolution du Stress et du Bien-être (moyenne mobile sur 5)",
            xaxis_title="Répondants (ordre chronologique)",
            yaxis_title="Score (1-5)"
        )
        st.plotly_chart(fig_c1, use_container_width=True)

        col_c1, col_c2 = st.columns(2)

        # Courbe 2 : Sommeil vs Bien-être (scatter + ligne de tendance)
        with col_c1:
            df_c2 = df.dropna(subset=["Sommeil", "BienEtre_Global"])
            fig_c2 = px.scatter(
                df_c2, x="Sommeil", y="BienEtre_Global",
                trendline="ols",
                title="Sommeil → Bien-être (tendance)",
                labels={"Sommeil": "Heures de sommeil", "BienEtre_Global": "Bien-être"},
                color_discrete_sequence=["#A855F7"]
            )
            fig_c2.update_layout(**PLOTLY_LAYOUT)
            fig_c2.update_traces(marker=dict(size=7, opacity=0.7))
            st.plotly_chart(fig_c2, use_container_width=True)

        # Courbe 3 : Sport vs Stress
        with col_c2:
            df_c3 = df.dropna(subset=["Sport", "Stress"])
            fig_c3 = px.scatter(
                df_c3, x="Sport", y="Stress",
                trendline="ols",
                title="Sport → Stress (tendance)",
                labels={"Sport": "Séances sport/semaine", "Stress": "Niveau de stress"},
                color_discrete_sequence=["#E879F9"]
            )
            fig_c3.update_layout(**PLOTLY_LAYOUT)
            fig_c3.update_traces(marker=dict(size=7, opacity=0.7))
            st.plotly_chart(fig_c3, use_container_width=True)

    # ---- TAB 4 : DIAGRAMMES EN BANDES ----
    with tab4:
        st.markdown("### ▬ Diagrammes en bandes (barres)")

        # Barres 1 : Stress moyen par filière
        if "Filiere" in df.columns and df["Filiere"].notna().sum() > 0:
            stress_filiere = (
                df.dropna(subset=["Filiere", "Stress"])
                  .groupby("Filiere")["Stress"]
                  .mean()
                  .reset_index()
                  .sort_values("Stress", ascending=False)
            )
            fig_b1 = px.bar(
                stress_filiere,
                x="Filiere", y="Stress",
                title="Stress moyen par Filière",
                labels={"Filiere": "Filière", "Stress": "Stress moyen"},
                color="Stress",
                color_continuous_scale=["#6D28D9", "#C084FC", "#F0ABFC"]
            )
            fig_b1.update_layout(**PLOTLY_LAYOUT)
            fig_b1.update_traces(marker_line_color="rgba(192,132,252,0.5)", marker_line_width=1)
            st.plotly_chart(fig_b1, use_container_width=True)

        col_b1, col_b2 = st.columns(2)

        # Barres 2 : Sommeil moyen par catégorie de stress
        with col_b1:
            sommeil_stress = (
                df.dropna(subset=["Stress_Cat", "Sommeil"])
                  .groupby("Stress_Cat", observed=True)["Sommeil"]
                  .mean()
                  .reset_index()
            )
            fig_b2 = px.bar(
                sommeil_stress,
                x="Stress_Cat", y="Sommeil",
                title="Sommeil moyen par niveau de stress",
                labels={"Stress_Cat": "Niveau de stress", "Sommeil": "Sommeil moyen (h)"},
                color="Sommeil",
                color_continuous_scale=["#F0ABFC", "#7C3AED"]
            )
            fig_b2.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_b2, use_container_width=True)

        # Barres 3 : Bien-être moyen par plage de sommeil
        with col_b2:
            be_sommeil = (
                df.dropna(subset=["Sommeil_Cat", "BienEtre_Global"])
                  .groupby("Sommeil_Cat", observed=True)["BienEtre_Global"]
                  .mean()
                  .reset_index()
            )
            fig_b3 = px.bar(
                be_sommeil,
                x="Sommeil_Cat", y="BienEtre_Global",
                title="Bien-être moyen par plage de sommeil",
                labels={"Sommeil_Cat": "Plage de sommeil", "BienEtre_Global": "Bien-être moyen"},
                color="BienEtre_Global",
                color_continuous_scale=["#6D28D9", "#4ADE80"]
            )
            fig_b3.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_b3, use_container_width=True)

    # ---- TAB 5 : CORRÉLATIONS ----
    with tab5:
        st.markdown("### 🔗 Matrice de corrélations")

        num_cols = ["Sommeil", "Sport", "Stress", "BienEtre_Global"]
        available = [c for c in num_cols if c in df.columns and df[c].notna().sum() > 3]

        if len(available) >= 2:
            corr_matrix = df[available].corr()

            fig_heat = go.Figure(data=go.Heatmap(
                z=corr_matrix.values,
                x=corr_matrix.columns,
                y=corr_matrix.columns,
                colorscale=[[0, "#1a0a2e"], [0.5, "#7C3AED"], [1, "#F0ABFC"]],
                text=np.round(corr_matrix.values, 2),
                texttemplate="%{text}",
                showscale=True,
                zmin=-1, zmax=1
            ))
            fig_heat.update_layout(
                **PLOTLY_LAYOUT,
                title="Corrélations entre les variables"
            )
            st.plotly_chart(fig_heat, use_container_width=True)

            # Interprétation automatique
            st.markdown("#### Interprétation")
            for i, col_a in enumerate(available):
                for col_b in available[i+1:]:
                    val = corr_matrix.loc[col_a, col_b]
                    if abs(val) >= 0.4:
                        direction = "positive" if val > 0 else "négative"
                        force = "forte" if abs(val) >= 0.6 else "modérée"
                        st.markdown(
                            f"<span class='badge'>r = {val:.2f}</span> "
                            f"Corrélation **{force} {direction}** entre **{col_a}** et **{col_b}**",
                            unsafe_allow_html=True
                        )

# ============================================================
# 11. PAGE : PRÉDICTIONS
# ============================================================
elif page == "🔮  Prédictions & IA":

    st.subheader("Prédiction du niveau de stress par IA")
    st.markdown(" ")

    df_clean = df.dropna(subset=["Sommeil", "Sport", "Stress"])

    if len(df_clean) < 10:
        st.warning("⚠️ Pas assez de données pour entraîner le modèle (minimum 10 réponses).")
    else:
        # Entraînement
        X = df_clean[["Sommeil", "Sport"]]
        y = df_clean["Stress"]
        model = LinearRegression().fit(X, y)
        score = model.score(X, y)

        col_info, col_pred = st.columns([1, 2])

        with col_info:
            st.markdown("""
            <div class='psycard'>
                <h3>📐 Modèle</h3>
                <p>Régression Linéaire Multiple</p>
                <p><b>Variables :</b> Sommeil + Sport</p>
                <p><b>Cible :</b> Niveau de stress</p>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Score R²", f"{score:.3f}", help="Plus proche de 1 = meilleur modèle")
            st.metric("Données d'entraînement", f"{len(df_clean)} réponses")

        with col_pred:
            st.markdown("#### Entrez vos paramètres")
            s_sleep = st.slider("🛌 Heures de sommeil par nuit", 3, 12, 7)
            s_sport = st.slider("🏃 Séances de sport par semaine", 0, 7, 2)

            pred = model.predict([[s_sleep, s_sport]])[0]
            pred = round(max(1.0, min(5.0, pred)), 2)

            # Jauge visuelle
            color_gauge = "#4ADE80" if pred < 2.5 else "#FBBF24" if pred < 3.5 else "#F87171"
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=pred,
                title={"text": "Stress estimé", "font": {"color": "#C084FC", "size": 18}},
                gauge={
                    "axis": {"range": [1, 5], "tickcolor": "#A78BFA"},
                    "bar": {"color": color_gauge},
                    "bgcolor": "rgba(26,10,46,0.5)",
                    "steps": [
                        {"range": [1, 2.5], "color": "rgba(74,222,128,0.15)"},
                        {"range": [2.5, 3.5], "color": "rgba(251,191,36,0.15)"},
                        {"range": [3.5, 5], "color": "rgba(248,113,113,0.15)"},
                    ],
                    "threshold": {"line": {"color": "white", "width": 3}, "value": pred}
                },
                number={"suffix": " / 5", "font": {"color": "#F0ABFC", "size": 30}}
            ))
            fig_gauge.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#D4C5F9"),
                height=280
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Conseil personnalisé
            if pred <= 2:
                st.success("😌 Stress très faible — Votre équilibre sommeil/sport semble excellent !")
            elif pred <= 3:
                st.info("🙂 Stress modéré — Quelques ajustements pourraient améliorer votre bien-être.")
            elif pred <= 4:
                st.warning("😟 Stress élevé — Envisagez d'améliorer votre hygiène de sommeil et votre activité physique.")
            else:
                st.error("😰 Stress très élevé — Il serait utile de consulter un professionnel de santé.")

# ============================================================
# 12. PAGE : PARTICIPER
# ============================================================
elif page == "📝  Participer à l'étude":

    st.subheader("Contribuez à la recherche")
    st.markdown(" ")

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("""
        <div class='psycard'>
            <h3>🎯 Objectif de l'étude</h3>
            <p>Cette étude analyse les liens entre le sommeil, l'activité physique et le bien-être psychologique des étudiants universitaires.</p>
            <br/>
            <p>Vos réponses sont <b>anonymes</b> et contribuent à mieux comprendre la santé mentale sur le campus.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='psycard'>
            <h3>⏱️ Durée</h3>
            <p>2 à 3 minutes seulement</p>
            <br/>
            <h3>🔒 Confidentialité</h3>
            <p>Données anonymes — aucune information personnelle collectée</p>
        </div>
        """, unsafe_allow_html=True)

    with col_right:
        st.markdown("""
        <div class='psycard' style='text-align:center; padding:40px;'>
            <div style='font-size:60px; margin-bottom:16px;'>📋</div>
            <h3>Répondre au questionnaire</h3>
            <p style='margin-bottom:24px; color:#A78BFA;'>Cliquez ci-dessous pour accéder au formulaire Google</p>
        </div>
        """, unsafe_allow_html=True)

        # Lien compatible avec toutes versions de Streamlit
        FORM_URL = "https://docs.google.com/forms/d/e/1FAIpQLSeu6HCHiWj4Y0e9wH5YvjPZ_15X8Xv-u8S_z_j-h_uS-x-u-w/viewform"
        st.markdown(
            f"""
            <div style='text-align:center; margin-top:16px;'>
                <a href='{FORM_URL}' target='_blank'
                   style='background:linear-gradient(135deg,#7C3AED,#9333EA);
                          color:white; text-decoration:none;
                          padding:14px 36px; border-radius:25px;
                          font-weight:600; font-size:16px;
                          box-shadow:0 4px 15px rgba(124,58,237,0.5);
                          display:inline-block;'>
                    📝 Ouvrir le formulaire
                </a>
            </div>
            """,
            unsafe_allow_html=True
        )

# ============================================================
# 13. FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div style='text-align:center; padding:16px; color:#6B7280; font-size:13px;'>
    🧠 PsyCampus Analytics — Étude sur le bien-être étudiant &nbsp;|&nbsp;
    Données anonymes &nbsp;|&nbsp; Mis à jour toutes les 60s
</div>
""", unsafe_allow_html=True)
