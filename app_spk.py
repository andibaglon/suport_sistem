# =========================================
# MODERN STREAMLIT UI - HEALTHCARE GREEN THEME
# =========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score

from imblearn.over_sampling import SMOTE

# =========================================
# PAGE CONFIG
# =========================================
st.set_page_config(
    page_title="SPK Prioritas Pasien",
    layout="wide",
    page_icon="🏥"
)

# =========================================
# CUSTOM CSS (MODERN HEALTH UI)
# =========================================
st.markdown("""
<style>
body {
    background-color: #f4fbf7;
}

.main {
    background: linear-gradient(135deg, #e8f5e9, #ffffff);
}

.block-container {
    padding: 2rem 2rem 2rem 2rem;
}

h1, h2, h3 {
    color: #1b5e20;
}

.stMetric {
    background-color: #e8f5e9;
    border-radius: 12px;
    padding: 10px;
}

.stButton>button {
    background-color: #2e7d32;
    color: white;
    border-radius: 10px;
    height: 45px;
    width: 100%;
    font-size: 16px;
}

.stButton>button:hover {
    background-color: #1b5e20;
}

.css-1d391kg {
    background-color: #f1f8f4;
}

.card {
    padding: 20px;
    border-radius: 15px;
    background-color: white;
    box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================
# DATA GENERATION
# =========================================
@st.cache_data
def generate_data():
    np.random.seed(42)
    n = 500

    df = pd.DataFrame({
        "umur": np.random.randint(1, 90, n),
        "tekanan_darah": np.random.randint(80, 180, n),
        "detak_jantung": np.random.randint(60, 140, n),
        "tingkat_darurat": np.random.choice([1,2,3], n),
        "riwayat_penyakit": np.random.choice([0,1], n),
        "kondisi_medis": np.random.choice(["ringan","sedang","berat"], n)
    })

    def rule_priority(r):
        score = 0
        if r.umur > 60: score += 2
        if r.tekanan_darah > 140: score += 2
        if r.detak_jantung > 110: score += 2
        if r.tingkat_darurat == 3: score += 3
        if r.riwayat_penyakit == 1: score += 1
        if r.kondisi_medis == "berat": score += 3

        if score >= 7: return 1
        elif score >= 4: return 2
        else: return 3

    df["prioritas"] = df.apply(rule_priority, axis=1)
    return df

# =========================================
# TRAIN MODEL
# =========================================
@st.cache_resource
def train_pipeline(df):
    X = df.drop("prioritas", axis=1)
    y = df["prioritas"]

    num_cols = ["umur", "tekanan_darah", "detak_jantung"]
    cat_cols = ["tingkat_darurat", "riwayat_penyakit", "kondisi_medis"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    model = Pipeline([
        ("prep", preprocessor),
        ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    model.fit(X_train, y_train)

    X_train_tr = model.named_steps['prep'].transform(X_train)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_tr, y_train)

    model.named_steps['clf'].fit(X_res, y_res)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc

# =========================================
# LOAD
# =========================================
df = generate_data()
model, acc = train_pipeline(df)

# =========================================
# HEADER
# =========================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.title("🏥 Sistem Pendukung Keputusan Prioritas Pasien")
st.caption("Pipeline + SMOTE + Decision Tree")
st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# METRICS
# =========================================
col1, col2, col3 = st.columns(3)
col1.metric("Akurasi Model", f"{acc:.2f}")
col2.metric("Jumlah Data", len(df))
col3.metric("Fitur", df.shape[1]-1)

# =========================================
# INPUT FORM (MODERN)
# =========================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🧾 Input Data Pasien")

c1, c2, c3 = st.columns(3)

with c1:
    umur = st.slider("Umur", 1, 100, 50)
    tekanan = st.slider("Tekanan Darah", 80, 200, 120)

with c2:
    detak = st.slider("Detak Jantung", 60, 150, 90)
    darurat = st.selectbox("Tingkat Darurat", [1,2,3])

with c3:
    riwayat = st.selectbox("Riwayat Penyakit", [0,1])
    kondisi = st.selectbox("Kondisi Medis", ["ringan","sedang","berat"])

predict_btn = st.button("🔍 Prediksi Sekarang")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# PREDICTION RESULT
# =========================================
if predict_btn:
    input_df = pd.DataFrame([{
        "umur": umur,
        "tekanan_darah": tekanan,
        "detak_jantung": detak,
        "tingkat_darurat": darurat,
        "riwayat_penyakit": riwayat,
        "kondisi_medis": kondisi
    }])

    pred = model.predict(input_df)[0]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("📌 Hasil Prediksi")

    if pred == 1:
        st.error("🔴 Prioritas Tinggi (Segera ditangani)")
    elif pred == 2:
        st.warning("🟡 Prioritas Sedang")
    else:
        st.success("🟢 Prioritas Rendah")

    st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# RANKING
# =========================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("🏆 Ranking Pasien")

df_pred = df.copy()
df_pred["prediksi"] = model.predict(df.drop("prioritas", axis=1))
ranking = df_pred.sort_values(by="prediksi")

st.dataframe(ranking.head(10), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# DATA SAMPLE
# =========================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("📊 Dataset Pasien")
st.dataframe(df.head(20), use_container_width=True)
st.markdown('</div>', unsafe_allow_html=True)
