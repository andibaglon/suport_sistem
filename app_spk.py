# =========================================
# 1. IMPORT
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
# 2. GENERATE DATASET (REALISTIS RS)
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
# 3. TRAIN PIPELINE (ANTI ERROR)
# =========================================
@st.cache_resource
def train_pipeline(df):
    X = df.drop("prioritas", axis=1)
    y = df["prioritas"]

    # Pisahkan kolom
    num_cols = ["umur", "tekanan_darah", "detak_jantung"]
    cat_cols = ["tingkat_darurat", "riwayat_penyakit", "kondisi_medis"]

    # Preprocessing aman
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_cols)
    ])

    # Model pipeline
    model = Pipeline([
        ("prep", preprocessor),
        ("clf", DecisionTreeClassifier(max_depth=5, random_state=42))
    ])

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Fit awal
    model.fit(X_train, y_train)

    # ======================
    # OPTIONAL: SMOTE
    # ======================
    # NOTE: SMOTE harus setelah transform → manual
    X_train_tr = model.named_steps['prep'].transform(X_train)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_tr, y_train)

    # Train ulang classifier
    model.named_steps['clf'].fit(X_res, y_res)

    # Evaluasi
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    return model, acc, X_test, y_test


# =========================================
# 4. LOAD
# =========================================
df = generate_data()
model, acc, X_test, y_test = train_pipeline(df)

# =========================================
# 5. UI
# =========================================
st.title("🏥 SPK Prioritas Pasien (Pipeline ML Ready)")
st.write("Model: Decision Tree + Pipeline + OneHotEncoder (Anti Error)")

st.metric("Akurasi Model", f"{acc:.2f}")

# ======================
# INPUT USER
# ======================
st.sidebar.header("Input Pasien")

umur = st.sidebar.slider("Umur", 1, 100, 50)
tekanan = st.sidebar.slider("Tekanan Darah", 80, 200, 120)
detak = st.sidebar.slider("Detak Jantung", 60, 150, 90)
darurat = st.sidebar.selectbox("Tingkat Darurat", [1,2,3])
riwayat = st.sidebar.selectbox("Riwayat Penyakit", [0,1])
kondisi = st.sidebar.selectbox("Kondisi Medis", ["ringan","sedang","berat"])

# ======================
# PREDIKSI
# ======================
if st.sidebar.button("Prediksi"):
    input_df = pd.DataFrame([{
        "umur": umur,
        "tekanan_darah": tekanan,
        "detak_jantung": detak,
        "tingkat_darurat": darurat,
        "riwayat_penyakit": riwayat,
        "kondisi_medis": kondisi
    }])

    pred = model.predict(input_df)[0]

    if pred == 1:
        st.error("🔴 Prioritas Tinggi")
    elif pred == 2:
        st.warning("🟡 Prioritas Sedang")
    else:
        st.success("🟢 Prioritas Rendah")

# ======================
# RANKING
# ======================
df_pred = df.copy()
df_pred["prediksi"] = model.predict(df.drop("prioritas", axis=1))

ranking = df_pred.sort_values(by="prediksi")

st.subheader("🏆 Ranking Pasien")
st.dataframe(ranking.head(10))

# ======================
# DATA
# ======================
st.subheader("📊 Sample Data")
st.dataframe(df.head(20))