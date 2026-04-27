# =========================================
# SPK PRIORITAS PASIEN - CLEAN PIPELINE
# =========================================

import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

import matplotlib.pyplot as plt
import seaborn as sns

# =========================================
# CONFIG
# =========================================
st.set_page_config(
    page_title="SPK Prioritas Pasien",
    layout="wide",
    page_icon="🏥"
)

# =========================================
# STYLE (MEDICAL UI)
# =========================================
st.markdown("""
<style>
body { background-color: #f6fbf9; }
.block-container { padding: 1.5rem 2rem; }

.header {
    background: linear-gradient(135deg, #2e7d32, #66bb6a);
    color: white;
    padding: 25px;
    border-radius: 16px;
}

.card {
    background: white;
    padding: 20px;
    border-radius: 16px;
    box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    margin-bottom: 20px;
}

.badge-high { background:#ffebee; color:#c62828; padding:15px; border-radius:10px; }
.badge-med { background:#fff8e1; color:#f9a825; padding:15px; border-radius:10px; }
.badge-low { background:#e8f5e9; color:#2e7d32; padding:15px; border-radius:10px; }

.stButton>button {
    background:#2e7d32;
    color:white;
    border-radius:10px;
    height:45px;
}
</style>
""", unsafe_allow_html=True)

# =========================================
# DATA GENERATION
# =========================================
@st.cache_data
def load_data():
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

    def label_rule(r):
        score = 0
        if r.umur > 60: score += 2
        if r.tekanan_darah > 140: score += 2
        if r.detak_jantung > 110: score += 2
        if r.tingkat_darurat == 3: score += 3
        if r.riwayat_penyakit == 1: score += 1
        if r.kondisi_medis == "berat": score += 3

        return 1 if score >= 7 else 2 if score >= 4 else 3

    df["prioritas"] = df.apply(label_rule, axis=1)
    return df

# =========================================
# MODEL PIPELINE
# =========================================
@st.cache_resource
def build_model(df):

    X = df.drop("prioritas", axis=1)
    y = df["prioritas"]

    num_cols = ["umur", "tekanan_darah", "detak_jantung"]
    cat_cols = ["tingkat_darurat", "riwayat_penyakit", "kondisi_medis"]

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.2, random_state=42
    )

    # transform
    X_train_tr = preprocessor.fit_transform(X_train)
    X_test_tr = preprocessor.transform(X_test)

    # imbalance handling
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X_train_tr, y_train)

    # model
    model = DecisionTreeClassifier(max_depth=5, random_state=42)
    model.fit(X_res, y_res)

    y_pred = model.predict(X_test_tr)
    acc = accuracy_score(y_test, y_pred)

    return {
        "preprocessor": preprocessor,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "accuracy": acc,
        "num_cols": num_cols,
        "cat_cols": cat_cols
    }

# =========================================
# LOAD SYSTEM
# =========================================
df = load_data()
artifacts = build_model(df)

# =========================================
# HEADER
# =========================================
st.markdown("""
<div class="header">
<h1>🏥 Sistem Pendukung Keputusan Prioritas Pasien</h1>
<p>Decision Support System berbasis Machine Learning</p>
</div>
""", unsafe_allow_html=True)

# =========================================
# METRICS
# =========================================
c1, c2, c3 = st.columns(3)
c1.metric("Akurasi", f"{artifacts['accuracy']:.2f}")
c2.metric("Data", len(df))
c3.metric("Fitur", df.shape[1]-1)

# =========================================
# INPUT VALIDATION FUNCTION
# =========================================
def validate_input(umur, tekanan, detak):
    if tekanan < 70 or tekanan > 220:
        return False, "Tekanan darah tidak valid"
    if detak < 40 or detak > 180:
        return False, "Detak jantung tidak valid"
    return True, ""

# =========================================
# INPUT FORM
# =========================================
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Input Data Pasien")

c1, c2, c3 = st.columns(3)

with c1:
    umur = st.slider("Umur", 1, 100, 50)

with c2:
    tekanan = st.slider("Tekanan Darah", 80, 200, 120)
    detak = st.slider("Detak Jantung", 60, 150, 90)

with c3:
    darurat = st.selectbox("Tingkat Darurat", [1,2,3])
    riwayat = st.selectbox("Riwayat Penyakit", [0,1])
    kondisi = st.selectbox("Kondisi Medis", ["ringan","sedang","berat"])

predict = st.button("Analisis")

st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# PREDICTION
# =========================================
if predict:

    valid, msg = validate_input(umur, tekanan, detak)

    if not valid:
        st.error(msg)
    else:
        input_df = pd.DataFrame([{
            "umur": umur,
            "tekanan_darah": tekanan,
            "detak_jantung": detak,
            "tingkat_darurat": darurat,
            "riwayat_penyakit": riwayat,
            "kondisi_medis": kondisi
        }])

        X_tr = artifacts["preprocessor"].transform(input_df)
        pred = artifacts["model"].predict(X_tr)[0]

        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Hasil Prioritas")

        if pred == 1:
            st.markdown('<div class="badge-high">PRIORITAS TINGGI - SEGERA</div>', unsafe_allow_html=True)
        elif pred == 2:
            st.markdown('<div class="badge-med">PRIORITAS SEDANG</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-low">PRIORITAS RENDAH</div>', unsafe_allow_html=True)

        st.markdown('</div>', unsafe_allow_html=True)

# =========================================
# TABS ANALISIS
# =========================================
tab1, tab2, tab3 = st.tabs(["Evaluasi", "Feature Importance", "Dataset"])

with tab1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(artifacts["y_test"], artifacts["y_pred"])
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", ax=ax)
    st.pyplot(fig)

    st.subheader("Classification Report")
    report = classification_report(
        artifacts["y_test"],
        artifacts["y_pred"],
        output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose())

with tab2:
    ohe = artifacts["preprocessor"].named_transformers_["cat"]
    encoded = ohe.get_feature_names_out(artifacts["cat_cols"])
    features = artifacts["num_cols"] + list(encoded)

    imp = artifacts["model"].feature_importances_

    df_imp = pd.DataFrame({
        "feature": features,
        "importance": imp
    }).sort_values(by="importance", ascending=False)

    st.dataframe(df_imp)

    fig2, ax2 = plt.subplots()
    ax2.barh(df_imp["feature"], df_imp["importance"])
    ax2.invert_yaxis()
    st.pyplot(fig2)

with tab3:
    st.dataframe(df.head(20))
