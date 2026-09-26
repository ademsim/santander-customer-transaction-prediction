from pathlib import Path

import numpy as np
import pandas as pd
import pickle
import streamlit as st

st.set_page_config(page_title="Santander Transaction Predictor")

BASE = Path(__file__).parent


@st.cache_resource
def load_model():
    return pickle.load(open(BASE / "santander_model.pkl", "rb"))


@st.cache_resource
def load_meta():
    return pickle.load(open(BASE / "santander_meta.pkl", "rb"))


model = load_model()
meta = load_meta()
features = meta["features"]
mean = pd.Series(meta["mean"])[features]
std = pd.Series(meta["std"])[features]
base_rate = meta["base_rate"]

st.title("Santander Transaction Predictor")
st.write(
    "A gradient boosting model (trained on the Kaggle Santander Customer Transaction dataset) estimates the "
    "probability that a customer will make a specific transaction. The 200 features are anonymized "
    f"(var_0 to var_199), so you can either explore a random customer or upload your own CSV. "
    f"On average, about {base_rate:.0%} of customers in the training data made the transaction."
)

tab_random, tab_csv = st.tabs(["Random customer", "Upload CSV"])

with tab_random:
    if "z" not in st.session_state or st.button("Generate a new random customer"):
        st.session_state["z"] = np.random.default_rng().normal(size=len(features))
    z = pd.Series(st.session_state["z"], index=features)

    chosen = st.multiselect(
        "Adjust some features (optional)",
        features,
        help="Values are shown as standard deviations from the training average (0 = typical customer).",
    )
    for name in chosen:
        z[name] = st.slider(f"{name} (standard deviations from average)", -4.0, 4.0, float(np.clip(z[name], -4, 4)), 0.1)

    row = pd.DataFrame([mean + std * z], columns=features)
    proba = float(model.predict_proba(row)[0, 1])
    st.metric("Probability of the transaction", f"{proba:.1%}", f"{proba - base_rate:+.1%} vs. average customer")
    st.progress(min(1.0, proba))

with tab_csv:
    st.write(f"Upload a CSV with the {len(features)} feature columns (var_0 ... var_199). An `ID_code` column is optional.")
    file = st.file_uploader("CSV file", type=["csv"])
    if file is not None:
        data = pd.read_csv(file)
        missing = [c for c in features if c not in data.columns]
        if missing:
            st.error(f"Missing {len(missing)} feature columns, for example: {missing[:3]}")
        else:
            out = pd.DataFrame({"probability": model.predict_proba(data[features])[:, 1]})
            if "ID_code" in data.columns:
                out.insert(0, "ID_code", data["ID_code"])
            st.dataframe(out.head(100), width="stretch")
            st.download_button("Download predictions", out.to_csv(index=False), "predictions.csv", "text/csv")

st.caption(
    "Model: HistGradientBoosting (validation AUC ≈ 0.88 on the Kaggle data). The random customer is generated from "
    "the average and spread of each feature in the training data, so it illustrates how the model reacts, "
    "not a real person."
)
