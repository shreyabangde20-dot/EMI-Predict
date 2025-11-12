# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib

st.set_page_config(page_title="EMIPredict AI", layout="wide")
st.title("EMIPredict AI — EMI Eligibility & Max EMI Prediction")

# File names (must be in same folder)
CSV_NAMES = ["emi_encoded.csv", "encoded_emi_data.csv", "EMI Predict.csv"]
MODEL_NAME = "best_model.pkl"  # optional

# --- Load dataset if available ---
DATA_PATH = None
for name in CSV_NAMES:
    if os.path.exists(name):
        DATA_PATH = name
        break

if DATA_PATH:
    df = pd.read_csv(DATA_PATH)
    st.success(f"Loaded dataset: {DATA_PATH} (shape: {df.shape})")
else:
    df = pd.DataFrame()  # empty schema
    st.warning("No emi_encoded.csv found in folder. Upload or place the CSV file in app folder.")

# --- Try load model ---
model = None
if os.path.exists(MODEL_NAME):
    try:
        model = joblib.load(MODEL_NAME)
        st.info(f"Loaded model from {MODEL_NAME}")
    except Exception as e:
        st.error(f"Failed to load {MODEL_NAME}: {e}")
        model = None
else:
    st.info("No saved model (best_model.pkl) found — app will use a simple fallback rule for demo.")

# Fallback simple rule (only if model None)
def simple_rule_predict(single_df):
    # This is a very basic heuristic: if credit_score>700 and monthly_salary sufficient -> eligible
    # Adjust thresholds based on your data
    result = {}
    try:
        cs = float(single_df.get("credit_score", single_df.get("Credit_Score", np.nan)))
        sal = float(single_df.get("monthly_salary", single_df.get("person_income", np.nan)))
        if np.isnan(cs) or np.isnan(sal):
            result["eligibility"] = None
        else:
            result["eligibility"] = 1 if (cs >= 700 and sal >= 25000) else 0
            result["eligibility_prob"] = 0.85 if result["eligibility"]==1 else 0.15
            result["pred_max_monthly_emi"] = sal * 0.25  # simple affordability assumption
    except Exception:
        result["eligibility"] = None
    return result

# --- Helper to align features with training dummies ---
@st.cache_data
def get_train_dummies(df):
    feat_cols = [c for c in df.columns if c not in ('emi_eligibility','max_monthly_emi')]
    return pd.get_dummies(df[feat_cols])

train_dummies = None
if not df.empty:
    train_dummies = get_train_dummies(df)

def prepare_input_for_model(input_df):
    # input_df: DataFrame of raw input(s)
    if train_dummies is None:
        # no baseline; just try to numeric-convert and return
        return input_df.select_dtypes(include=[np.number])
    combined = pd.concat([train_dummies.head(0), input_df], ignore_index=True, sort=False)
    combined = pd.get_dummies(combined)
    for c in train_dummies.columns:
        if c not in combined.columns:
            combined[c] = 0
    combined = combined[train_dummies.columns]
    return combined

# ---------- UI: Sidebar navigation ----------
page = st.sidebar.selectbox("Page", ["Home", "Single Predict", "Batch Predict", "Data Explorer"])

# ---------- Home ----------
if page == "Home":
    st.header("Welcome")
    st.markdown("- Use **Single Predict** to input one customer and get eligibility + max EMI prediction.")
    st.markdown("- Use **Batch Predict** to upload CSV and get bulk predictions.")
    st.markdown("- Place `emi_encoded.csv` and optional `best_model.pkl` in this folder for best results.")
    if not df.empty:
        st.subheader("Sample data")
        st.dataframe(df.head(8))

# ---------- Single Predict ----------
elif page == "Single Predict":
    st.header("Single customer prediction")
    st.write("Fill the fields. If your dataset has different column names, use matching names in the uploaded CSV.")
    # Build simple input form using most common EMI columns
    cols = {
        "age": ("Age", 30),
        "monthly_salary": ("Monthly Salary", 30000),
        "credit_score": ("Credit Score", 650),
        "existing_loans": ("Existing Loans (count)", 0),
        "current_emi_amount": ("Current EMI total (monthly)", 0),
        "requested_amount": ("Requested Loan Amount", 200000),
        "requested_tenure": ("Requested Tenure (months)", 24)
    }
    input_data = {}
    with st.form("single_form"):
        for k,(label,default) in cols.items():
            val = st.number_input(label, value=float(default))
            input_data[k] = val
        submitted = st.form_submit_button("Predict")
    if submitted:
        input_df = pd.DataFrame([input_data])
        X_input = prepare_input_for_model(input_df)
        if model is not None:
            try:
                pred_clf = None; prob = None; pred_reg = None
                if hasattr(model, "predict_proba"):
                    pred_clf = model.predict(X_input)[0]
                    prob = model.predict_proba(X_input)[:,1][0]
                else:
                    pred_clf = model.predict(X_input)[0]
                # If model is single pipeline for clf+reg, adapt accordingly. Here assume classifier-only or regressor-only saved.
                # We attempt both: classifier predict, regressor predict
                try:
                    pred_reg = model.predict(X_input)[0]
                except Exception:
                    pred_reg = None
                st.write({"eligibility": int(pred_clf) if pred_clf is not None else None, "prob": float(prob) if prob is not None else None, "pred_max_monthly_emi": float(pred_reg) if pred_reg is not None else None})
                if pred_clf == 1:
                    st.success("✅ Eligible")
                else:
                    st.error("❌ Not Eligible")
                if pred_reg is not None:
                    st.info(f"Predicted max monthly EMI: {pred_reg:.2f}")
            except Exception as e:
                st.error(f"Model prediction failed: {e}")
        else:
            res = simple_rule_predict(input_data)
            if res.get("eligibility") is None:
                st.error("Not enough numeric inputs for fallback rule.")
            else:
                if res["eligibility"]==1:
                    st.success("✅ Eligible (fallback)")
                else:
                    st.error("❌ Not Eligible (fallback)")
                st.info(f"Predicted max monthly EMI (fallback): {res.get('pred_max_monthly_emi'):.2f}")

# ---------- Batch Predict ----------
elif page == "Batch Predict":
    st.header("Batch prediction (upload CSV)")
    uploaded = st.file_uploader("Upload CSV with features (no targets required)", type=["csv"])
    if uploaded is not None:
        batch = pd.read_csv(uploaded)
        st.write("Uploaded shape:", batch.shape)
        X_batch = prepare_input_for_model(batch)
        preds_clf = None
        preds_reg = None
        if model is not None:
            try:
                # If model supports predict_proba (classifier)
                if hasattr(model, "predict_proba"):
                    preds_clf = model.predict(X_batch)
                    probs = model.predict_proba(X_batch)[:,1]
                    batch["pred_eligibility"] = preds_clf
                    batch["pred_eligibility_prob"] = probs
                else:
                    # model might be regressor
                    preds_reg = model.predict(X_batch)
                    batch["pred_max_monthly_emi"] = preds_reg
            except Exception as e:
                st.error("Prediction failed: " + str(e))
        else:
            # fallback
            out = []
            for _, row in batch.iterrows():
                out.append(simple_rule_predict(row.to_dict()))
            out_df = pd.DataFrame(out)
            batch = pd.concat([batch, out_df], axis=1)
        st.dataframe(batch.head(50))
        csv = batch.to_csv(index=False).encode('utf-8')
        st.download_button("Download predictions CSV", csv, file_name="predictions.csv", mime="text/csv")

# ---------- Data Explorer ----------
elif page == "Data Explorer":
    st.header("Data Explorer")
    if df.empty:
        st.info("No CSV loaded into app folder.")
    else:
        st.write(df.describe(include='all').T)
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            col = st.selectbox("Column to visualize", numeric_cols)
            st.write(df[col].describe())
            st.bar_chart(df[col].dropna().sample(min(1000, len(df))).values)

st.markdown("---")
st.caption("EMIPredict AI • Simple Streamlit demo. For production: use saved preprocessing pipeline and saved models.")
