
import streamlit as st
import numpy as np
import pickle
import plotly.express as px

# Page Config
st.set_page_config(
    page_title="Diabetes Risk – Easy Input Form",
    layout="centered"
)

# Load model (cached for efficiency)
@st.cache_data
def load_model():
    return pickle.load(open('D:\\projects\\sid\\diabetes_prediction_rfc.pkl', 'rb'))

model = load_model()

# Header
st.title("Diabetes Risk Predictor")
st.write("Fill the form below and click **Submit** to get your result.")

# Use a form to group inputs and require submission
with st.form("predict_form", clear_on_submit=False, enter_to_submit=True):
    st.markdown("### Enter Your Health Metrics")
    
    # Group key inputs with helpful labels and placeholders
    pregnancies = st.number_input("Pregnancies (times)", min_value=0, step=1)
    glucose = st.number_input("Glucose (mg/dL)", min_value=0, help="Fasting blood sugar level")
    bp = st.number_input("Blood Pressure (mm Hg)", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0, format="%.1f")
    dpf = st.number_input("DPF", min_value=0.0, format="%.3f", help="Diabetes Pedigree Function")
    age = st.number_input("Age (years)", min_value=0, step=1)
    
    submit_button = st.form_submit_button("Submit & Predict")

if submit_button:
    # Validate that mandatory fields are filled
    missing = []
    for name, val in zip(
        ["Glucose", "BMI", "Age"],
        [glucose, bmi, age]
    ):
        if val == 0 or val == 0.0:
            missing.append(name)
    if missing:
        st.error(f"Please provide realistic non-zero values for: {', '.join(missing)}")
    else:
        x = np.array([[pregnancies, glucose, bp, 0, 0, bmi, dpf, age]])
        pred = model.predict(x)[0]
        proba = model.predict_proba(x)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("Result:")
        msg = "🛑 High Risk" if pred else "✅ Low Risk"
        st.metric("Diabetes Risk", msg, delta=f"{proba:.1%}" if proba is not None else "N/A")

        df_vis = {
            "Feature": ["Glucose", "BMI", "Age"],
            "Value": [glucose, bmi, age]
        }
        fig = px.bar(df_vis, x="Feature", y="Value", title="Key Input Overview", color="Feature")
        st.plotly_chart(fig, use_container_width=True)

        with st.expander("Tips for Input Accuracy"):
            st.write("""
            - Glucose should not be zero—make sure you measured it correctly.
            - BMI is important—provide your actual height & weight.
            - Double-check your inputs before submitting.
            """)
