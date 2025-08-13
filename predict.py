import streamlit as st
import pandas as pd
import pickle

with open('rf_pipeline_after_tuning.pkl', 'rb') as file_5:
  model = pickle.load(file_5)

def run():
    st.write('# Predict Loan Approval')

    # form inference
    with st.form("my_form"):
        no_of_dependents = st.number_input("Jumlah Tanggungan", min_value=0, step=1)
        income_annum = st.number_input("Pendapatan Tahunan", min_value=0)
        loan_amount = st.number_input("Jumlah Pinjaman", min_value=0)
        loan_term = st.number_input("Jangka Waktu Pinjaman (tahun)", min_value=0, max_value=20)
        cibil_score = st.number_input("CIBIL Score", min_value=0, max_value=900)

        residential_assets_value = st.number_input("Nilai Aset Tempat Tinggal", min_value=0)
        commercial_assets_value = st.number_input("Nilai Aset Komersial", min_value=0)
        luxury_assets_value = st.number_input("Nilai Aset Mewah", min_value=0)
        bank_asset_value = st.number_input("Nilai Aset Bank", min_value=0)

        education = st.selectbox("Pendidikan", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Wiraswasta", ["Yes", "No"])

        submit = st.form_submit_button("Predict")

    # Inference dataset
    data_inf = {
        "no_of_dependents": no_of_dependents,
        "income_annum": income_annum,
        "loan_amount": loan_amount,
        "loan_term": loan_term,
        "cibil_score": cibil_score,
        "residential_assets_value": residential_assets_value,
        "commercial_assets_value": commercial_assets_value,
        "luxury_assets_value": luxury_assets_value,
        "bank_asset_value": bank_asset_value,
        "education": education,
        "self_employed": self_employed
    }

    data_inf = pd.DataFrame([data_inf])
    st.dataframe(data_inf)

    # predict
    # predict
    if submit:
        result = model.predict(data_inf)
        status = "✅ Disetujui" if result[0] == 1 else "❌ Ditolak"
        color = "#16a34a" if result[0] == 1 else "#dc2626"  # hijau / merah

        st.markdown(
            f"<h2 style='color:{color};'>Prediksi Loan Status: {status}</h2>",
            unsafe_allow_html=True
        )


