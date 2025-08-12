import streamlit as st
from PIL import Image
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

def run():
    st.write('# Loan Approval Analysis')
    gambar = Image.open('happy.jpg')
    st.image(gambar)

    #Latar Belakang
    st.write("# Latar Belakang")
    st.write('''Persetujuan pinjaman merupakan layanan penting di bank atau lembaga keuangan untuk menentukan apakah pengajuan nasabah disetujui atau ditolak. Proses manual sering memakan waktu dan berisiko perbedaan penilaian antar petugas.
            \n Dengan machine learning, keputusan dapat dibuat otomatis berdasarkan data historis seperti pendapatan, riwayat kredit, dan status pekerjaan. Model prediksi ini membantu mempercepat proses, meningkatkan akurasi, mengurangi risiko, 
            dan memberikan keputusan yang konsisten sehingga kepuasan nasabah meningkat.''')
    st.write("# Target Audience")
    st.write('''Model ini ditujukan untuk bank dan lembaga keuangan sebagai alat bantu dalam menilai kelayakan pengajuan pinjaman, sehingga proses persetujuan lebih cepat, objektif, dan efisien.''')

    #dataset
    st.write("# Dataset")
    data = pd.read_csv('loaned_data_clean.csv')
    #tampilakn dataframe
    st.dataframe(data)  

    #Visualisasi
    st.write("# Exploratory Data Analysis")

    st.write('## Distribusi status persetujuan pinjaman')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(data=data, x='loan_status', palette='Blues', ax=ax)
    ax.set_title('Distribusi Loan Status')
    st.pyplot(fig)

    st.write('## Perbedaan skor CIBIL antara tipe pinjaman')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=data, x='loan_status', y='cibil_score', palette='Blues', ax=ax)
    ax.set_title('Distribusi CIBIL Score Berdasarkan Loan Status')
    st.pyplot(fig)

    st.write('## Korelasi antar variabel numerik')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.scatterplot(
        data=data,
        x='income_annum',
        y='loan_amount',
        hue='loan_status',
        palette='Blues',
        ax=ax
    )
    ax.set_title('Loan Amount vs Income Berdasarkan Loan Status')
    st.pyplot(fig)

    st.write('## Distribusi jangka waktu pinjaman per kategori persetujuan')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.countplot(
        data=data,
        x='loan_term',
        hue='loan_status',
        palette='Blues',
        ax=ax
    )
    ax.set_title('Distribusi Loan Term berdasarkan Loan Status')
    st.pyplot(fig)