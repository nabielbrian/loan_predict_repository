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
    st.write('Approved sebanyak 2.680 data (62,1%) dan Rejected sebanyak 1.635 data (37,9%). Visualisasi ini menunjukkan jumlah Approved lebih banyak dibanding Rejected.')

    st.write('## Perbedaan skor CIBIL antara tipe pinjaman')
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.boxplot(data=data, x='loan_status', y='cibil_score', palette='Blues', ax=ax)
    ax.set_title('Distribusi CIBIL Score Berdasarkan Loan Status')
    st.pyplot(fig)
    st.write("Grafik menunjukkan bahwa pemohon dengan status Approved memiliki median skor CIBIL sekitar 710 (IQR ±620–800) yang jauh lebih tinggi dibandingkan pemohon dengan status Rejected, yang memiliki median sekitar 430 (IQR ±370–490). Sebaran skor pada kelompok Approved juga berada di kisaran yang lebih tinggi, sedangkan kelompok Rejected umumnya di kisaran skor rendah. Perbedaan median yang besar ini menegaskan bahwa semakin tinggi skor CIBIL, semakin besar peluang persetujuan pinjaman")

    st.write('## Hubungan antar Loan Amount dengan Loan Status?')
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
    st.write('Grafik menunjukkan adanya korelasi positif antara pendapatan tahunan dan jumlah pinjaman, di mana pemohon dengan pendapatan >±8 juta dan pinjaman >±3 juta didominasi oleh status Approved. Sebaliknya, pada pendapatan <±5 juta dan pinjaman <±2,5 juta, lebih banyak ditemukan status Rejected. Meskipun jarang, terdapat kasus pemohon dengan pendapatan tinggi namun tetap Rejected, menandakan bahwa pendapatan bukan satu-satunya faktor penentu persetujuan.')

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
    st.write('Grafik menunjukkan bahwa tenor 2 dan 4 tahun memiliki jumlah persetujuan tertinggi dengan selisih besar dibandingkan penolakan. Untuk tenor di atas 6 tahun, selisih persetujuan dan penolakan semakin kecil, menandakan peluang persetujuan lebih rendah. Secara umum, tenor pendek cenderung lebih disukai dan memiliki tingkat persetujuan lebih tinggi, meskipun tenor panjang tetap memperoleh persetujuan dalam jumlah cukup besar namun dengan selisih lebih tipis.')


