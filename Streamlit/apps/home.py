import streamlit as st

def app():
    st.title('Business Description')

    st.subheader('Objective') 
    st.write('Prediksi obat menggunakan metode klasifikasi Random Forest')

    st.subheader('Description') 
    st.write('Pengobatan sendiri (self medication) merupakan upaya yang paling banyak dilakukan masyarakat untuk mengatasi keluhan atau gejala penyakit sebelum mereka memutuskan mencari pertolongan ke pusat pelayanan kesehatan/petugas kesehatan. Tapi ada keuntungan dan kekurangan dalam proses self medication, maka dari itu dalam membuat sebuah machine learning diperlukan data yang valid dan proses yang benar. Kami menghadirkan sebuah machine learning berbasis python yang akan menunjang self medication dalam memberikan pertolongan pertama khususnya dalam pemilihan obat berdasarkan karakteristik karakteristik yang dimiliki oleh pasien')