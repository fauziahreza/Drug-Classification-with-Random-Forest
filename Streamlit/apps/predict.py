import streamlit as st
import pandas as pd
import numpy as np

import pickle

def app():
    st.title('Drug Prediction')
    upload_file = st.file_uploader("Unggah Dataset")
    if upload_file is None:
        st.info('File belum di Unggah')
    else:
        global df
        df = pd.read_csv(upload_file)
        main()
    
loaded_model = pickle.load(open(r'C:\Users\fauziah reza o\Documents\Learn Data Science\Tugas UAS\Tugas UAS\Deploy\trained_model.pkl', 'rb'))

def drug_predict(age, sex, bp, col, na_to_k):
    if sex == "Male":
        sex = 1
    if sex == "Female":
        sex = 0
    
    if bp == "Low":
        bp = 1
    elif bp == "Normal":
        bp = 2
    elif bp == "High":
        bp = 0

    if col == "Normal":
        col = 1
    elif col == "High":
        col = 0
    
    input_data_as_numpy_array = np.asarray([[age, sex, bp, col, na_to_k]])
    
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    Prediction = loaded_model.predict(input_data_reshaped)

    if Prediction == 0:
        pred = "Drugs Y"
    if Prediction == 1:
        pred = "Drugs A"
    if Prediction == 2:
        pred = "Drugs B"
    if Prediction == 3:
        pred = "Drugs C"
    if Prediction == 4:
        pred = "Drugs X"
    return pred

def main():
    age = st.number_input('Masukkan umur anda')
    sex = st.selectbox('Masukkan jenis kelamin anda', ('Male', 'Female'))
    bp = st.selectbox('Masukkan tekanan darah anda', ('Low','Normal','High'))
    col = st.selectbox('Masukkan colesterol anda',('Normal', 'High'))
    na_to_k = st.number_input('Masukkan Na_to_K anda')

    result =""
    if st.button("Predict"):
        result = drug_predict(age, sex, bp, col, na_to_k)
    st.success('Kamu cocok mengkonsumsi obat {}'.format(result))

if __name__ == '__main__':
    main()