import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def app():
    upload_file = st.file_uploader("Unggah Dataset")
    if upload_file is None:
        st.info('File belum di Unggah')
    else:
        df = pd.read_csv(upload_file)

        st.header("Data Understanding")
        # Inisialisasi nama kolom
        df.columns = ['Age',
                'Sex',
                'BP',
                'Cholesterol',
                'Na_to_K',
                'Drug']
        st.subheader("Dataset Awal")
        st.write(df)    

        st.header("Data Preparation")
        
        # Mengubah data categorical jadi numerik
        encoder = LabelEncoder()

        df['Sex'] = encoder.fit_transform(df['Sex'])
        df['BP'] = encoder.fit_transform(df['BP'])
        df['Cholesterol'] = encoder.fit_transform(df['Cholesterol'])
        df['Drug'] = encoder.fit_transform(df['Drug'])
        
        st.subheader("Dataset Tipe Numerik")
        st.write(df.head())

        # Hapus data yang ingin dijadikan label
        X = df.drop('Drug', axis = 1).copy()
        X.head()
        
        # Jadikan drug sebagai label
        Y = df['Drug'].copy()
        Y.head()        

        # Membagi data train dan test
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0)

        st.subheader("Split Data Train and Test")

        col1, col2 = st.columns([1,1])
        
        col1.caption("Data Training:")
        col1.write(X_train)
        col1.caption("Training Label Shape:")
        col1.write(Y_train.shape)

        col2.caption("Data Testing:")
        col2.write(X_test)
        col2.caption("Testing Label Shape:")
        col2.write(Y_test.shape)
