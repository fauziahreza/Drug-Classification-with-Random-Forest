from apps import evaluasi
import streamlit as st
from multiapp import MultiApp
from apps import preprocessing, evaluasi, home, predict

import pickle

app = MultiApp()

st.title("Prediction With Random Forest")

# Tambah app
app.add_app("Business Description", home.app)
app.add_app("Data Preprocessing", preprocessing.app)
app.add_app("Evaluation Model", evaluasi.app)
app.add_app("Prediction", predict.app)

# app.add_app("Evaluasi Model", evaluasi.app)
# The main app
app.run()