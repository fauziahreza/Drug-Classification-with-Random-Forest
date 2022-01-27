import streamlit as st

import pickle

class MultiApp:
    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        app = st.sidebar.selectbox(
            'Navigation',
            self.apps,
            format_func=lambda app: app['title'])

        app['function']()

    def upload(self):
        upload_file = st.file_uploader("Unggah Dataset")
        if upload_file is None:
            st.info('File belum di Unggah')