import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt 
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn import metrics 
from sklearn.metrics import classification_report
from sklearn.metrics import plot_confusion_matrix
from sklearn.ensemble import RandomForestClassifier

def app():
    params = dict()
    max_depth = st.sidebar.slider('max_depth', 2, 15)
    params['max_depth'] = max_depth
    n_estimators = st.sidebar.slider('n_estimators', 1, 100)
    params['n_estimators'] = n_estimators
    upload_file = st.file_uploader("Unggah Dataset")
    if upload_file is None:
        st.info('File belum di Unggah')
    else:
        df = pd.read_csv(upload_file)

        st.header("Data Exploration")
        # Inisialisasi nama kolom
        df.columns = ['Age',
                'Sex',
                'BP',
                'Cholesterol',
                'Na_to_K',
                'Drug']
        st.subheader("Dataset Awal")
        st.write(df)    

        # Hapus data yang ingin dijadikan label
        X = df.drop('Drug', axis = 1).copy()
        X.head()
        
        # Jadikan drug sebagai label
        Y = df['Drug'].copy()
        Y.head()
        
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

        # st.subheader("Dataset Label Tipe Numerik")
        # st.write(Y)

        # st.header("Split Train and Test Data")
        # Membagi data train dan test dengan 80% 20%
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0) # 80% training 20 % Testing

        clf_dt = RandomForestClassifier(n_estimators=params['n_estimators'],criterion='entropy',max_depth=params['max_depth'], random_state=0)
        clf_dt = clf_dt.fit(X_train, Y_train)
        clf_dt = clf_dt.estimators_

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

        # Jumlah
        # st.subheader("Features and Labels Shape")
        # st.write('Training Features Shape:', X_train.shape)
        # st.write('Training Labels Shape:', Y_train.shape)
        # st.write('Testing Features Shape:', X_test.shape)
        # st.write('Testing Labels Shape:', Y_test.shape)


        # Visualiasi Decision Tree dari klasifikasi metode random forest
        st.header("Prediction With Random Forest")

        # Create Decision Tree classifer object
        clf = RandomForestClassifier()

        # Train Decision Tree Classifer
        clf = clf.fit(X_train,Y_train)

        #Predict the response for test dataset
        Y_pred = clf.predict(X_test)

        # Menentukan Akurasi
        st.write(classification_report(Y_test, Y_pred))
        st.write("Accuracy: %0.2f" % (100* metrics.accuracy_score(Y_test, Y_pred)),'%')

        # Buat checkbox untuk menampilkan correlation heatmap
        if st.checkbox("Tampilkan Plot Decision Tree"):
            st.header("Random Forest Tree Visualization")
            plt.figure(figsize=(30,10))
            tree.plot_tree(clf_dt[0],filled= True, feature_names = X.columns,class_names=["DrugX","drugA","drugB","drugC","drugY" ])
            st.pyplot(plt)  

        if st.checkbox("Tampilkan Plot Confusion Matrix"):
            st.header("Confussion Matrix")
            plt.figure(figsize = (10, 6))
            plot_confusion_matrix(clf, X_test, Y_test, display_labels=["DrugY", "drugC", "drugX", "drugA", "drugB"])
            st.pyplot(plt)

        if st.checkbox("Tampilkan Correlation Heatmap"):
            st.subheader("Correlation Heatmap")

            plt.figure(figsize=(16,9))
            ax = sns.heatmap(X.corr(),annot = True,cmap = 'viridis')
            st.pyplot(plt)