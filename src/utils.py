import streamlit as st
from sklearn.preprocessing import StandardScaler, MinMaxScaler

@st.experimental_memo(show_spinner=False)
def Scale(a,b):
    lr_scaler = MinMaxScaler()
    knn_scaler = MinMaxScaler()

    lr_scaler.fit(a)
    knn_scaler.fit(b)

    return lr_scaler,knn_scaler