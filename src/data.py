import streamlit as st
import urllib
import shutil

@st.experimental_memo(show_spinner=False)
def load_data():
    with st.spinner(" ⏳ Downloading data... this may take awhile! \n Don't stop it!"):
        url_data = 'https://github.com/dinhsang1999/streamlit-thesis/releases/download/data/data.zip'
        url_train = 'https://github.com/dinhsang1999/streamlit-thesis/releases/download/data/train.zip'
        url_test = 'https://github.com/dinhsang1999/streamlit-thesis/releases/download/data/test.zip'
        url_model = 'https://github.com/dinhsang1999/streamlit-thesis/releases/download/data/model.zip'
        urllib.request.urlretrieve(url_data,"data.zip")
        urllib.request.urlretrieve(url_train,"train.zip")
        urllib.request.urlretrieve(url_test,"test.zip")
        urllib.request.urlretrieve(url_model,"model.zip")

        shutil.unpack_archive('data.zip')
        shutil.unpack_archive('train.zip')
        shutil.unpack_archive('test.zip')
        shutil.unpack_archive('model.zip')


