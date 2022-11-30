import streamlit as st
import joblib as job

from catboost import CatBoostRegressor
from xgboost import XGBRegressor

@st.experimental_memo(show_spinner=False)
def load_model():
    with st.spinner(" ⏳ Downloading model... this may take awhile! \n Don't stop it!"):
        model_cat = CatBoostRegressor()
        model_cat.load_model('./model/model_cat.cbm')

        model_xgb = XGBRegressor()
        model_xgb.load_model('./model/model_xgb.json')

        model_rf = job.load("./model/model_rf.joblib")
        model_lr = job.load("./model/model_lr.joblib")
        model_knn = job.load("./model/model_knn.joblib")

    return model_cat,model_xgb,model_rf,model_lr,model_knn