# --- LIBRARY ---

import streamlit as st
import pandas as pd
import numpy as np
import os
from PIL import Image
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import mean_squared_error

from src.model import load_model
from src.utils import Scale
from src.data import load_data

####--------------------MAIN---------------------####

## Set-up default web
st.set_page_config(layout="wide", page_icon="👨‍🎓", page_title="Thesis-Cao-Long")

### INPUT
st.sidebar.header('Input Features')

#Load Data
if os.path.isdir('./data') and os.path.isdir('./model') and os.path.isdir('./test') and os.path.isdir('./train'):
       pass
else:
       load_data()

# Load csv test
df_cat = pd.read_csv('./test/test_cat.csv')
df_knn = pd.read_csv('./test/test_knn.csv')
df_lr = pd.read_csv('./test/test_lr.csv')
df_rf = pd.read_csv('./test/test_rf.csv')
df_xgb = pd.read_csv('./test/test_xgb.csv')
df_train = pd.read_csv('./train/train.csv')
y_train = pd.read_csv('./train/target_train.csv')
df_val = pd.read_csv('./train/val.csv')
y_val = pd.read_csv('./train/target_val.csv')

### OUTPUT
image_cover = Image.open('/home/dhsang/streamlit/streamlit-thesis/images/ML.jpeg')
st.image(image_cover,use_column_width= True)

st.write("""
# Predict Total Sale Next Month From Machine Learning
This is the daily sales data of one of the largest Russian software companies - 1C Company
""")

# Define input each model
shop_id = st.sidebar.selectbox("shop_ID",df_cat["shop_id"].unique())
subset = df_cat[df_cat['shop_id']==shop_id]
item_id = st.sidebar.selectbox("item_ID",subset["item_id"].unique())

var_cat = subset[subset["item_id"]==item_id]

cat_features =  ['shop_id', 'item_id', 'item_price', 'mean_item_price', 'item_cnt',
       'mean_item_cnt', 'transactions', 'year', 'month', 'item_price_unit',
       'hist_min_item_price', 'hist_max_item_price', 'price_increase',
       'price_decrease', 'item_cnt_min', 'item_cnt_max', 'item_cnt_mean',
       'item_cnt_std', 'item_cnt_shifted1', 'item_cnt_shifted2',
       'item_cnt_shifted3', 'item_trend', 'shop_mean', 'item_mean',
       'shop_item_mean', 'year_mean', 'month_mean']

xgb_features = ['item_cnt','item_cnt_mean', 'item_cnt_std', 'item_cnt_shifted1', 
                'item_cnt_shifted2', 'item_cnt_shifted3', 'shop_mean', 
                'shop_item_mean', 'item_trend', 'mean_item_cnt']

rf_features = ['shop_id', 'item_id', 'item_cnt', 'transactions', 'year',
               'item_cnt_mean', 'item_cnt_std', 'item_cnt_shifted1', 
               'shop_mean', 'item_mean', 'item_trend', 'mean_item_cnt']

lr_features = ['item_cnt', 'item_cnt_shifted1', 'item_trend', 'mean_item_cnt', 'shop_mean']

knn_features = ['item_cnt', 'item_cnt_mean', 'item_cnt_std', 'item_cnt_shifted1',
                'item_cnt_shifted2', 'shop_mean', 'shop_item_mean', 
                'item_trend', 'mean_item_cnt']

#Load dataloader
var_xgb = var_cat[xgb_features]
var_rf = var_cat[rf_features]
var_lr = var_cat[lr_features]
var_knn = var_cat[knn_features]

cat_train = df_train[cat_features]
xgb_train = df_train[xgb_features]
rf_train = df_train[rf_features]
lr_train = df_train[lr_features]
knn_train = df_train[knn_features]

cat_val = df_val[cat_features]
xgb_val = df_val[xgb_features]
rf_val = df_val[rf_features]
lr_val = df_val[lr_features]
knn_val = df_val[knn_features]

#Load model
model_cat,model_xgb,model_rf,model_lr,model_knn = load_model()
lr_scaler,knn_scaler = Scale(lr_train,knn_train)


with st.spinner(" ⏳ Processing training... this may take awhile! \n Don't stop it!"):
       #CatBoost Predict
       catboost_train_pred = model_cat.predict(cat_train)
       catboost_val_pred = model_cat.predict(cat_val)

       st.write("#### CATBOOST")
       st.write('Train rmse:', np.sqrt(mean_squared_error(y_train, catboost_train_pred)))
       st.write('Validation rmse:', np.sqrt(mean_squared_error(y_val, catboost_val_pred)))
       mae = mean_absolute_error(y_val, catboost_val_pred)
       st.write("MAE:",mae)
       mape = mean_absolute_percentage_error(catboost_val_pred,y_val)
       st.write("MAPE:",mape)

       #XGB Predict
       xgb_train_pred = model_cat.predict(xgb_train)
       xgb_val_pred = model_cat.predict(xgb_val)

       st.write("#### XGB")
       st.write('Train rmse:', np.sqrt(mean_squared_error(y_train, xgb_train_pred)))
       st.write('Validation rmse:', np.sqrt(mean_squared_error(y_val, xgb_val_pred)))
       mae = mean_absolute_error(y_val, xgb_val_pred)
       st.write("MAE:",mae)
       mape = mean_absolute_percentage_error(xgb_val_pred,y_val)
       st.write("MAPE:",mape)

       #random forest Predict
       rf_train_pred = model_cat.predict(rf_train)
       rf_val_pred = model_cat.predict(rf_val)

       st.write("#### Random Forest")
       st.write('Train rmse:', np.sqrt(mean_squared_error(y_train, rf_train_pred)))
       st.write('Validation rmse:', np.sqrt(mean_squared_error(y_val, rf_val_pred)))
       mae = mean_absolute_error(y_val, rf_val_pred)
       st.write("MAE:",mae)
       mape = mean_absolute_percentage_error(rf_val_pred,y_val)
       st.write("MAPE:",mape)

       #Lenear Regression
       lr_train_pred = model_cat.predict(lr_train)
       lr_val_pred = model_cat.predict(lr_val)

       st.write("#### Linear Regression")
       st.write('Train rmse:', np.sqrt(mean_squared_error(y_train, lr_train_pred)))
       st.write('Validation rmse:', np.sqrt(mean_squared_error(y_val, lr_val_pred)))
       mae = mean_absolute_error(y_val, lr_val_pred)
       st.write("MAE:",mae)
       mape = mean_absolute_percentage_error(lr_val_pred,y_val)
       st.write("MAPE:",mape)

       #KNN
       knn_train_pred = model_cat.predict(knn_train)
       knn_val_pred = model_cat.predict(knn_val)

       st.write("#### Linear Regression")
       st.write('Train rmse:', np.sqrt(mean_squared_error(y_train, knn_train_pred)))
       st.write('Validation rmse:', np.sqrt(mean_squared_error(y_val, knn_val_pred)))
       mae = mean_absolute_error(y_val, knn_val_pred)
       st.write("MAE:",mae)
       mape = mean_absolute_percentage_error(knn_val_pred,y_val)
       st.write("MAPE:",mape)

model_list = ["CatBoost", "XGBoost", "RandomForest", "Linear Regression", "KNN Regression"]

if st.button('Show result'):
       with st.spinner(" ⏳ Waiting ... this may take awhile! \n Don't stop it!"):
              rs_cat = model_cat.predict(var_cat)
              rs_xgb = model_xgb.predict(var_xgb)
              rs_rf = model_rf.predict(var_rf)
              rs_knn = model_knn.predict(var_knn)
              var_lr = lr_scaler.transform(var_lr)
              rs_lr = model_lr.predict(var_lr)
              var_knn = knn_scaler.transform(var_knn)
              rs_knn =model_knn.predict(var_knn)

              result_list = [rs_cat,rs_xgb,rs_rf,rs_lr,rs_knn]

              df_rs = pd.DataFrame(result_list)
              df_rs = df_rs.transpose()
              df_rs.columns = model_list
       st.sucess("Done")
       st.dataframe(df_rs)
       


       





















