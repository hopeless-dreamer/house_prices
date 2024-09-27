# base
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Важная настройка для корректной настройки pipeline!
import sklearn
sklearn.set_config(transform_output="pandas")

# Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.linear_model import LinearRegression

# Preprocessing
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler, MinMaxScaler, OrdinalEncoder, TargetEncoder
from sklearn.model_selection import GridSearchCV, StratifiedKFold

# for model learning
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score

#models
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from catboost import CatBoostRegressor
# Metrics
from sklearn.metrics import accuracy_score

from sklearn.inspection import permutation_importance
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import f_regression, chi2
from scipy.stats import ttest_ind
from sklearn.metrics import root_mean_squared_log_error

import streamlit as st
import joblib
from io import BytesIO


st.title(':orange[Моделирование цены на недвижимость]')
st.subheader('***:violet[Для обучения была использована модель Catboost]***')
file = st.file_uploader(':orange[Загрузите ваш файл csv]', type=['csv'])

if file is not None:
    file = pd.read_csv(file)
else:
    st.stop()
    
ml_pipeline = joblib.load('ml_pipeline.pkl')

result = ml_pipeline.predict(file)

submission = pd.DataFrame()
submission['Id']=file['Id']
submission['SalePrice']=result

buffer = BytesIO()
submission.to_csv(buffer, index=False)
buffer.seek(0) 

st.download_button("Скачайте файл", data=buffer, file_name="submission.csv", mime="text/csv")