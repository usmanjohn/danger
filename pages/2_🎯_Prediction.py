import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pickle
import requests
from streamlit_lottie import st_lottie
import json
    
st.set_page_config(page_title='Predictions', 
    page_icon=':game_die:', 
    layout='wide')

# Introduction
def lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

pre_lottie=lottie_url('https://assets4.lottiefiles.com/packages/lf20_34qRI0i4ti.json')


st_lottie(pre_lottie, key = 'target', height=100)
st.markdown('---')



#Modeling
st.markdown("<h3 style = 'text-align:center'>Predict whether a student will get a job!</h3>",
    unsafe_allow_html=True)
st.markdown('---')
st.markdown('Choose from followings and predict possibility of getting a job.')



# Reading the dataset
@st.cache
def load_data():
    data = pd.read_csv('students.csv')
    return data
df = load_data()

np.random.seed(1)
major_spec = ['TMBA', 'SEMBA', 'ME', 'IMMBA', 'PMBA','FMBA','MFE','IMMS','GBP',"EMBA",'DMFBA']
major_generated = np.random.choice(major_spec, 
    p = [0.08,0.09,0.1,0.07,0.06,0.085,0.115,0.05,0.15,0.095,0.105],
    size = len(df))
# Preprocess
@st.cache
def next_data(df):
    global major_generated
    df = df.drop(['ssc_b', 'hsc_b'], axis = 1)

    column_names=['gender', '10_grade', '12_grade', 'spec_higher_edu',
    'degree_percent','undergrad_major', 'work_exp' ,'employ_test', 
    'post_grad_spec', 'post_grad_percent','status', 'salary' ]
    df.columns = column_names

    df['salary'] = df['salary'].round()
    df['10_grade'] = df['10_grade'].round()
    df['12_grade'] = df['12_grade'].round()
    df['post_grad_percent'] = df['post_grad_percent'].round()
    df['degree_percent'] = df['degree_percent'].round()
    df.drop('post_grad_spec', axis = 1, inplace = True)
    df.loc[df['status']=='Not Placed', 'salary']= 0
    df['department'] = major_generated
    df['gender']=df['gender'].map({'M':'Male', 'F':'Female'})
    return df
dataset = next_data(df=df)



from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import Pipeline

encoder = OneHotEncoder()
scaler = StandardScaler()
model = RandomForestClassifier()
or_encode = OrdinalEncoder()


#################

cols_to_drop = ['spec_higher_edu', '10_grade', '12_grade', 'employ_test', 'post_grad_percent', 'salary']
train = dataset.drop(cols_to_drop, axis = 1)
X = train.drop('status', axis = 1)

def user_input():
    left, right = st.columns(2)
    pre_gender = left.radio('CHoose a gender', options=X['gender'].unique())
    grade_percent = right.slider('Students Mark', min_value = 50, max_value=100, value=75)
    left, middle, right= st.columns(3)
    undergrad_major = left.radio('Undergraduate major', options=X['undergrad_major'].unique())
    work_experience = middle.radio('Does Student have previous work experience?', options=X['work_exp'].unique())
    departmet_of = right.selectbox("Choose a student's department", options=X['department'].unique())
    features = pd.DataFrame({'gender':pre_gender, 'degree_percent': grade_percent, 'undergrad_major':undergrad_major,
                            'work_exp':work_experience, 'department':departmet_of}, index = [0])
    
    return features
input_df = user_input()
class_predict = pd.concat([input_df, X], axis=0)
class_dump = pd.get_dummies(class_predict)
class_scaled = scaler.fit_transform(class_dump)
st.write(class_predict)
class_predict_1 = class_scaled[:1]


load_pickle = pickle.load(open('classifier_model.pkl','rb'))
prediction = load_pickle.predict(class_predict_1)
prediction_prob = load_pickle.predict_proba(class_predict_1)
st.subheader('Will the Student get a job?')
job_find = np.array(['No', 'Yes'] )
st.write(job_find[prediction])

st.write("With the probability of "+ str((np.round(100*np.max(prediction_prob),1)))+' %')

st.markdown('---')

st.markdown("<h3 style = 'text-align:center'>Predict the possible Salary!</h3>",
    unsafe_allow_html=True)
st.markdown('---')


#Salary Prediction
scaler_new = StandardScaler()
regress = dataset[dataset['status']=='Placed']
X_1 = regress.drop(['spec_higher_edu', '10_grade', '12_grade', 'employ_test', 'salary', 'status'], axis =1)
y_new = regress['salary']

def user_input_new():
    left, middle, right = st.columns(3)
    pre_gender_1 = left.radio('CHoose a gende here', options=X_1['gender'].unique())
    postgrad_major = middle.radio('Choose a department', options=X_1['department'].unique())
    under_major = right.radio('Choose the undergraduate major', options=X_1['undergrad_major'].unique())
    left, middle, right = st.columns(3)
    under_g_per = left.slider('Undergraduate grade percent', min_value=50, max_value=100, value=80)
    graduate_grade = middle.slider('Graduate grade percent', min_value=50, max_value=100, value=70)
    work_exper = right.radio('Have previous work experience?', options=X_1['work_exp'].unique())

    features_1 = pd.DataFrame({'gender':pre_gender_1, 'degree_percent': under_g_per, 'undergrad_major':under_major,
                           'work_exp':work_exper, 'post_grad_percent':graduate_grade,'department':postgrad_major}, index = [0])
    
    return features_1
input_df_new = user_input_new()
X_reg_pre = pd.concat([input_df_new, X_1], axis = 0)
X_new_dump = pd.get_dummies(X_reg_pre)
X_new_scaled = scaler_new.fit_transform(X_new_dump)
X_regress_pre = X_new_scaled[:1]


load_pickle_1 = pickle.load(open('regressor_model.pkl','rb'))
prediction_1 = load_pickle_1.predict(X_regress_pre)
st.subheader('How much is the predicted salary?')
fin_val = prediction_1/100
st.write(str(np.float64(np.round(fin_val, 2)))+' USD')


hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
