import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots



st.set_page_config(page_title='Visual Insights', 
    page_icon=':banana:', 
    layout='wide')

# importing module

from streamlit_extras.dataframe_explorer import dataframe_explorer
# importing required module


# Introduction
st.markdown("<h2 style = 'text-align:center'> Visual Insights of the Students & Departments</h2>",unsafe_allow_html= True)

st.markdown('---')

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
#Explore Data
filtered_df = dataframe_explorer(dataset)
st.dataframe(filtered_df, use_container_width= True)


# Universal selection
st.sidebar.header('Universal Filtering  :game_die:')
st.sidebar.markdown('Your selection here applies to all graphs')
st.sidebar.markdown('If you want to apply selection to a certain graph itself, you can use selection box on top of each graphs')


select_experience = st.sidebar.multiselect('Students with Workexperience', 
 options=dataset['work_exp'].unique(), 
 default=dataset['work_exp'].unique())
st.sidebar.markdown("---")
select_department = st.sidebar.multiselect(label= 'Select the Department of the student',
 options=dataset['department'].unique(), default=dataset['department'].unique()) 
st.sidebar.markdown("---")
select_gender = st.sidebar.multiselect(label = 'Select Students Gender',
 options=dataset['gender'].unique(),
 default=dataset['gender'].unique())
st.sidebar.markdown('---')



df_1 = dataset[(dataset['gender'].isin(select_gender))&(dataset['department'].isin(select_department))&(dataset['work_exp'].isin(select_experience))]

### Let's Go to Body
st.markdown("---")
st.markdown("---")
st.markdown("<h2 style = 'text-align:center'>Students Grade Percentage</h2>", unsafe_allow_html=True)
left, right= st.columns([2,3])
gender_for_grade = left.multiselect(label= "Select the Gender", options = df_1['gender'].unique(),
 default=df_1['gender'].unique())
spec_for_grade = right.multiselect(label="Select the Department",
 options= df_1['department'].unique(), default = df_1['department'].unique())

# Query
df_2 = df_1[(df_1['department'].isin(spec_for_grade))&(df_1['gender'].isin(gender_for_grade))]

# Distribution of scores
title_pos = 0.5
#fig1
#fig2
fig2 = px.histogram(data_frame = df_2, x = '12_grade', 
    title="Students marks at High School")
fig2.update_layout(margin_l = 0, margin_r = 2, 
    yaxis_title = "",
    title_x = title_pos,
    xaxis_title = "Grade Percent")
fig2.update_traces(marker=dict(color="#e377c2"),
    marker_line_width=1,
    marker_line_color="black")

fig1 = px.histogram(data_frame = df_2, x = 'degree_percent', 
    title="Students marks at Bachelors", 
    color_discrete_sequence=px.colors.qualitative.G10)
fig1.update_layout(margin_l = 0, margin_r = 2,
    xaxis_title = "Grade Percent",
    title_x = title_pos, yaxis_title = 'Frequency')
fig1.update_traces(marker=dict(color="aqua"),
    marker_line_width=1,
    marker_line_color="black")

#fig_3
fig3 = px.histogram(data_frame = df_2, x = 'post_grad_percent', 
    title="Students marks at Master")
fig3.update_layout(margin_l = 0,margin_r = 2,yaxis_title = "",
 title_x = title_pos,xaxis_title = "Grade Percent")
fig3.update_traces(marker=dict(color="antiquewhite"),
    marker_line_width=1,marker_line_color="black")


left, middle, right = st.columns(3)
left.plotly_chart(fig2, use_container_width = True)
middle.plotly_chart(fig1, use_container_width = True)
right.plotly_chart(fig3, use_container_width = True)

# Pie chart
st.markdown('---')
st.markdown('---')

st.markdown("<h3 style = 'text-align:center'>Students and their Employed Rate by Department</h3>",
    unsafe_allow_html=True)

grouped_df_3 = df_1.groupby('department').count()[['gender']]
grouped_df_3 = grouped_df_3.reset_index().rename({'gender':'counting'}, axis =1)
grouped_df_3 = grouped_df_3.sort_values(by = 'counting', ascending = False)

def get_pull(alfa):
    pull_list = []
    puller = 0.1
    for i,value in enumerate(alfa):
        pull_list.append(puller - i/100) 
    return pull_list

pull_values = get_pull(grouped_df_3['counting'])
colors = ["red", "green", "blue", "goldenrod", "magenta"]

group_salary = pd.pivot_table(data = df_1, index = 'department',
    columns = 'status', values = 'salary', aggfunc=({'salary':'count'}))
group_salary['total'] = group_salary['Not Placed']+group_salary['Placed']
group_salary['not_placed_perc'] = group_salary['Not Placed']/group_salary['total']*100
group_salary['placed_perc'] = group_salary['Placed']/group_salary['total']*100
group_salary = group_salary.sort_values(by = 'placed_perc')
group_salary.reset_index(inplace=True)
group_salary['not_placed_perc'] = group_salary['not_placed_perc'].round()
group_salary['placed_perc'] = group_salary['placed_perc'].round()

#fig = go.Figure(data=[go.Pie(labels=grouped_df_3['department'], values=grouped_df_3['counting'],
#    pull=pull_values)])
#fig.update_layout(
#        title={
#            'text' : "Department",
#            'x':0.5,
#            'xanchor': 'auto'},
#    margin = dict(t = 60,l = 0,r = 0,b = 0), legend_x = 0,
#    legend_y = 0.1, legend_font_size = 17, font_size = 20)
#fig.update_traces(hoverinfo='label+percent', textinfo='label+value', textfont_size=20, textposition = 'outside',
#                marker=dict(line=dict(color='#000000', width=2)))

fig_l = make_subplots(
    rows=1, cols=2,
    specs=[[{"type": "domain"},{"type": "xy"}]])

fig_l.add_trace(go.Bar(
    x=group_salary['department'],
    y=group_salary['placed_perc'],
    name='Employed',
    text = group_salary['placed_perc']),
    row=1, col=2)
fig_l.add_trace(go.Bar(
    x=group_salary['department'],
    y=group_salary['not_placed_perc'],
    name="Unemployed",
    text=group_salary['not_placed_perc']
), row = 1, col = 2)

fig_l.add_trace(go.Pie(labels=grouped_df_3['department'], values=grouped_df_3['counting'],
                pull=pull_values),
            row=1, col=1)

fig_l.update_traces(hoverinfo='label+value', textinfo='label+percent', textposition = 'auto',showlegend = False,
                marker=dict(line=dict(color='#000000', width=2)), selector=dict(type='pie'))

fig_l.update_traces(showlegend = True, textposition = 'outside', 
                marker=dict(line=dict(color='#000000', width=2)), selector=dict(type='bar'))


fig_l.update_layout(
    margin_t = 120,
    xaxis_tickangle = -45, 
    xaxis_title = 'Department',
    height = 500,
    yaxis_title = 'Percentage',
    legend_font_size = 18)

st.plotly_chart(fig_l, use_container_width=True)

#fig1 = go.Figure()
#fig1.add_trace(go.Bar(
#    x=group_salary['department'],
#    y=group_salary['placed_perc'],
#    name='Have a Job',
#    
#    text = group_salary['placed_perc']
#))
#
#fig1.add_trace(go.Bar(
#    x=group_salary['department'],
#    y=group_salary['not_placed_perc'],
#    name="Not have a Job",
#    text=group_salary['not_placed_perc']
#))
#fig1.update_layout(title_text='Job Placement of Students', 
#    title_font_size = 22, 
#    title_x = 0.3,
#    margin_t = 120,
#    xaxis_tickangle = -45, 
#    xaxis_title = 'Department',
#    height = 500,
#    yaxis_title = 'Percentage',
#     legend_font_size = 18)
#
#fig1.update_traces(texttemplate='%{text:.2s}', textposition='inside', opacity = 0.9, )
#
#
#
#
#
#fig1 = go.Figure()
#fig1.add_trace(go.Bar(
#    x=group_salary['department'],
#    y=group_salary['placed_perc'],
#    name='Have a Job',
#    
#    text = group_salary['placed_perc']
#))
#
#fig1.add_trace(go.Bar(
#    x=group_salary['department'],
#    y=group_salary['not_placed_perc'],
#    name="Not have a Job",
#    text=group_salary['not_placed_perc']
#))
#fig1.update_layout(title_text='Job Placement of Students', 
#    title_font_size = 22, 
#    title_x = 0.3,
#    margin_t = 120,
#    xaxis_tickangle = -45, 
#    xaxis_title = 'Department',
#    height = 500,
#    yaxis_title = 'Percentage',
#     legend_font_size = 18)
#
#fig1.update_traces(texttemplate='%{text:.2s}', textposition='inside', opacity = 0.9, )

st.markdown('---')
st.markdown('---')

# Sunburst
fig_n = px.sunburst(df_1, path=['status','gender', 'department'], color = df_1.index,  
    color_continuous_scale='RdBu', width=1000, height=1000)
fig_n.update_coloraxes(showscale=False)
fig_n.update_layout(height = 700, title_text = 'How likely the Student get job by Social Factors?', 
    title_font_size = 20, title_x = 0.6)
fig_n.update_traces(textinfo="label+percent entry")
st.plotly_chart(fig_n, use_container_width = True,height = 700)






hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
