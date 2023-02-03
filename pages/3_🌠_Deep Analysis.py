import streamlit as st
import plotly.express as px
import pandas as pd
import requests
from plotly import graph_objects as go
from streamlit_lottie import st_lottie

st.set_page_config(page_title='Deep Analysis', 
    page_icon=':bow_and_arrow:', 
    layout='wide')

def lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

pre_lotti=lottie_url('https://assets8.lottiefiles.com/packages/lf20_AaXSI8h0rb.json')


st_lottie(pre_lotti, key = 'forget', height=100)
st.markdown('---')



fig = go.Figure()

fig.add_trace(go.Funnel(
    name = 'LinkedIn',
    y = ["Website visit", "Downloads", "Applicants", "Accepted", 'Finalized'],
    x = [120, 60, 30, 20, 15],
    textposition='inside',
    textinfo = "value+percent initial")
    )

fig.add_trace(go.Funnel(
    name = 'Twitter',
    orientation = "h",
    y = ["Website visit", "Downloads", "Applicants", "Accepted", 'Finalized'],
    x = [90, 70, 50, 10, 7],
    textposition = "inside",
    textinfo = "value+percent total"))

fig.add_trace(go.Funnel(
    name = 'Facebook',
    orientation = "h",
    y = ["Website visit", "Downloads", "Applicants", "Accepted", 'Finalized'],
    x = [100, 60, 40, 3,1],
    textposition = "outside",
    textinfo = "value+percent previous"))

fig.update_layout(title_text = 'Advertisements Effect', 
    title_pad_r = 4, title_pad_l = 6)


st.plotly_chart(fig, use_container_width=True)
st.markdown('---')

df = px.data.gapminder()
df['department'] = df['continent'].map({'Asia':"Computer Science", 
    'Europe':'Nanotechnology', 
    'Africa':"Robotics", 
    'Americas':"Engineering",
    'Oceania':'MBA'})


df = df[df['year']>1970]
df['pop'] = df['pop']/1000000


fig = px.bar(df, x="department", y="pop", color="department",
    animation_frame="year", animation_group="country",
    labels={"pop": "Thesis numbers"}, range_y=[0,4000])


fig.update_layout(
    title={
        'text': "Published Thesis Quantity by Department",
        'y':0.9,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})


st.plotly_chart(fig, use_container_width=True)
st.markdown('---')


import plotly.figure_factory as ff

df = [dict(Task="Advertise on different Social Media", Start='2022-10-01', Finish='2022-12-30', Resource='Complete'),
      dict(Task="Recieve Early Track applications", Start='2022-12-01', Finish='2022-12-31', Resource='Complete'),
      dict(Task="Advertise on different Social Media", Start='2023-02-01', Finish='2023-03-10', Resource='Incomplete'),
      dict(Task="Recieve Regular Track applications", Start='2023-02-15', Finish='2023-03-15', Resource='Incomplete'),
      dict(Task="Check Early Track applications", Start='2022-12-10', Finish='2023-01-31', Resource='Incomplete'),
      dict(Task="Send Early Track Results", Start='2023-02-01', Finish='2023-02-15', Resource='Incomplete'),
      dict(Task="Interview Early Track Results", Start='2023-02-20', Finish='2023-03-10', Resource='Not Started'),
      dict(Task="Confirm Early Track Applicants", Start='2023-03-15', Finish='2023-03-25', Resource='Not Started'),
      dict(Task="Check Regular Track applications", Start='2023-03-10', Finish='2023-03-31', Resource='Not Started'),
      dict(Task="Send Regular Track Results", Start='2023-04-01', Finish='2023-04-10', Resource='Not Started'),
      dict(Task="Interview Regular Track Applicants", Start='2023-04-15', Finish='2023-04-28', Resource='Not Started'),
      dict(Task="Confirm Regular Track Applicants", Start='2023-04-30', Finish='2023-05-10', Resource='Not Started'),
      dict(Task="Finalisation", Start='2023-05-20', Finish='2023-05-30', Resource='Not Started'),
      ]
      
colors = {'Not Started': 'rgb(220, 0, 0)',
          'Incomplete': (1, 0.9, 0.16),
          'Complete': 'rgb(0, 255, 100)'}

fig = ff.create_gantt(df, colors=colors, index_col='Resource', 
                        show_colorbar=True,
                        group_tasks=True, title='Duration')

fig.update_layout(margin = dict(t = 0,l = 0,r = 0,b = 0), 
    legend_xanchor = 'right', legend_yanchor = 'top',legend_y = 1.0, 
    legend_font_size = 17, 
    font_size = 20, title_x = 1)
fig.update_xaxes(showgrid = True)
fig.update_yaxes(showgrid = True)


st.plotly_chart(fig, use_container_width=True)
st.markdown('---')

stages = ["Website & Poster visit", "Contacting", "Potential applicants", "Application Done", "Get Accepted"]
df_mtl = pd.DataFrame(dict(number=[100, 50, 32, 11, 4], stage=stages))
df_mtl['Internship'] = 'International Intern'
df_toronto = pd.DataFrame(dict(number=[100, 70, 60, 42, 20], stage=stages))
df_toronto['Internship'] = 'Local Intern'
df = pd.concat([df_mtl, df_toronto], axis=0)
fig = px.funnel(df, 
    x='number', 
    y='stage', 
    color='Internship', 
    title = 'Local & International Internship')
fig.update_layout(title_x = 0.5)



st.plotly_chart(fig, use_container_width= True)



hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
