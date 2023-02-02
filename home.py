import streamlit as st



st.set_page_config(page_title='Multipage App', page_icon=":anchor:", layout='wide')
st.markdown('<h3 style =  "text-align:center">Main Page</h3>', unsafe_allow_html=True)

st.markdown('Hello, this web application is for authoritiy of Universities, such as KAIST. It will help them easily access & controll the information of their University.')
st.success('Please choose one of the menu on the sidebar')



if "my_input" not in st.session_state:
    st.session_state['my_input']= "default"
my_input = st.text_input('Input favourite word here', key = ['my_input'])
submit = st.button('Submit')
if submit:
    st.session_state['my_input'] = my_input
    st.write('Amazing word: ', my_input)


import requests

from streamlit_lottie import st_lottie
import json

def load_lottifile(filepath: str):
    with open(filepath, 'r') as f:
        return json.load(f)

def lottie_url(url: str):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()


lottie_hello = load_lottifile("alfa.json")
lottie_middle = lottie_url('https://assets4.lottiefiles.com/packages/lf20_gjiu6QkgBx.json')
lottie_bye = lottie_url('https://assets4.lottiefiles.com/packages/lf20_mKMcjgVTY6.json')
left, mid, right = st.columns([1,1,1])
with left:
    use_container_width = True
    st_lottie(lottie_hello)
with mid:
    use_container_width = True
    st_lottie(lottie_middle, key = 'midd')
with right:
    use_container_width = True
    st_lottie(lottie_bye, key = 'bye',quality='high', reverse=True)
hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)


