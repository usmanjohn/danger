import streamlit as st



st.set_page_config(page_title='Multipage App', page_icon=":anchor:", layout='wide')
st.markdown('<h3 style =  "text-align:center">Main Page</h3>', unsafe_allow_html=True)

st.markdown('Hello, this web application is for authoritiy of Universities, such as KAIST. It will help them easily access & controll the information of their University.')
st.success('Please choose one of the menu on the sidebar')



if "my_input" not in st.session_state:
    st.session_state['my_input']= "default"
my_input = st.text_input('Input a text here', key = ['my_input'])
submit = st.button('Submit')
if submit:
    st.session_state['my_input'] = my_input
    st.write('You have entered: ', my_input)



hide_st_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        </style>
        """
st.markdown(hide_st_style, unsafe_allow_html=True)


