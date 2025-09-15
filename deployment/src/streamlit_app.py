import streamlit as st
import eda
import prediction

st.set_page_config(
    page_title='Aircraft Passenger Satisfaction Prediction',
    layout='wide',
    initial_sidebar_state='expanded'
)

page = st.sidebar.selectbox('Select Page: ', ('Exploratory Data Analysis (EDA)', 'Prediction'))


if page == 'Exploratory Data Analysis (EDA)':
    eda.run()
else:
    prediction.run()

st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 12px; margin-top: 50px;">
        © 2025 Zhaky Baridwan Triaji. All rights reserved. <br>
        References: Dataset from <a href="https://www.kaggle.com" target="_blank" style="color: gray;">Kaggle</a>
    </div>
    """,
    unsafe_allow_html=True
)