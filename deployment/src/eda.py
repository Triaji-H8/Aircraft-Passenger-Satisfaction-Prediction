import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from PIL import Image

# Cache dataset agar tidak didownload ulang tiap reload
@st.cache_data
def load_data():
    url = 'https://huggingface.co/datasets/BesottenJenny/airline_dataset/resolve/main/dataset_airline.csv'
    df = pd.read_csv(url)
    return df

# Load dataset
df = load_data()

def run():
    # Judul dan subjudul
    st.title('Aircraft Passenger Satisfaction Prediction')
    st.subheader('This page lets you explore the dataset used in the model')

    # Menambahkan gambar
    gambar = Image.open('./src/airline.jpg')
    st.image(gambar, caption='Image Source: wallpaper.com')

    # Teks statis untuk memberi info loading
    st.markdown(
        '<p style="color:gray; font-size:14px; font-style:italic;">It may take a while to load the CSV dataset... please wait for 2 minutes and DO NOT refresh the page :)</p>',
        unsafe_allow_html=True
    )

    # Menampilkan dataframe
    st.dataframe(df)

    # Visualisasi distribusi target
    st.write('### Distribution Plot of Passenger Satisfaction') 
    fig, ax = plt.subplots(figsize=(15, 5))
    sns.countplot(x='satisfaction', data=df, palette='pastel', ax=ax)
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(
            f'{height:,} ({height/len(df)*100:.1f}%)',
            (p.get_x() + p.get_width()/2, height),
            ha='center', va='bottom',
            fontsize=10, fontweight='bold'
        )
    st.pyplot(fig)

    st.write("""
**Target Distribution (`satisfaction`)**:  
- The *neutral or dissatisfied* category dominates with 56.55% of passengers, while *satisfied* accounts for only 43.45%.  
- From a business perspective, this indicates that more than half of passengers have a low or neutral level of satisfaction, which could potentially affect the airline's reputation and customer loyalty.
""")

    # Heatmap korelasi fitur numerik
    st.write('### Correlation Heatmap for Numerical Features') 
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    corr_matrix = df[numeric_cols].corr()
    fig = px.imshow(
        corr_matrix,
        text_auto=".2f",
        color_continuous_scale='RdBu_r',
        title='Correlation Heatmap (Numerical Features)',
        width=1000,
        height=800
    )
    st.plotly_chart(fig, use_container_width=False)

    st.write("""
**Business Implication:**  
In-flight service features that directly interact with passengers (`boarding, entertainment, seat comfort, leg room, cabin crew service`, and `cleanliness`) have the **greatest influence** on satisfaction. Therefore, the airline should prioritize investing in improving the in-flight experience to enhance customer satisfaction.  
Meanwhile, factors such as `schedule delays`, although not as influential as service-related features, should still be addressed to prevent potential dissatisfaction.
""")

    # Distribusi kategori
    st.write('### Distribution Plot of Categorical Columns') 
    categorical_cols = ['Gender', 'Customer Type', 'Type of Travel', 'Class']

    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(10, 5))
        sns.countplot(
            x=col,
            data=df,
            hue='satisfaction',
            palette={'satisfied': '#4CAF50', 'neutral or dissatisfied': '#F44336'},
            ax=ax
        )
        ax.set_title(f'Distribution of {col} by Satisfaction', fontsize=14, fontweight='bold')
        ax.legend(title='Satisfaction', loc='upper right')
        st.pyplot(fig)

# Aktivasi script
if __name__ == '__main__':
    run()