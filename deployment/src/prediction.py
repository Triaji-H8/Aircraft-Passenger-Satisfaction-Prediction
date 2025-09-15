# Import library
import streamlit as st
import numpy as np
import pandas as pd
import cloudpickle
import gzip
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay

# Load pipeline
with gzip.open('./src/rf_pipeline_model.pkl.gz', 'rb') as f:
    pipeline = cloudpickle.load(f)

def run():
    # Membuat title
    st.title('Aircraft Passenger Satisfaction Prediction')
    
    # Membuat subheader
    st.subheader('This page lets you predict customer satisfaction towards their flight experience with the airline')

    # Form input
    with st.form(key='airline_satisfaction_form'):
        st.write('---')
        st.subheader("Passenger Information")
        st.write('*These two features have no correlation to satisfaction. You are allowed to skip it as it is.*')
        unnamed_0 = st.number_input("Unnamed: 0 (Row Index)", min_value=0, step=1, value=279)
        id_col = st.text_input("ID", value="279279")
        st.write('---')
        gender = st.selectbox("Gender", ["Male", "Female"])
        customer_type = st.selectbox("Customer Type", ["Loyal Customer", "disloyal Customer"], help="If they frequently use the airline, select `Loyal Customer`")
        age = st.number_input("Age", min_value=0, max_value=100, value=30, step=1)
        type_of_travel = st.selectbox("Type of Travel", ["Business travel", "Personal Travel"])
        class_type = st.selectbox("Class", ["Business", "Eco", "Eco Plus"])
        flight_distance = st.number_input("Flight Distance", min_value=0, max_value=10000, value=500, step=50, help='Flight distance estimation in miles')
        st.write('---')
        st.subheader("Service Ratings (1-5)")
        st.write('1 = very bad, 5 = very good')
        inflight_wifi_service = st.slider("Inflight wifi service", min_value=1, max_value=5, step=1, value=3)
        departure_arrival_time_convenient = st.slider("Departure/Arrival time convenient", min_value=1, max_value=5, step=1, value=3)
        ease_of_online_booking = st.slider("Ease of Online booking", min_value=1, max_value=5, step=1, value=3)
        gate_location = st.slider("Gate location", min_value=1, max_value=5, step=1, value=3)
        food_and_drink = st.slider("Food and drink", min_value=1, max_value=5, step=1, value=3)
        online_boarding = st.slider("Online boarding", min_value=1, max_value=5, step=1, value=3)
        seat_comfort = st.slider("Seat comfort", min_value=1, max_value=5, step=1, value=3)
        inflight_entertainment = st.slider("Inflight entertainment", min_value=1, max_value=5, step=1, value=3)
        on_board_service = st.slider("On-board service", min_value=1, max_value=5, step=1, value=3)
        leg_room_service = st.slider("Leg room service", min_value=1, max_value=5, step=1, value=3)
        baggage_handling = st.slider("Baggage handling", min_value=1, max_value=5, step=1, value=3)
        checkin_service = st.slider("Checkin service", min_value=1, max_value=5, step=1, value=3)
        inflight_service = st.slider("Inflight service", min_value=1, max_value=5, step=1, value=3)
        cleanliness = st.slider("Cleanliness", min_value=1, max_value=5, step=1, value=3)
        st.write('---')
        st.subheader("Flight Timings")
        departure_delay = st.number_input("Departure Delay in Minutes", min_value=0, max_value=1440, value=0, step=15, help='Departure delay in minutes')
        arrival_delay = st.number_input("Arrival Delay in Minutes", min_value=0, max_value=1440, value=0, step=15, help='Arrival delay in minutes')
        st.write('---')
        submitted = st.form_submit_button("Predict Satisfaction")
        st.write('---')

    # Buat DataFrame sesuai urutan kolom asli
    data_inf = pd.DataFrame([{
        "Unnamed: 0": unnamed_0,
        "id": id_col,
        "Gender": gender,
        "Customer Type": customer_type,
        "Age": age,
        "Type of Travel": type_of_travel,
        "Class": class_type,
        "Flight Distance": flight_distance,
        "Inflight wifi service": inflight_wifi_service,
        "Departure/Arrival time convenient": departure_arrival_time_convenient,
        "Ease of Online booking": ease_of_online_booking,
        "Gate location": gate_location,
        "Food and drink": food_and_drink,
        "Online boarding": online_boarding,
        "Seat comfort": seat_comfort,
        "Inflight entertainment": inflight_entertainment,
        "On-board service": on_board_service,
        "Leg room service": leg_room_service,
        "Baggage handling": baggage_handling,
        "Checkin service": checkin_service,
        "Inflight service": inflight_service,
        "Cleanliness": cleanliness,
        "Departure Delay in Minutes": departure_delay,
        "Arrival Delay in Minutes": arrival_delay
    }])

    st.write("### Input Data Preview")
    st.dataframe(data_inf)

    if submitted:
        # Prediksi langsung menggunakan pipeline
        prediction = pipeline.predict(data_inf)[0]
        prediction_proba = pipeline.predict_proba(data_inf)[0]

        st.write("### Prediction Result")
        if prediction == "satisfied":
            st.success(f"Passenger is predicted to be **Satisfied** (Probability: {prediction_proba.max():.2%})")
            st.info("""
        **Business Recommendation:**
        - Maintain high-quality in-flight service, especially on **boarding, entertainment, seat comfort, leg room, cabin crew service, and cleanliness**.
        - Offer loyalty programs or special offers to retain satisfied customers.
        - Regularly monitor service quality to prevent any decline that could impact satisfaction.
        """)
        else:
            st.error(f"Passenger is predicted to be **Neutral or Dissatisfied** (Probability: {prediction_proba.max():.2%})")
            st.warning("""
        **Business Recommendation:**
        - Focus on improving **boarding process efficiency, seat comfort, leg room space, entertainment quality, cabin crew service, and cleanliness**.
        - Analyze and reduce potential delays to minimize dissatisfaction.
        - Implement targeted surveys to identify pain points from the passenger’s perspective.
        """)

# Aktivasi
if __name__ == '__main__':
    run()