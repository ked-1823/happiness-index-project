import streamlit as st
import numpy as np
import pandas as pd
import joblib

@st.cache_resource
def load_models():
    model = joblib.load('model.joblib')
    le_gender = joblib.load('gender_encoder.joblib')
    le_platform = joblib.load('platform_encoder.joblib')
    return model, le_gender, le_platform

model, le, le_p = load_models()

st.title("😊 Happiness Index Predictor")

age = st.number_input('Enter your age:', min_value=0.0, step=1.0)
gender = st.selectbox('Select your gender:', ['Male', 'Female'])
daily_screen = st.number_input('Enter daily screen time (hours):', min_value=0.0)
sleep = st.slider('Your approx sleep quality (1–10):', 1, 10, 5)
stress = st.slider('Approx stress level (1–10):', 1, 10, 7)
day_without = st.number_input('Enter number of days without social media:', min_value=0, max_value=30, step=1)
Exercise_Frequency = st.slider('How many days per week do you exercise?', 0, 7, 2)

Social_Media_Platform = st.text_input(
    'Which social media app do you use the most?',
    placeholder='e.g. X, Twitter, Instagram...'
)

if st.button('Predict Happiness Index'):
    if Social_Media_Platform.strip() == '':
        st.warning('⚠️ Please enter a social media platform before predicting.')
    else:
        Social_Media_Platform = Social_Media_Platform.strip().title()

        if Social_Media_Platform in ['X', 'Twitter', 'X (Twitter)', 'Twitter (X)']:
            Social_Media_Platform = 'X (Twitter)'

        if Social_Media_Platform not in le_p.classes_:
            st.info(f"ℹ️ Unrecognized platform '{Social_Media_Platform}', using default: X (Twitter)")
            Social_Media_Platform = 'X (Twitter)'

        gender_le = le.transform([gender])[0]
        social_le = le_p.transform([Social_Media_Platform])[0]

        x = pd.DataFrame([[age, gender_le, daily_screen, sleep, stress,
                           day_without, Exercise_Frequency, social_le]],
                         columns=['Age','Gender','Daily_Screen_Time(hrs)',
                                  'Sleep_Quality(1-10)','Stress_Level(1-10)',
                                  'Days_Without_Social_Media','Exercise_Frequency(week)',
                                  'Social_Media_Platform'])

        pred = model.predict(x)
        st.success(f'Predicted Happiness (1–10): {pred[0]:.2f}')
