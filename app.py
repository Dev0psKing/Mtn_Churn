
import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load('mtn_churn_model.pkl')
meta = joblib.load('model_metadata.pkl')

st.title('⌛ MTN Nigeria Churn Predictor')
st.markdown('Identify at-risk subscribers and take action.')

# Input fields
with st.sidebar:
    st.header('Subscriber Details')
    age = st.slider('Age', 18, 90, 30)
    state = st.selectbox('State', meta['states'])
    device = st.selectbox('Device', meta['devices'])
    plan = st.selectbox('Plan', meta['plans'])
    satisfaction = st.slider('Satisfaction (1-5)', 1, 5, 3)
    usage = st.number_input('Data Usage (GB)', 0.0, 500.0, 10.0)

# Predict
if st.button('Predict Churn Risk'):
    # Pre-process minimal input
    # Note: For simplicity, we use placeholder values for complex engineered features
    input_data = pd.DataFrame({
        'Age': [age], 'State': [state], 'MTN Device': [device], 'Gender': ['Male'],
        'Satisfaction Rate': [satisfaction], 'Customer Tenure in months': [12],
        'Subscription Plan': [plan], 'Unit Price': [5000], 'Number of Times Purchased': [5],
        'Total Revenue': [25000], 'Data Usage': [usage], 'Revenue_per_Purchase': [5000],
        'Usage_Intensity': [usage/13], 'Tenure_Group': ['1-2 Years'], 
        'Is_High_Value': [0], 'Review_Sentiment_Score': [0.5], 'Review_Sentiment_Class': ['Positive']
    })
    
    prob = model.predict_proba(input_data)[0][1]
    st.metric('Churn Probability', f'{prob*100:.2f}%')
    
    if prob > 0.6: st.error('High Risk: Immediate Retention Offer Recommended!')
    else: st.success('Low Risk: Maintain Standard Engagement.')
