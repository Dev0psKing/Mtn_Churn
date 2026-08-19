import streamlit as st
import pandas as pd
import joblib
from textblob import TextBlob

# Load model and metadata (cached so it's not reloaded from disk on every click)
@st.cache_resource
def load_artifacts():
    model = joblib.load('mtn_churn_model.pkl')
    meta = joblib.load('model_metadata.pkl')
    return model, meta

model, meta = load_artifacts()

HIGH_RISK_THRESHOLD = meta['decision_threshold']
MEDIUM_RISK_THRESHOLD = meta['medium_risk_threshold']
HIGH_VALUE_REVENUE = meta['high_value_revenue_threshold']

st.title('MTN Nigeria Churn Predictor')
st.markdown('Identify at-risk subscribers and take action.')
st.info(
    "Every subscriber has a story. This tool turns usage, tenure, satisfaction, "
    "spending, and customer feedback into a churn-risk signal helping retention "
    "teams know who may need attention first."
)

# Input fields
with st.sidebar:
    st.header('Subscriber Details')
    age = st.slider('Age', 18, 90, 30)
    state = st.selectbox('State', meta['states'])
    device = st.selectbox('Device', meta['devices'])
    gender = st.selectbox('Gender', meta['genders'])
    plan = st.selectbox('Plan', meta['plans'])
    satisfaction = st.slider('Satisfaction (1-5)', 1, 5, 3)
    usage = st.number_input('Data Usage (GB)', 0.0, 500.0, 10.0)

    st.header('Account History')
    tenure_months = st.number_input('Customer Tenure (months)', 0, 200, 12)
    unit_price = st.number_input('Unit Price (NGN)', 0, 500000, 5000, step=500)
    num_purchases = st.number_input('Number of Times Purchased', 0, 200, 5)
    total_revenue = st.number_input('Total Revenue (NGN)', 0, 5000000, 25000, step=1000)

    st.header('Latest Review (optional)')
    review_text = st.text_area(
        'Customer review text',
        value='',
        help='Leave blank to treat sentiment as neutral.'
    )


def tenure_group(months):
    if months <= 12:
        return '0-1 Year'
    elif months <= 24:
        return '1-2 Years'
    elif months <= 48:
        return '2-4 Years'
    return '4+ Years'


def get_sentiment(text):
    if not text or not text.strip():
        return 0.0, 'Neutral'
    score = TextBlob(text).sentiment.polarity
    if score > 0.1:
        label = 'Positive'
    elif score < -0.1:
        label = 'Negative'
    else:
        label = 'Neutral'
    return score, label


# Predict
if st.button('Predict Churn Risk'):
    revenue_per_purchase = total_revenue / (num_purchases + 1)
    usage_intensity = usage / (tenure_months + 1)
    t_group = tenure_group(tenure_months)
    is_high_value = int(total_revenue >= HIGH_VALUE_REVENUE)
    sentiment_score, sentiment_class = get_sentiment(review_text)

    input_data = pd.DataFrame({
        'Age': [age], 'State': [state], 'MTN Device': [device], 'Gender': [gender],
        'Satisfaction Rate': [satisfaction], 'Customer Tenure in months': [tenure_months],
        'Subscription Plan': [plan], 'Unit Price': [unit_price],
        'Number of Times Purchased': [num_purchases],
        'Total Revenue': [total_revenue], 'Data Usage': [usage],
        'Revenue_per_Purchase': [revenue_per_purchase],
        'Usage_Intensity': [usage_intensity], 'Tenure_Group': [t_group],
        'Is_High_Value': [is_high_value], 'Review_Sentiment_Score': [sentiment_score],
        'Review_Sentiment_Class': [sentiment_class]
    })

    prob = model.predict_proba(input_data)[0][1]

    st.metric(
        'Churn Probability',
        f'{prob * 100:.1f}%'
    )

    if prob >= HIGH_RISK_THRESHOLD:
        st.error(
            '🔴 Priority Retention\n\n'
            'This subscriber shows elevated churn risk. '
            'Consider proactive retention outreach.'
        )
    elif prob >= MEDIUM_RISK_THRESHOLD:
        st.warning(
            '🟠 Watch List\n\n'
            'This subscriber shows moderate churn risk. '
            'Monitor engagement and consider proactive outreach.'
        )
    else:
        st.success(
            '🟢 Low Risk\n\n'
            'This subscriber currently shows relatively low churn risk. '
            'Maintain standard engagement.'
        )
    with st.expander('See computed features used by the model'):
        st.write(input_data.T.rename(columns={0: 'Value'}))
