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
    total_revenue = unit_price * num_purchases
    st.metric('Total Revenue (NGN)', f'{total_revenue:,}')
    st.caption('Calculated automatically as Unit Price × Number of Times Purchased.')

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

    # ---- Determine risk tier, plain-language meaning, and recommended action ----
    if prob >= HIGH_RISK_THRESHOLD:
        tier = 'HIGH RISK'
        emoji = '🔴'
        headline = 'This subscriber is likely to leave soon.'
        meaning = (
            "Among past subscribers who looked similar to this one (same plan, usage, "
            "satisfaction, and history), a large share ended up cancelling their service. "
            "This is a strong warning sign, not a certainty, but it means this subscriber "
            "deserves attention now rather than later."
        )
        action = (
            "**Recommended action:** Reach out proactively within the next few days. "
            "Consider a loyalty discount, a plan upgrade offer, or a personal call to "
            "understand their concerns before they decide to leave."
        )
        box_fn = st.error
    elif prob >= MEDIUM_RISK_THRESHOLD:
        tier = 'MODERATE RISK'
        emoji = '🟠'
        headline = 'This subscriber shows some early warning signs.'
        meaning = (
            "This subscriber isn't in immediate danger of leaving, but their profile "
            "shares some traits with subscribers who eventually churned. Worth keeping "
            "an eye on rather than ignoring."
        )
        action = (
            "**Recommended action:** No urgent action needed, but add them to a watch "
            "list. A satisfaction check-in or a small engagement nudge (e.g. a data "
            "bonus) can help before risk increases."
        )
        box_fn = st.warning
    else:
        tier = 'LOW RISK'
        emoji = '🟢'
        headline = 'This subscriber is likely to stay.'
        meaning = (
            "This subscriber's profile closely matches past subscribers who remained "
            "loyal to MTN. There's no strong signal of dissatisfaction or disengagement "
            "right now."
        )
        action = (
            "**Recommended action:** No action needed. Continue standard engagement "
            "and normal service."
        )
        box_fn = st.success

    st.markdown('---')
    st.subheader(f'{emoji} {tier}')
    st.write(f"**{headline}**")

    box_fn(meaning)
    st.markdown(action)

    # ---- Explain WHY: compare each field to a typical loyal subscriber ----
    ref = meta.get('reference_profile')
    if ref:
        field_labels = {
            'Age': 'Age', 'State': 'State', 'MTN Device': 'Device', 'Gender': 'Gender',
            'Satisfaction Rate': 'Satisfaction rating', 'Customer Tenure in months': 'Tenure (months)',
            'Subscription Plan': 'Plan', 'Unit Price': 'Plan price', 'Number of Times Purchased': 'Purchase frequency',
            'Total Revenue': 'Total spend', 'Data Usage': 'Data usage', 'Revenue_per_Purchase': 'Spend per purchase',
            'Usage_Intensity': 'Usage intensity', 'Tenure_Group': 'Tenure group', 'Is_High_Value': 'High-value status',
            'Review_Sentiment_Score': 'Review sentiment', 'Review_Sentiment_Class': 'Review sentiment'
        }
        impacts = []
        for col in input_data.columns:
            if col not in ref:
                continue
            swapped = input_data.copy()
            swapped[col] = ref[col]
            swapped_prob = model.predict_proba(swapped)[0][1]
            delta = prob - swapped_prob  # positive = this field is pushing risk UP vs a typical loyal subscriber
            impacts.append((field_labels.get(col, col), delta))

        # Deduplicate by label (Revenue_per_Purchase/Usage_Intensity are derived, keep the clearer driver fields only)
        seen = set()
        deduped = []
        for label, delta in impacts:
            if label in seen:
                continue
            seen.add(label)
            deduped.append((label, delta))

        deduped.sort(key=lambda x: abs(x[1]), reverse=True)
        top = [d for d in deduped if abs(d[1]) >= 0.01][:4]

        if top:
            st.markdown('**Why this result:** compared to a typical subscriber who stayed with MTN, this subscriber differs mainly in:')
            for label, delta in top:
                if delta > 0:
                    st.write(f"- 🔺 **{label}** is increasing this subscriber's risk")
                else:
                    st.write(f"- 🔻 **{label}** is lowering this subscriber's risk")
        else:
            st.markdown('**Why this result:** this subscriber closely matches a typical MTN subscriber overall, no single factor stands out strongly.')

    with st.expander('See the underlying probability and risk scale'):
        st.metric('Churn Probability', f'{prob * 100:.1f}%')
        st.caption(
            f"Scale used: below {MEDIUM_RISK_THRESHOLD*100:.0f}% = Low Risk (bottom 25% of subscribers), "
            f"{MEDIUM_RISK_THRESHOLD*100:.0f}%-{HIGH_RISK_THRESHOLD*100:.0f}% = Moderate Risk (middle ~50%), "
            f"above {HIGH_RISK_THRESHOLD*100:.0f}% = High Risk (top 25% of subscribers). "
            "These cutoffs were set so risk tiers split real subscribers into meaningful groups, "
            "rather than flagging most people as one tier."
        )

    with st.expander('See computed features used by the model'):
        st.write(input_data.T.rename(columns={0: 'Value'}))