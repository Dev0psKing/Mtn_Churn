# MTN Nigeria Churn Predictor

A machine learning app that estimates the probability a MTN Nigeria subscriber will churn, built to help retention teams prioritize outreach.

**Live app:** https://collins-uwabor-mtn-churn-machine-learning-project.streamlit.app/

## Overview

Subscriber churn is a major cost driver in Nigeria's telecom market. This project builds an end-to-end pipeline, from raw subscriber data to a deployed prediction tool, that flags at-risk subscribers so retention teams can act before they leave.

Given a subscriber's demographics, plan, usage, satisfaction rating, and account history, the app returns a churn probability and a risk tier (Low / Medium / High).

## How it works

- **Model:** Random Forest classifier (400 trees), chosen over Logistic Regression, Decision Tree, and Gradient Boosting by cross-validated ROC-AUC rather than default-threshold accuracy or recall, since accuracy and 0.5-threshold recall depend heavily on an arbitrary cutoff.
- **Performance:** ROC-AUC of 0.69 on a held-out test set. This is moderate, not strong, and the app states this directly rather than implying more confidence than the model has.
- **Decision threshold:** tuned on the precision-recall curve to maximize F1 for the churn class (0.235, not the default 0.5), because in a retention context missing an actual churner is usually costlier than flagging a loyal customer for outreach.
- **Features:** demographics (age, state, gender, device), plan and pricing, usage and usage intensity, tenure, revenue history, satisfaction rating, and sentiment extracted from free-text customer reviews via TextBlob.
- **Top drivers** (from feature importance): usage intensity, data usage, and tenure rank ahead of satisfaction and sentiment as churn predictors in this dataset.

## Project structure

```
├── app.py                          # Streamlit app
├── mtn_churn_model.pkl             # Trained sklearn Pipeline (preprocessing + Random Forest)
├── model_metadata.pkl              # UI dropdown options, decision thresholds, high-value cutoff
├── requirements.txt                # Pinned dependencies (scikit-learn==1.6.1 required)
├── mtn_customer_churn.csv          # Training dataset (974 subscribers)
└── Final_Project_Notebook.ipynb    # Full pipeline: EDA, feature engineering, model comparison,
                                     # hyperparameter/threshold tuning, XAI, deployment code
```

## Running locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app loads `mtn_churn_model.pkl` and `model_metadata.pkl` at startup, so keep them in the same directory as `app.py`.

**Note on scikit-learn version:** the model was trained and pickled with scikit-learn 1.6.1. Installing a newer scikit-learn version can raise an `AttributeError` on load, since internal serialization details change between releases. `requirements.txt` pins this for you; don't loosen it without re-saving the model.

## Deployment

Deployed on [Streamlit Community Cloud](https://share.streamlit.io), connected to this GitHub repo. On redeploy, set the Python version explicitly to 3.11 or 3.12 in Advanced Settings, scikit-learn 1.6.1 doesn't have a prebuilt wheel for newer Python versions, which forces a slow source build that can stall indefinitely.

## Limitations

- Trained on 974 subscribers; a small dataset for this kind of task.
- ROC-AUC ~0.69 reflects a real but limited ceiling on what's learnable from the available features, this should support human judgment in retention decisions, not replace it.
- Correlational, not causal: the model identifies patterns associated with churn, not what causes it.
- Should be retrained periodically as subscriber behavior and market conditions shift.

## Author

Uwabor Collins