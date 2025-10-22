import streamlit as st
import pandas as pd
import joblib

# Load trained salary prediction model
model = joblib.load('salary_rf_model.pkl')

# List of features — must match training phase
features = [
    'Seniority_Level', 'Skill_Count', 'Company_Size_Num',
    'Python', 'SQL', 'Excel', 'Tableau', 'Machine Learning',
    'Power BI', 'R', 'SAS',
    'State_CA', 'State_TX', 'State_NY', 'State_IL', 'State_PA',
    'Industry_Group_Other', 'Industry_Group_Tech'
]

# Streamlit App UI
st.title("Data Analyst Salary Prediction App")

st.write("Provide job-role details to estimate expected salary.")

# --------------------------
# Input collection
# --------------------------
seniority = st.number_input("Seniority Level (0=Junior, 1=Analyst, 2=Senior, 3=Lead, 4=Manager):", 0, 4, 1)
skill_count = st.number_input("Number of Core Skills:", 0, 20, 4)
company_size = st.number_input("Company Size (1 to 7):", 1, 7, 3)

skills = {}
for skill in ['Python', 'SQL', 'Excel', 'Tableau', 'Machine Learning', 'Power BI', 'R', 'SAS']:
    skills[skill] = st.radio(f"{skill} required?", [0, 1], horizontal=True)

states = {}
for state in ['CA', 'TX', 'NY', 'IL', 'PA']:
    states[f'State_{state}'] = st.radio(f"Job in {state}?", [0, 1], horizontal=True)

industry = {}
for ig in ['Industry_Group_Other', 'Industry_Group_Tech']:
    industry[ig] = st.radio(f"{ig.replace('Industry_Group_', 'Industry: ')}", [0, 1], horizontal=True)

# --------------------------
# Combine all inputs properly
# --------------------------
input_dict = {
    'Seniority_Level': seniority,
    'Skill_Count': skill_count,
    'Company_Size_Num': company_size
}
input_dict.update(skills)
input_dict.update(states)
input_dict.update(industry)

# Display what’s captured
st.write("Collected Input Dictionary:", input_dict)

# Build DataFrame correctly (1 row x 18 columns)
input_df = pd.DataFrame([input_dict])

# Add any missing columns and keep order
for col in features:
    if col not in input_df.columns:
        input_df[col] = 0

input_df = input_df[features].apply(pd.to_numeric, errors='coerce').fillna(0)

# Show debug info
st.dataframe(input_df)
st.write("Shape of input data:", input_df.shape)

# --------------------------
# Prediction
# --------------------------
if st.button("Predict Salary"):
    if input_df.shape[1] != len(features):
        st.error(f"Feature mismatch: got {input_df.shape[1]} columns, expected {len(features)}.")
    else:
        prediction = model.predict(input_df)[0]
        st.success(f"Estimated Salary: ${prediction:,.2f}")
