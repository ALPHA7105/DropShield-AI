import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree

df = pd.read_csv("DropShield AI Dataset.csv")
df["Parental Education"] = df["Parental Education"].fillna(df["Parental Education"].mode()[0])
df.rename(columns={"Attendance (%)": "Attendance","Grade Average": "Grade_Avg","Commute Distance (km)": "Commute_Dist","Family Income (INR/month)": "Income","Parental Education": "Parental_Education","Extra Support": "Support","Internet Access": "Internet","Dropped Out": "Dropped_Out"}, inplace=True)
label_encode = {True: 1, False: 0, "Male": 1, "Female": 0}
df.replace(label_encode,inplace=True)

x_values = df[['Gender','Age','Attendance','Grade_Avg','Commute_Dist','Income','Support','Internet']]
y_values = df[['Dropped_Out']]
standardise = StandardScaler()

x_train, x_test, y_train, y_test = train_test_split(x_values,y_values,test_size=0.25,random_state=10)
standardise = StandardScaler()
x_train_scale = standardise.fit_transform(x_train)
x_test_scale = standardise.transform(x_test)

model = tree.DecisionTreeClassifier(max_depth=7)
model = model.fit(x_train,y_train)
#y_predict = dt.predict(x_test)

st.set_page_config(page_title="DropShield AI", layout="wide", page_icon="🛡️")

st.markdown("""
    <style>
    .stButton > button {
        background-color: black !important;
        color: white !important;
        padding: 1rem 2rem;
        font-size: 18px;
        font-weight: 600;
        border: 2px solid #444 !important;  /* Dark gray border */
        border-radius: 12px;
        cursor: pointer;
        transition: color 0.3s ease, border-color 0.3s ease;
        width: 100%;
    }

    .stButton > button:hover {
        color: #1a73e8 !important;  /* Blue text on hover */
        border-color: #1a73e8 !important;  /* Optional: Blue border on hover */
    }

    .main-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        margin-top: 2rem;
    }
    .center-content {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        margin-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='text-align: center; margin-top: 1rem;'>
        <h1 style='font-size: 60px; margin-bottom: 0;'>🎓 DropShield AI 🛡️</h1>
        <h3 style='margin-top: 0;'>📊 Predicting School Dropout Risk using Machine Learning 🧠</h3>
    </div>
""", unsafe_allow_html=True)

st.markdown("Fill in student details to predict dropout risk. 🚸")

gender = st.selectbox("Gender", ["Male", "Female"])
gender_encoded = 1 if gender == "Male" else 0
age = st.slider("Age", 13, 19)
attendance = st.slider("Attendance (%)", 30.0, 100.0)
grade = st.slider("Grade Average (1.0 to 10.0)", 1.0, 10.0)
commute = st.slider("Commute Distance (km)", 0.1, 25.0)
income = st.number_input("Family Income (INR/month)", 2000, 60000)
support = st.selectbox("Extra Support", [True, False])
internet = st.selectbox("Internet Access", [True, False])

if st.button("Predict Dropout Risk"):
    try:
        input_data = np.array([[gender_encoded, age, attendance, grade,
                                commute, income, support, internet]])

        prediction = model.predict(input_data)[0]

        if prediction:
            st.error("⚠️ High Dropout Risk")
        else:
            st.success("✅ Low Dropout Risk")

    except Exception as e:
        st.warning(f"⚠️ Something went wrong: {e}")


st.subheader("💡 Suggestions")

if attendance < 60.0:
    st.warning("📉 Low attendance. Consider engaging parents and offering attendance incentives.")
    
if grade < 5.0:
    st.warning("📚 Low academic performance. Recommend academic support or tutoring.")
    
if income < 8000:
    st.warning("💰 Low family income. Explore scholarship or subsidy programs.")
    
if commute > 10.0:
    st.warning("🚍 Long commute. Suggest looking into transport assistance or flexible learning.")
    
if not internet:
    st.warning("🌐 No internet access. Consider offline resources or community access centers.")
    
if not support:
    st.info("🤝 Additional school support may help this student succeed.")
    
if prediction == False and all([
    attendance >= 90.0,
    grade >= 8.0,
    income > 10000,
    support,
    internet
]):
    st.success("🎯 This student is on the right track! Encourage continued effort and support.")

st.markdown("---")

st.markdown("""
<hr style="margin-top: 3rem; border-top: 1px solid #444;">
<div style='display: flex; justify-content: space-between; padding: 20px; color: gray; font-size: 14px;'>
    <div style="flex: 1; text-align: left;">
        <b>📊 School Dropout Risk Prediction using Machine Learning 🧠</b><br>
        Explore the mysteries of the universe with us.
    </div>
    <div style="flex: 1; text-align: center;">
        <b style="color: #ffffff;">🎓 DropShield AI 🛡️</b><br>
        Your gateway to space knowledge
    </div>
    <div style="flex: 1; text-align: right;">
        <b>This webpage and AI model was created by Sarvesh Kore:</b><br>
        📧 ssworld7105@gmail.com<br>📞 +971 563711020
    </div>
</div>
""", unsafe_allow_html=True)
