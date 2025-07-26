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

st.markdown("---")

st.markdown("Fill in student details to predict dropout risk. 🚸")

defaults = {
    "gender": "Male",
    "age": 16,
    "attendance": 50.0,
    "grade": 5.0,
    "commute": 5.0,
    "income": 10000,
    "support": True,
    "internet": True,
    "predicted": False
}
for key, val in defaults.items():
    if key not in st.session_state:
        st.session_state[key] = val

col1, col2 = st.columns(2)

with col1:
    st.session_state.gender = st.selectbox("Gender", ["Male", "Female"], index=0 if st.session_state.gender == "Male" else 1)
    st.session_state.age = st.slider("Age", 13, 19, value=st.session_state.age)
    st.session_state.attendance = st.slider("Attendance (%)", 30.0, 100.0, step=0.5, value=st.session_state.attendance)
    st.session_state.internet = st.selectbox("Internet Access", [True, False], index=0 if st.session_state.internet else 1)

with col2:
    st.session_state.grade = st.slider("Grade Average", 1.0, 10.0, step=0.1, value=st.session_state.grade)
    st.session_state.commute = st.slider("Commute Distance (km)", 0.1, 25.0, step=0.1, value=st.session_state.commute)
    st.session_state.income = st.number_input("Family Income (INR/month)", 2000, 60000, step=100, value=st.session_state.income)
    st.session_state.support = st.selectbox("Extra Support", [True, False], index=0 if st.session_state.support else 1)

if st.button("🔍 Predict Dropout Risk"):
    gender_encoded = 1 if st.session_state.gender == "Male" else 0
    input_data = np.array([[
        gender_encoded,
        st.session_state.age,
        st.session_state.attendance,
        st.session_state.grade,
        st.session_state.commute,
        st.session_state.income,
        st.session_state.support,
        st.session_state.internet
    ]])
    
    prediction = bool(model.predict(input_data)[0])
    st.session_state.predicted = True
    st.session_state.prediction_result = prediction

if st.session_state.predicted:
    prediction = st.session_state.prediction_result

    if prediction:
        st.error("⚠️ High Dropout Risk")
    else:
        st.success("✅ Low Dropout Risk")

    st.subheader("💡 Suggestions")

    if st.session_state.attendance < 60.0:
        st.warning("📉 Low attendance. Consider engaging parents and offering attendance incentives.")
    
    if st.session_state.grade < 5.0:
        st.warning("📚 Low academic performance. Recommend academic support or tutoring.")
    
    if st.session_state.income < 8000:
        st.warning("💰 Low family income. Explore scholarship or subsidy programs.")
    
    if st.session_state.commute > 10.0:
        st.warning("🚍 Long commute. Suggest looking into transport assistance or flexible learning.")
    
    if not st.session_state.internet:
        st.warning("🌐 No internet access. Consider offline resources or community access centers.")
    
    if not st.session_state.support:
        st.info("🤝 Additional school support may help this student succeed.")
    
    if prediction == False and all([
        st.session_state.attendance >= 90.0,
        st.session_state.grade >= 8.0,
        st.session_state.income > 10000,
        st.session_state.support,
        st.session_state.internet
    ]):
        st.success("🎯 This student is on the right track! Encourage continued effort and support.")

    if st.session_state.predicted:
        st.markdown("### ")
        if st.button("🔁 Reset"):
            for key in defaults:
                st.session_state[key] = defaults[key]
            st.experimental_rerun()

st.markdown("""
<hr style="margin-top: 3rem; border-top: 1px solid #444;">
<div style='display: flex; justify-content: space-between; padding: 20px; color: gray; font-size: 14px;'>
    <div style="flex: 1; text-align: left;">
        <b>The original model was made in Jupyter Notebook.</b><br>
        A copy of that model was deployed in this webpage.
    </div>
    <div style="flex: 1; text-align: center;">
        <b style="color: #ffffff;">🎓 DropShield AI 🛡️</b><br>
        📊 Predicting School Dropout Risk using Machine Learning 🧠<br>Built with ❤️ for student success.
    </div>
    <div style="flex: 1; text-align: right;">
        <b>This webpage and AI model was created by Sarvesh Kore:</b><br>
        📧 ssworld7105@gmail.com<br>📞 +971 563711020
    </div>
</div>
""", unsafe_allow_html=True)
