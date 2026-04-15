import streamlit as st
import pickle
import numpy as np
import time

# --- Page Configuration ---
st.set_page_config(page_title="Student Grade Predictor", page_icon="🎓", layout="centered")

# --- Custom CSS for Animation & Styling ---
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stButton>button {
        width: 100%;
        border-radius: 20px;
        height: 3em;
        background-color: #4CAF50;
        color: white;
        transition: 0.3s;
    }
    .stButton>button:hover { background-color: #45a049; transform: scale(1.02); }
    @keyframes fadeIn { from { opacity: 0; } to { opacity: 1; } }
    .reportview-container { animation: fadeIn 2s; }
    </style>
    """, unsafe_allow_html=True)

# --- Load Model ---
@st.cache_resource
def load_model():
    with open('ModelSt.pkl', 'rb') as file:
        return pickle.load(file)

model = load_model()

# --- UI Elements ---
st.title("🎓 Student Performance Predictor")
st.markdown("Enter student details below to predict the **Final Grade**.")

with st.sidebar:
    st.header("About")
    st.info("This tool uses an SVM model to predict student success based on academic habits.")
    st.image("https://cdn-icons-png.flaticon.com/512/3413/3413535.png", width=100)

# --- Input Form ---
with st.container():
    col1, col2 = st.columns(2)
    
    with col1:
        attendance = st.slider("Attendance Rate (%)", 0, 100, 85)
        study_hours = st.number_input("Study Hours Per Week", 0, 168, 15)
        prev_grade = st.number_input("Previous Grade (0-100)", 0, 100, 75)
    
    with col2:
        support = st.selectbox("Parental Support", ["Low", "Medium", "High"])
        activities = st.radio("Extracurricular Activities", ["Yes", "No"])
        gender = st.selectbox("Gender", ["Male", "Female"])

# Map categorical inputs to numerical (Adjust based on your specific encoding)
support_map = {"Low": 0, "Medium": 1, "High": 2}
binary_map = {"Yes": 1, "No": 0, "Male": 0, "Female": 1}

# --- Prediction Logic ---
if st.button("Predict Performance"):
    # Prepare feature array - Ensure the order matches your training set
    # Features: [AttendanceRate, StudyHoursPerWeek, PreviousGrade, Extracurricular, Support, Gender...]
    features = np.array([[attendance, study_hours, prev_grade, 
                          binary_map[activities], support_map[support], binary_map[gender]]])
    
    # Simple animation
    progress_bar = st.progress(0)
    for percent_complete in range(100):
        time.sleep(0.01)
        progress_bar.progress(percent_complete + 1)
    
    # Assuming the model was trained with more features, ensure shape matches
    # This example provides a placeholder for the 11 features detected in your .pkl
    full_features = np.zeros((1, 11)) 
    full_features[0, :6] = features[0] 
    
    prediction = model.predict(full_features)
    
    st.success(f"### Predicted Outcome: {'Pass / High Performance' if prediction[0] == 1 else 'Further Review Needed'}")
    st.balloons()
