import streamlit as st
import joblib
import pandas as pd

# --------------------------------
# PAGE CONFIG
# --------------------------------
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide"
)

st.title("🎓 Student Performance Predictor")
st.write("Predict student performance using Machine Learning")

# --------------------------------
# LOAD MODEL
# --------------------------------
model = joblib.load("model.pkl")

# --------------------------------
# LOAD DATASET
# --------------------------------
data = pd.read_csv("student_data.csv")

# --------------------------------
# SESSION STATE
# --------------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------
# USER INPUTS
# --------------------------------
st.subheader("Enter Student Details")

study_hours = st.number_input(
    "Study Hours",
    min_value=0.0,
    max_value=15.0,
    value=5.0
)

attendance = st.number_input(
    "Attendance (%)",
    min_value=0.0,
    max_value=100.0,
    value=75.0
)

previous_marks = st.number_input(
    "Previous Marks",
    min_value=0.0,
    max_value=100.0,
    value=70.0
)

# --------------------------------
# PREDICT BUTTON
# --------------------------------
if st.button("Predict Performance"):

    prediction = model.predict(
        [[study_hours, attendance, previous_marks]]
    )

    score = prediction[0]

    st.subheader("📈 Prediction Result")

    st.progress(min(int(score), 100))

    st.success(
        f"Predicted Score: {score:.2f}"
    )

    # Grade Prediction

    if score >= 90:

        st.success("🏆 Grade: A+")

    elif score >= 80:

        st.success("🥇 Grade: A")

    elif score >= 70:

        st.info("🥈 Grade: B")

    elif score >= 60:

        st.warning("🥉 Grade: C")

    else:

        st.error("❌ Grade: D")

    # Store History

    st.session_state.history.append(
        {
            "Study Hours": study_hours,
            "Attendance": attendance,
            "Previous Marks": previous_marks,
            "Predicted Score": round(score, 2)
        }
    )

# --------------------------------
# HISTORY TABLE
# --------------------------------
if len(st.session_state.history) > 0:

    st.subheader("📋 Prediction History")

    history_df = pd.DataFrame(
        st.session_state.history
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

# --------------------------------
# VISUALIZATION
# --------------------------------
st.subheader("📊 Dataset Visualization")

col1, col2 = st.columns(2)

with col1:

    st.write("### Study Hours vs Final Score")

    chart1 = data.set_index(
        "StudyHours"
    )["FinalScore"]

    st.line_chart(chart1)

with col2:

    st.write("### Attendance vs Final Score")

    chart2 = data.set_index(
        "Attendance"
    )["FinalScore"]

    st.line_chart(chart2)

# --------------------------------
# DATASET PREVIEW
# --------------------------------
st.subheader("📄 Dataset Preview")

st.dataframe(
    data,
    use_container_width=True
)

# --------------------------------
# FOOTER
# --------------------------------
st.markdown("---")

st.write(
    "Built using Python, Scikit-Learn, Pandas and Streamlit 🚀"
)