
import sys
from pathlib import Path

# --------------------------------------------------
# PROJECT PATH CONFIGURATION
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

# Make the project root accessible to Python
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))


# --------------------------------------------------
# IMPORTS
# --------------------------------------------------

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from src.analytics.prediction import (
    get_performance_category,
    load_model,
    predict_student_score,
    prepare_student_data,
)

# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DATA_PATH = BASE_DIR / "data" / "processed" / "students_cleaned.csv"

PASS_MARK = 50

st.set_page_config(
    page_title="Student Performance Analytics",
    page_icon="📊",
    layout="wide",
)

sns.set_theme(style="whitegrid")


# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

@st.cache_data
def load_data(path):
    return pd.read_csv(path)


st.title("📊 Student Performance Analytics")
st.caption(
    "An interactive dashboard for exploring academic performance "
    "and student learning patterns."
)

if not DATA_PATH.exists():
    st.error("Cleaned dataset not found.")
    st.info("Run `poetry run python main.py` first.")
    st.stop()

data = load_data(DATA_PATH)

required_columns = [
    "name",
    "grade",
    "class_level",
    "subject",
    "term",
    "study_hours",
    "attendance",
    "score",
]

missing = [
    column for column in required_columns
    if column not in data.columns
]

if missing:
    st.error(f"Missing columns: {', '.join(missing)}")
    st.stop()

# Convert numeric fields safely
numeric_columns = [
    "age",
    "study_hours",
    "attendance",
    "homework_completion",
    "class_participation",
    "score",
]

for column in numeric_columns:
    if column in data.columns:
        data[column] = pd.to_numeric(
            data[column],
            errors="coerce",
        )


# --------------------------------------------------
# SIDEBAR FILTERS
# --------------------------------------------------

st.sidebar.header("Dashboard Filters")

def add_filter(label, column):
    options = sorted(data[column].dropna().unique().tolist())

    return st.sidebar.multiselect(
        label,
        options=options,
        default=options,
    )


selected_classes = add_filter("Class Level", "class_level")
selected_subjects = add_filter("Subject", "subject")
selected_terms = add_filter("Term", "term")
selected_grades = add_filter("Academic Grade", "grade")

if "gender" in data.columns:
    selected_genders = add_filter("Gender", "gender")
else:
    selected_genders = None

min_score = float(data["score"].min())
max_score = float(data["score"].max())

score_range = st.sidebar.slider(
    "Score Range",
    min_value=min_score,
    max_value=max_score,
    value=(min_score, max_score),
)

filtered_data = data[
    data["class_level"].isin(selected_classes)
    & data["subject"].isin(selected_subjects)
    & data["term"].isin(selected_terms)
    & data["grade"].isin(selected_grades)
    & data["score"].between(*score_range)
].copy()

if selected_genders is not None:
    filtered_data = filtered_data[
        filtered_data["gender"].isin(selected_genders)
    ]


# --------------------------------------------------
# PERFORMANCE OVERVIEW
# --------------------------------------------------

st.subheader("Performance Overview")

if filtered_data.empty:
    st.warning("No students match your selected filters.")
    st.stop()

total_students = len(filtered_data)
average_score = filtered_data["score"].mean()
pass_rate = (
    filtered_data["score"].ge(PASS_MARK).mean() * 100
)
highest_score = filtered_data["score"].max()

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Students", total_students)
col2.metric("Average Score", f"{average_score:.1f}%")
col3.metric("Pass Rate", f"{pass_rate:.1f}%")
col4.metric("Highest Score", f"{highest_score:.1f}")


# --------------------------------------------------
# ACADEMIC PERFORMANCE
# --------------------------------------------------

st.divider()
st.subheader("Academic Performance")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Score Distribution**")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.histplot(
        data=filtered_data,
        x="score",
        bins=15,
        kde=True,
        ax=ax,
    )

    ax.axvline(
        PASS_MARK,
        linestyle="--",
        label="Pass Mark",
    )

    ax.set_xlabel("Score")
    ax.set_ylabel("Number of Students")
    ax.legend()

    st.pyplot(fig)
    plt.close(fig)


with col2:
    st.markdown("**Average Score by Subject**")

    subject_scores = (
        filtered_data.groupby("subject")["score"]
        .mean()
        .sort_values(ascending=False)
    )

    st.bar_chart(subject_scores)


col1, col2 = st.columns(2)

with col1:
    st.markdown("**Average Score by Class Level**")

    class_scores = (
        filtered_data.groupby("class_level")["score"]
        .mean()
    )

    st.bar_chart(class_scores)

with col2:
    st.markdown("**Academic Grade Distribution**")

    grade_counts = (
        filtered_data["grade"]
        .value_counts()
        .sort_index()
    )

    st.bar_chart(grade_counts)


# --------------------------------------------------
# LEARNING PATTERNS
# --------------------------------------------------

st.divider()
st.subheader("Learning Patterns")

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Study Hours vs Score**")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.scatterplot(
        data=filtered_data,
        x="study_hours",
        y="score",
        hue="grade",
        ax=ax,
    )

    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Score")

    st.pyplot(fig)
    plt.close(fig)


with col2:
    st.markdown("**Attendance vs Score**")

    fig, ax = plt.subplots(figsize=(8, 4))

    sns.scatterplot(
        data=filtered_data,
        x="attendance",
        y="score",
        hue="grade",
        ax=ax,
    )

    ax.set_xlabel("Attendance (%)")
    ax.set_ylabel("Score")

    st.pyplot(fig)
    plt.close(fig)


# --------------------------------------------------
# HOMEWORK AND PARTICIPATION
# --------------------------------------------------

st.divider()
st.subheader("Homework and Classroom Engagement")

col1, col2 = st.columns(2)

if "homework_completion" in filtered_data.columns:
    with col1:
        st.markdown("**Homework Completion vs Score**")

        fig, ax = plt.subplots(figsize=(8, 4))

        sns.regplot(
            data=filtered_data,
            x="homework_completion",
            y="score",
            scatter_kws={"alpha": 0.5},
            ax=ax,
        )

        ax.set_xlabel("Homework Completion (%)")
        ax.set_ylabel("Score")

        st.pyplot(fig)
        plt.close(fig)

if "class_participation" in filtered_data.columns:
    with col2:
        st.markdown("**Class Participation vs Score**")

        participation = (
            filtered_data.groupby("class_participation")["score"]
            .mean()
        )

        st.bar_chart(participation)


# --------------------------------------------------
# PASS / FAIL ANALYSIS
# --------------------------------------------------

st.divider()
st.subheader("Pass / Fail Analysis")

filtered_data["result"] = filtered_data["score"].apply(
    lambda score: "Pass" if score >= PASS_MARK else "Fail"
)

result_counts = (
    filtered_data["result"]
    .value_counts()
    .reindex(["Pass", "Fail"], fill_value=0)
)

col1, col2 = st.columns(2)

with col1:
    st.markdown("**Student Results**")
    st.bar_chart(result_counts)

with col2:
    passed = result_counts["Pass"]
    failed = result_counts["Fail"]

    st.metric("Students Passed", passed)
    st.metric("Students Failed", failed)


# --------------------------------------------------
# STUDENT CHARACTERISTICS
# --------------------------------------------------

st.divider()
st.subheader("Student Characteristics")

col1, col2 = st.columns(2)

if "gender" in filtered_data.columns:
    with col1:
        st.markdown("**Students by Gender**")
        st.bar_chart(filtered_data["gender"].value_counts())

if "internet_access" in filtered_data.columns:
    with col2:
        st.markdown("**Internet Access**")
        st.bar_chart(
            filtered_data["internet_access"].value_counts()
        )


# --------------------------------------------------
# MACHINE LEARNING PREDICTION
# --------------------------------------------------

st.divider()
st.subheader("🤖 Student Score Prediction")

st.write(
    "Enter student information below to estimate their "
    "expected academic score using the trained machine "
    "learning model."
)

try:
    model = load_model()

except FileNotFoundError:
    st.error(
        "Trained machine learning model not found. "
        "Run the data pipeline first."
    )

    model = None


if model is not None:

    with st.form("prediction_form"):

        col1, col2, col3 = st.columns(3)

        with col1:

            age = st.number_input(
                "Age",
                min_value=10,
                max_value=25,
                value=16,
            )

            gender = st.selectbox(
                "Gender",
                sorted(data["gender"].dropna().unique()),
            )

            class_level = st.selectbox(
                "Class Level",
                sorted(
                    data["class_level"]
                    .dropna()
                    .unique()
                ),
            )

            subject = st.selectbox(
                "Subject",
                sorted(
                    data["subject"]
                    .dropna()
                    .unique()
                ),
            )

        with col2:

            term = st.selectbox(
                "Term",
                sorted(data["term"].dropna().unique()),
            )

            study_hours = st.number_input(
                "Weekly Study Hours",
                min_value=0.0,
                max_value=40.0,
                value=4.0,
                step=0.5,
            )

            attendance = st.number_input(
                "Attendance (%)",
                min_value=0.0,
                max_value=100.0,
                value=85.0,
            )

            homework_completion = st.number_input(
                "Homework Completion (%)",
                min_value=0.0,
                max_value=100.0,
                value=80.0,
            )

        with col3:

            class_participation = st.number_input(
                "Class Participation",
                min_value=1,
                max_value=5,
                value=3,
            )

            internet_access = st.selectbox(
                "Internet Access",
                sorted(
                    data["internet_access"]
                    .dropna()
                    .unique()
                ),
            )

            extracurricular_options = (
                data["extracurricular_activity"]
                .dropna()
                .unique()
                .tolist()
            )

            extracurricular_activity = st.selectbox(
                "Extracurricular Activity",
                sorted(extracurricular_options),
            )

            parent_education = st.selectbox(
                "Parent Education",
                sorted(
                    data["parent_education"]
                    .dropna()
                    .unique()
                ),
            )

        submitted = st.form_submit_button(
            "Predict Student Score"
        )

    if submitted:

        student_data = prepare_student_data(
            age=age,
            gender=gender,
            class_level=class_level,
            subject=subject,
            term=term,
            study_hours=study_hours,
            attendance=attendance,
            homework_completion=homework_completion,
            class_participation=class_participation,
            internet_access=internet_access,
            extracurricular_activity=extracurricular_activity,
            parent_education=parent_education,
        )

        predicted_score = predict_student_score(
            model,
            student_data,
        )

        category = get_performance_category(
            predicted_score
        )

        st.divider()
        st.subheader("Prediction Result")

        result_col1, result_col2 = st.columns(2)

        with result_col1:

            st.metric(
                "Predicted Score",
                f"{predicted_score:.1f}%",
            )

        with result_col2:

            st.metric(
                "Performance Category",
                category,
            )

        if predicted_score < PASS_MARK:

            st.warning(
                "⚠️ This student may benefit from "
                "additional academic support."
            )

        elif predicted_score >= 80:

            st.success(
                "🌟 The model predicts strong academic "
                "performance."
            )

        else:

            st.info(
                "This student is predicted to meet the "
                "expected academic performance level."
            )


# --------------------------------------------------
# STUDENT RECORDS
# --------------------------------------------------

st.divider()
st.subheader("Student Records")

search = st.text_input("Search students by name")

student_records = filtered_data.copy()

if search:
    student_records = student_records[
        student_records["name"].str.contains(
            search,
            case=False,
            na=False,
        )
    ]

st.dataframe(
    student_records,
    width='stretch',
    hide_index=True,
)

st.caption(
    f"Showing {len(student_records)} of "
    f"{len(filtered_data)} filtered student records."
)


# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------

csv = filtered_data.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Filtered Student Data",
    data=csv,
    file_name="student_performance_report.csv",
    mime="text/csv",
)

st.divider()

st.caption(
    "Student Performance Analytics | Built with Python, "
    "Pandas, Matplotlib, Seaborn and Streamlit"
)
