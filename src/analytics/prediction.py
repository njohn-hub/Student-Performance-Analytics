from pathlib import Path

import joblib
import pandas as pd

# --------------------------------------------------
# MODEL CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "student_score_model.joblib"


# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------


def load_model():
    """
    Load the trained student performance prediction model.

    The saved joblib file may contain either the model directly
    or a dictionary containing the model and additional results.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    loaded_object = joblib.load(MODEL_PATH)

    # If the saved file contains a dictionary,
    # extract the trained model.
    if isinstance(loaded_object, dict):

        if "model" in loaded_object:
            return loaded_object["model"]

        if "trained_model" in loaded_object:
            return loaded_object["trained_model"]

        raise KeyError(
            "Could not find a trained model in the saved "
            "model dictionary."
        )

    # Otherwise, the file contains the model directly.
    return loaded_object


# --------------------------------------------------
# PREPARE STUDENT DATA
# --------------------------------------------------


def prepare_student_data(
    age,
    gender,
    class_level,
    subject,
    term,
    study_hours,
    attendance,
    homework_completion,
    class_participation,
    internet_access,
    extracurricular_activity,
    parent_education,
):
    """
    Convert student information into a dataframe
    compatible with the trained machine learning model.
    """

    student_data = pd.DataFrame(
        [
            {
                "age": age,
                "gender": gender,
                "class_level": class_level,
                "subject": subject,
                "term": term,
                "study_hours": study_hours,
                "attendance": attendance,
                "homework_completion": homework_completion,
                "class_participation": class_participation,
                "internet_access": internet_access,
                "extracurricular_activity": extracurricular_activity,
                "parent_education": parent_education,
            }
        ]
    )

    return student_data


# --------------------------------------------------
# PREDICT STUDENT SCORE
# --------------------------------------------------


def predict_student_score(model, student_data):
    """
    Predict the student's expected academic score
    """

    prediction = model.predict(student_data)

    return float(prediction[0])


def get_performance_category(score):
    """
    Classify a predicted score into a performance category.
    """

    if score >= 80:
        return "Excellent"

    if score >= 70:
        return "Good"

    if score >= 60:
        return "Satisfactory"

    if score >= 50:
        return "Pass"

    return "Needs Academic Support"
