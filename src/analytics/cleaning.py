import re

import numpy as np
import pandas as pd


def normalize_column_names(data):
    """Normalize column names into lowercase from snake_case"""

    data = data.copy()

    normalized_columns = []

    for column in data.columns:

        column = str(column).strip().lower()

        column = re.sub(r"[^a-z0-9]+", "_", column)

        column = column.strip("_")

        normalized_columns.append(column)

    if len(normalized_columns) != len(set(normalized_columns)):
        raise ValueError("Column normalization produced duplicate column names ")

    data.columns = normalized_columns

    return data


def handle_missing_values(data, impute_features=False):
    """Handle missing values using column-specific strategies

    Examination scores are intentionally not imputed
    """

    data = data.copy()

    # convert blank text values into missing values
    text_columns = ["name", "grade", "gender", "subject"]

    for column in text_columns:

        if column in data.columns:

            data[column] = data[column].astype("string").str.strip().replace("", pd.NA)

    # fill missing categories
    if "grade" in data.columns:
        data["grade"] = data["grade"].fillna("Unknown")

    # numerical feature columns
    numerical_features = ["study_hours", "attendance", "age"]

    if impute_features:

        for column in numerical_features:

            if column in data.columns:

                median_value = data[column].median()

                if pd.notna(median_value):
                    data[column] = data[column].fillna(median_value)
    return data


def remove_duplicates(data):
    """
    Remove duplicate records

    If a student_id exists, retain the first occurrence of each non-null student Id

    also remove exact duplicate rows
    """

    data = data.copy()

    initial_count = len(data)

    if "student_id" in data.columns:

        valid_ids = data["student_id"].notna()

        duplicate_ids = valid_ids & data["student_id"].duplicated(keep="first")

        data = data.loc[~duplicate_ids].copy()

    # remove exact duplicate rows
    data = data.drop_duplicates(keep="first")

    removed = initial_count - len(data)

    print(f"Duplicate records removed: {removed}")

    return data


def clean_numeric_columns(data):
    """convert numerical columns and invalidate impossible values"""

    data = data.copy()

    numerical_columns = ["score", "study_hours", "attendance", "age"]

    for column in numerical_columns:

        if column not in data.columns:
            continue

        # convert values to numeric
        data[column] = pd.to_numeric(data[column], errors="coerce")

        # replace infinity and negative infinity with NaN
        data[column] = data[column].where(np.isfinite(data[column]), np.nan)

    # valid examination scores: 0 to 100
    if "score" in data.columns:

        invalid_scores = (data["score"] < 0) | (data["score"] > 100)

        data.loc[invalid_scores, "score"] = np.nan

    # study hours cannot be negative
    if "study_hours" in data.columns:

        data.loc[data["study_hours"] < 0, "study_hours"] = np.nan

    # attendance must be between 0 and 1
    if "attendance" in data.columns:

        invalid_attendance = (data["attendance"] < 0) | (data["attendance"] > 100)

        data.loc[invalid_attendance, "attendance"] = np.nan

    # age must be positive
    if "age" in data.columns:

        data.loc[data["age"] <= 0, "age"] = np.nan

    return data


def standardize_categories(data):
    """Normalize categorical values"""

    data = data.copy()

    # standardize  grade values
    if "grade" in data.columns:

        grade = data["grade"].astype("string").str.strip().str.upper()

        # Normalize labels such as GRADE A to A
        grade = grade.str.replace(r"^GRADE\s+", "", regex=True)

        data["grade"] = grade

    # standardize gender labels without assuming
    # that every dataset uses the same categories
    if "gender" in data.columns:

        data["gender"] = (
            data["gender"]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.title()
        )

    # normalize subject labels
    if "subject" in data.columns:

        data["subject"] = (
            data["subject"]
            .astype("string")
            .str.strip()
            .str.replace(r"\s+", " ", regex=True)
            .str.title()
        )

    return data


def detect_outliers(data, columns=None):
    """
    Detect outliers using the IQR(Inter Quartile Range) method

    Returns a dataframe with outlier flags
    Does not delete or modify observations

    """

    data = data.copy()

    if columns is None:
        columns = ["score", "study_hours", "attendance"]

    outlier_report = {}

    for column in columns:

        if column not in data.columns:
            continue

        values = data[column].dropna()

        if values.empty:
            continue

        q1 = values.quantile(0.25)
        q3 = values.quantile(0.75)

        iqr = q3 - q1

        lower_bound = q1 - (1.5 * iqr)
        upper_bound = q3 + (1.5 * iqr)

        flag_column = f"{column}_outlier"

        data[flag_column] = (data[column] < lower_bound) | (data[column] > upper_bound)

        outlier_report[column] = {
            "Q1": q1,
            "Q3": q3,
            "IQR": iqr,
            "Lower Bound": lower_bound,
            "Upper Bound": upper_bound,
            "Outliers": int(data[flag_column].sum()),
        }

    report = pd.DataFrame(outlier_report).T

    return data, report


def validate_dataset(data):
    """
    Validate the cleaned student dataset.

    Returns a dataframe containing validation issues
    """

    issues = []

    required_columns = {"name", "grade", "score", "study_hours"}

    # required columns
    missing_columns = required_columns - set(data.columns)

    for column in sorted(missing_columns):

        issues.append(
            {"column": column, "Issue": "Missing required Column", "Count": 1}
        )

    # cannot perform the required checks without
    # the core schema
    if missing_columns:
        return pd.DataFrame(issues)

    # 2. Missing names
    missing_names = data["name"].isna().sum()

    if missing_names:

        issues.append(
            {
                "Column": "name",
                "Issue": "Missing student names",
                "Count": int(missing_names),
            }
        )

    # 3. Missing Scores
    missing_scores = data["score"].isna().sum()

    if missing_scores:

        issues.append(
            {
                "Column": "score",
                "Issue": "Missing examination scores",
                "Count": int(missing_scores),
            }
        )

    # 4. Invalid scores
    invalid_scores = ((data["score"] < 0) | (data["score"] > 100)).sum()

    if invalid_scores:

        issues.append(
            {
                "Column": "score",
                "Issue": "Scores outside 0-100",
                "Count": int(invalid_scores),
            }
        )

    # 5. Invalid study hours
    invalid_hours = (data["study_hours"] < 0).sum()

    if invalid_hours:

        issues.append(
            {
                "Column": "study_hours",
                "Issue": "Negative study hours",
                "Count": int(invalid_hours),
            }
        )

    # 6. Invalid grades
    valid_grades = {"A", "B", "C", "D", "E", "UNKNOWN"}

    invalid_grades = (~data["grade"].isin(valid_grades)).sum()

    if invalid_grades:

        issues.append(
            {
                "Column": "grade",
                "Issue": "Unrecognized grade categories",
                "Count": int(invalid_grades),
            }
        )

    # 7. Duplicate ids
    if "student_id" in data.columns:

        duplicate_ids = (
            data["student_id"].notna() & data["student_id"].duplicated()
        ).sum()

        if duplicate_ids:

            issues.append(
                {
                    "Column": "student_id",
                    "Issue": "Duplicate student ids",
                    "Count": int(duplicate_ids),
                }
            )

    # 8. Attendance validation
    if "attendance" in data.columns:

        invalid_attendance = (
            (data["attendance"] < 10) | (data["attendance"] > 100)
        ).sum()

        if invalid_attendance:

            issues.append(
                {
                    "Column": "attendance",
                    "Issue": "Attendance outside the range of 0-100",
                    "Count": int(invalid_attendance),
                }
            )

    return pd.DataFrame(issues, columns=["Column", "Issue", "Count"])


def clean_dataset(data, impute_features=True):
    """
    Execute the complete student data cleaning pipeline.

    Returns:
        cleaned_data: processed Dataframe
        reports: Dictionary containing quality reports
    """

    # preserve original data
    cleaned_data = data.copy()

    # initial value-missing report
    initial_missing = cleaned_data.isnull().sum()

    # step 1. Normalize column names
    cleaned_data = normalize_column_names(cleaned_data)

    # step 2. Clean numerical values
    cleaned_data = clean_numeric_columns(cleaned_data)

    # step 3. Standardize categories
    cleaned_data = standardize_categories(cleaned_data)

    # step 4. Remove duplicate records
    cleaned_data = remove_duplicates(cleaned_data)

    # step 5. Handle missing values
    cleaned_data = handle_missing_values(cleaned_data, impute_features=impute_features)

    # step 6. Detect outliers
    cleaned_data, outlier_report = detect_outliers(cleaned_data)

    # step 7. Validate final dataset
    validation_report = validate_dataset(cleaned_data)

    # final missing value report
    final_missing = cleaned_data.isnull().sum()

    quality_report = (
        pd.DataFrame(
            {
                "Missing Before": initial_missing.reindex(cleaned_data.columns),
                "Missing After": final_missing,
            }
        )
        .fillna(0)
        .astype(int)
    )

    reports = {
        "quality": quality_report,
        "outliers": outlier_report,
        "validation": validation_report,
    }

    return cleaned_data, reports
