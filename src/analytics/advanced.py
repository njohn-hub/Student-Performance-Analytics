
from pathlib import Path

import pandas as pd


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PASS_MARK = 50
LOW_ATTENDANCE = 75
LOW_HOMEWORK = 50

NUMERIC_FEATURES = [
    "age",
    "study_hours",
    "attendance",
    "homework_completion",
    "class_participation",
    "score",
]


# --------------------------------------------------
# CORRELATION ANALYSIS
# --------------------------------------------------

def analyze_correlations(data):
    """Analyze relationships between numerical variables."""

    available_columns = [
        column
        for column in NUMERIC_FEATURES
        if column in data.columns
    ]

    numeric_data = data[available_columns].copy()

    for column in available_columns:
        numeric_data[column] = pd.to_numeric(
            numeric_data[column],
            errors="coerce",
        )

    correlation_matrix = numeric_data.corr(method="pearson")

    if "score" in correlation_matrix.columns:
        score_correlations = (
            correlation_matrix["score"]
            .drop("score")
            .dropna()
            .sort_values(
                key=lambda values: values.abs(),
                ascending=False,
            )
        )
    else:
        score_correlations = pd.Series(dtype=float)

    return correlation_matrix, score_correlations


# --------------------------------------------------
# SUBJECT PERFORMANCE
# --------------------------------------------------

def analyze_subject_performance(data):
    """Compare student performance across subjects."""

    if "subject" not in data.columns:
        return pd.DataFrame()

    result = (
        data.groupby("subject", observed=True)
        .agg(
            Students=("score", "count"),
            Average_Score=("score", "mean"),
            Median_Score=("score", "median"),
            Highest_Score=("score", "max"),
            Lowest_Score=("score", "min"),
            Pass_Rate=(
                "score",
                lambda scores: scores.ge(PASS_MARK).mean() * 100,
            ),
        )
        .round(2)
        .sort_values("Average_Score", ascending=False)
    )

    return result


# --------------------------------------------------
# CLASS LEVEL PERFORMANCE
# --------------------------------------------------

def analyze_class_performance(data):
    """Compare performance across school class levels."""

    if "class_level" not in data.columns:
        return pd.DataFrame()

    result = (
        data.groupby("class_level", observed=True)
        .agg(
            Students=("score", "count"),
            Average_Score=("score", "mean"),
            Median_Score=("score", "median"),
            Pass_Rate=(
                "score",
                lambda scores: scores.ge(PASS_MARK).mean() * 100,
            ),
        )
        .round(2)
    )

    return result


# --------------------------------------------------
# TERM PERFORMANCE
# --------------------------------------------------

def analyze_term_performance(data):
    """Analyze performance across academic terms."""

    if "term" not in data.columns:
        return pd.DataFrame()

    result = (
        data.groupby("term", observed=True)
        .agg(
            Students=("score", "count"),
            Average_Score=("score", "mean"),
            Median_Score=("score", "median"),
            Pass_Rate=(
                "score",
                lambda scores: scores.ge(PASS_MARK).mean() * 100,
            ),
        )
        .round(2)
    )

    return result


# --------------------------------------------------
# STUDENTS NEEDING ACADEMIC SUPPORT
# --------------------------------------------------

def identify_students_for_support(data):
    """
    Identify student records that may benefit from additional support.

    These are screening indicators, not definitive diagnoses.
    """

    result = data.copy()

    result["score"] = pd.to_numeric(
        result["score"],
        errors="coerce",
    )

    if "attendance" in result.columns:
        result["attendance"] = pd.to_numeric(
            result["attendance"],
            errors="coerce",
        )

    if "homework_completion" in result.columns:
        result["homework_completion"] = pd.to_numeric(
            result["homework_completion"],
            errors="coerce",
        )

    result["support_reasons"] = ""

    conditions = {
        "Below pass mark": result["score"] < PASS_MARK,
    }

    if "attendance" in result.columns:
        conditions["Low attendance"] = (
            result["attendance"] < LOW_ATTENDANCE
        )

    if "homework_completion" in result.columns:
        conditions["Low homework completion"] = (
            result["homework_completion"] < LOW_HOMEWORK
        )

    for reason, condition in conditions.items():
        result.loc[
            condition.fillna(False),
            "support_reasons",
        ] += reason + "; "

    result["support_reasons"] = (
        result["support_reasons"].str.rstrip("; ")
    )

    students = result[
        result["support_reasons"] != ""
    ].copy()

    preferred_columns = [
        "student_id",
        "name",
        "class_level",
        "subject",
        "term",
        "score",
        "attendance",
        "homework_completion",
        "support_reasons",
    ]

    available_columns = [
        column
        for column in preferred_columns
        if column in students.columns
    ]

    return students[available_columns].sort_values(
        "score",
        ascending=True,
        na_position="last",
    )


# --------------------------------------------------
# AUTOMATED INSIGHTS
# --------------------------------------------------

def generate_insights(
    data,
    correlations,
    subjects,
    classes,
    terms,
    support_students,
):
    """Generate readable insights from analytics results."""

    insights = []

    average_score = data["score"].mean()
    pass_rate = data["score"].ge(PASS_MARK).mean() * 100

    insights.append(
        f"Overall average score: {average_score:.2f}%."
    )

    insights.append(
        f"Overall pass rate: {pass_rate:.2f}% "
        f"(pass mark: {PASS_MARK})."
    )

    if not subjects.empty:
        best_subject = subjects["Average_Score"].idxmax()
        weakest_subject = subjects["Average_Score"].idxmin()

        insights.append(
            f"Highest-performing subject: {best_subject} "
            f"({subjects.loc[best_subject, 'Average_Score']:.2f}%)."
        )

        insights.append(
            f"Lowest-performing subject: {weakest_subject} "
            f"({subjects.loc[weakest_subject, 'Average_Score']:.2f}%)."
        )

    if not classes.empty:
        best_class = classes["Average_Score"].idxmax()
        weakest_class = classes["Average_Score"].idxmin()

        insights.append(
            f"Highest-performing class level: {best_class}."
        )

        insights.append(
            f"Lowest-performing class level: {weakest_class}."
        )

    if not terms.empty:
        best_term = terms["Average_Score"].idxmax()
        weakest_term = terms["Average_Score"].idxmin()

        insights.append(
            f"Highest average score by term: {best_term}."
        )

        insights.append(
            f"Lowest average score by term: {weakest_term}."
        )

    if not correlations.empty:
        strongest_feature = correlations.index[0]
        correlation_value = correlations.iloc[0]

        insights.append(
            f"Strongest numerical association with score: "
            f"{strongest_feature} "
            f"(Pearson correlation: {correlation_value:.2f})."
        )

    insights.append(
        f"{len(support_students)} student records meet at least "
        "one academic support screening criterion."
    )

    insights.append(
        "Correlation indicates association, not causation. "
        "Support flags should be reviewed by teachers "
        "before decisions are made."
    )

    return insights


# --------------------------------------------------
# RUN COMPLETE ADVANCED ANALYSIS
# --------------------------------------------------

def run_advanced_analytics(data, reports_dir):
    """Run advanced analytics and export reports."""

    reports_dir = Path(reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    data = data.copy()

    if "score" not in data.columns:
        raise ValueError(
            "The dataset must contain a score column."
        )

    data["score"] = pd.to_numeric(
        data["score"],
        errors="coerce",
    )

    data = data.dropna(subset=["score"])

    if data.empty:
        raise ValueError(
            "No valid score records are available for advanced analysis."
        )

    correlation_matrix, correlations = analyze_correlations(data)

    subject_performance = analyze_subject_performance(data)
    class_performance = analyze_class_performance(data)
    term_performance = analyze_term_performance(data)

    support_students = identify_students_for_support(data)

    insights = generate_insights(
        data,
        correlations,
        subject_performance,
        class_performance,
        term_performance,
        support_students,
    )

    # EXPORT REPORTS

    correlation_matrix.to_csv(
        reports_dir / "correlation_matrix.csv"
    )

    correlations.rename(
        "Correlation_With_Score"
    ).to_csv(
        reports_dir / "score_correlations.csv"
    )

    subject_performance.to_csv(
        reports_dir / "subject_performance.csv"
    )

    class_performance.to_csv(
        reports_dir / "class_performance.csv"
    )

    term_performance.to_csv(
        reports_dir / "term_performance.csv"
    )

    support_students.to_csv(
        reports_dir / "students_for_support.csv",
        index=False,
    )

    with open(
        reports_dir / "advanced_insights.txt",
        "w",
        encoding="utf-8",
    ) as file:
        file.write("STUDENT PERFORMANCE ANALYTICS\n")
        file.write("=" * 50 + "\n\n")

        for number, insight in enumerate(insights, start=1):
            file.write(f"{number}. {insight}\n")

    # DISPLAY RESULTS

    print("\nADVANCED ANALYTICS")
    print("=" * 50)

    print("\nCORRELATIONS WITH SCORE")
    print(correlations)

    print("\nSUBJECT PERFORMANCE")
    print(subject_performance)

    print("\nCLASS LEVEL PERFORMANCE")
    print(class_performance)

    print("\nTERM PERFORMANCE")
    print(term_performance)

    print("\nSTUDENTS IDENTIFIED FOR SUPPORT")
    print(f"Total: {len(support_students)}")

    print("\nAUTOMATED INSIGHTS")

    for number, insight in enumerate(insights, start=1):
        print(f"{number}. {insight}")

    print("\nAdvanced reports exported successfully.")

    return {
        "correlations": correlations,
        "correlation_matrix": correlation_matrix,
        "subjects": subject_performance,
        "classes": class_performance,
        "terms": term_performance,
        "support_students": support_students,
        "insights": insights,
    }
