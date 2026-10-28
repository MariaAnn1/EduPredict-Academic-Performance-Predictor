import streamlit as st
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# EDUPREDICT - STUDENT ACADEMIC PERFORMANCE ANALYTICS
# =========================================================

st.set_page_config(
    page_title="EduPredict | Academic Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

BASE_DIR = Path(__file__).parent

DATA_FILE = BASE_DIR / "student_academic_performance_dataset_final.csv"

FEATURES = [
    "Attendance_Percentage",
    "Assignment_Score",
    "Internal_Mark",
    "Study_Hours_Per_Day",
    "Previous_GPA",
    "Class_Participation"
]

TARGET = "Performance"


# =========================================================
# PROFESSIONAL DESIGN
# =========================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

.stApp {
    background: #F5F7FC;
    color: #17233C;
    font-family: 'DM Sans', sans-serif;
}

.block-container {
    max-width: 1450px;
    padding-top: 2rem;
    padding-bottom: 3rem;
}

[data-testid="stSidebar"] {
    background: #14223D;
}

[data-testid="stSidebar"] * {
    color: #FFFFFF;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
    color: #C5D0E2;
}

.hero {
    background: linear-gradient(120deg, #14223D, #263F70, #4263EB);
    padding: 42px;
    border-radius: 22px;
    color: white;
    margin-bottom: 26px;
    box-shadow: 0 12px 32px rgba(30, 49, 88, 0.12);
}

.hero h1 {
    font-family: 'Manrope', sans-serif;
    font-size: 42px;
    font-weight: 800;
    color: white;
    margin: 12px 0;
}

.hero p {
    color: #E0E8FF;
    font-size: 17px;
    max-width: 750px;
}

.eyebrow {
    color: #C6D5FF;
    letter-spacing: 3px;
    font-size: 12px;
    font-weight: 700;
}

.section-title {
    font-family: 'Manrope', sans-serif;
    font-size: 24px;
    font-weight: 800;
    color: #17233C;
    margin-top: 20px;
}

.section-description {
    color: #718096;
    margin-bottom: 20px;
}

div[data-testid="stMetric"] {
    background: white;
    padding: 22px;
    border-radius: 16px;
    border: 1px solid #E5EAF3;
    box-shadow: 0 4px 14px rgba(20, 34, 61, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #718096;
}

div[data-testid="stMetricValue"] {
    color: #17233C;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}

.card {
    background: white;
    padding: 23px;
    border-radius: 17px;
    border: 1px solid #E5EAF3;
    margin-bottom: 15px;
    min-height: 145px;
}

.card h3 {
    color: #17233C;
    font-size: 19px;
}

.card p {
    color: #718096;
    line-height: 1.7;
}

div.stButton > button[kind="primary"] {
    background: #3155CC;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 12px;
    font-weight: 700;
}

div.stButton > button[kind="primary"]:hover {
    background: #233F9E;
}

.stProgress > div > div > div > div {
    background-color: #4263EB;
}

hr {
    border-color: #E3E8F1;
}

.small-note {
    color: #718096;
    font-size: 13px;
    line-height: 1.7;
}

.result-box {
    background: white;
    border: 1px solid #E5EAF3;
    padding: 25px;
    border-radius: 18px;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# LOAD DATASET
# =========================================================

@st.cache_data
def load_dataset(path, modified_time):
    return pd.read_csv(path)


if not DATA_FILE.exists():

    st.error("Dataset not found!")

    st.markdown("""
    ### How to fix this

    Place this CSV file in the same folder as `app.py`:

    `student_academic_performance_dataset_3000_rows.csv`

    Your folder should contain:

    - app.py
    - student_academic_performance_dataset_3000_rows.csv

    Then restart the website.
    """)

    st.stop()


df = load_dataset(
    str(DATA_FILE),
    DATA_FILE.stat().st_mtime
)


# =========================================================
# VALIDATE DATASET
# =========================================================

missing_columns = [
    column for column in FEATURES + [TARGET]
    if column not in df.columns
]

if missing_columns:

    st.error(
        "The dataset is missing these columns: "
        + ", ".join(missing_columns)
    )

    st.stop()


for column in FEATURES:
    df[column] = pd.to_numeric(df[column], errors="coerce")

df[TARGET] = df[TARGET].astype("string").str.strip()

df = df.dropna(subset=[TARGET])
df = df[df[TARGET] != ""]


if len(df) < 10 or df[TARGET].nunique() < 2:
    st.error(
        "The dataset needs at least 10 valid records "
        "and at least 2 performance categories."
    )
    st.stop()


# =========================================================
# TRAIN MODEL AUTOMATICALLY
# =========================================================

@st.cache_resource
def train_model(data):

    X = data[FEATURES].copy()
    y = data[TARGET].astype(str).copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=150,
                random_state=42,
                class_weight="balanced",
                min_samples_leaf=2
            )
        )
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, X_test, y_test


try:

    model, accuracy, X_test, y_test = train_model(df)

except Exception as error:

    st.error("The model could not be trained.")
    st.code(str(error))

    st.info(
        "Check the dataset's Performance labels and "
        "the values in the six input columns."
    )

    st.stop()


classes = list(model.classes_)


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

with st.sidebar:

    st.markdown("## 🎓 EduPredict")

    st.caption("ACADEMIC INTELLIGENCE PLATFORM")

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Performance Predictor",
            "Student Explorer",
            "Model Insights",
            "About EduPredict"
        ]
    )

    st.markdown("---")

    st.markdown("### System status")

    st.success("Dataset connected")

    st.success("Model trained")

    st.caption(f"{len(df):,} records loaded")

    st.markdown("---")

    st.caption(
        "Helping students understand academic patterns "
        "through machine learning."
    )


# =========================================================
# REUSABLE PAGE HEADER
# =========================================================

def page_header(title, description, label="EDUPREDICT"):

    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">{label}</div>
            <h1>{title}</h1>
            <p>{description}</p>
        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title, description=None):

    

    if description:
        st.markdown(
            f'<div class="section-description">'
            f'{description}</div>',
            unsafe_allow_html=True
        )


# =========================================================
# PAGE 1: OVERVIEW
# =========================================================

if page == "Overview":

    page_header(
        "Make progress visible.",
        "Explore academic patterns, understand learning habits, "
        "and discover insights that can support student success.",
        "STUDENT SUCCESS PLATFORM  /  ACADEMIC ANALYTICS"
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Student records",
        f"{len(df):,}"
    )

    col2.metric(
        "Learning indicators",
        "06"
    )

    col3.metric(
        "Performance groups",
        df[TARGET].nunique()
    )

    col4.metric(
        "Holdout accuracy",
        f"{accuracy:.1%}"
    )

    st.write("")

    left, right = st.columns([1.2, 0.8])

    with left:

        section_title(
            "Performance distribution",
            "Number of records in each performance category."
        )

        distribution = (
            df[TARGET]
            .value_counts()
            .rename_axis("Performance")
            .to_frame("Students")
        )

        st.bar_chart(distribution)

    with right:

        section_title(
            "What we analyse",
            "Six indicators contribute to the prediction."
        )

        st.markdown("""
        <div class="card">
            <h3>📅 Attendance</h3>
            <p>Class attendance percentage.</p>
        </div>

        <div class="card">
            <h3>📝 Assessments</h3>
            <p>Assignment scores and internal marks.</p>
        </div>

        <div class="card">
            <h3>📚 Learning habits</h3>
            <p>Daily study hours and classroom participation.</p>
        </div>

        <div class="card">
            <h3>📈 Academic history</h3>
            <p>Previous GPA as an indicator of past performance.</p>
        </div>
        """, unsafe_allow_html=True)

    section_title(
        "Your academic analytics workspace",
        "Choose a section from the sidebar to explore."
    )

    a, b, c = st.columns(3)

    with a:

        st.markdown("""
        <div class="card">
            <h3>🎯 Predict</h3>
            <p>
            Enter student indicators and explore the predicted
            performance category and class probabilities.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with b:

        st.markdown("""
        <div class="card">
            <h3>🔎 Explore</h3>
            <p>
            Browse student records, filter performance groups,
            and download selected data.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with c:

        st.markdown("""
        <div class="card">
            <h3>🧠 Understand</h3>
            <p>
            Inspect model feature importance, evaluation results,
            and limitations.
            </p>
        </div>
        """, unsafe_allow_html=True)


# =========================================================
# PAGE 2: PERFORMANCE PREDICTOR
# =========================================================

elif page == "Performance Predictor":

    page_header(
        "Student performance predictor",
        "Create a student profile and explore a machine-learning "
        "estimate based on the available academic indicators.",
        "INTERACTIVE PREDICTION WORKSPACE"
    )

    st.info(
        "This tool provides an educational estimate, "
        "not a judgement of a student's ability."
    )

    ranges = {
        "Attendance_Percentage": (
            "Attendance percentage", 0, 100, 1
        ),
        "Assignment_Score": (
            "Assignment score", 0, 100, 1
        ),
        "Internal_Mark": (
            "Internal mark", 0, 100, 1
        ),
        "Study_Hours_Per_Day": (
            "Study hours per day", 0, 24, 0.5
        ),
        "Previous_GPA": (
            "Previous GPA", 0, 10, 0.1
        ),
        "Class_Participation": (
            "Class participation", 0, 10, 1
        )
    }

    with st.form("prediction_form"):

        section_title(
            "Student profile",
            "Adjust the values to explore different scenarios."
        )

        col1, col2 = st.columns(2)

        values = {}

        for index, feature in enumerate(FEATURES):

            label, minimum, maximum, step = ranges[feature]

            series = df[feature].dropna()

            if len(series) > 0:

                minimum = min(minimum, float(series.min()))
                maximum = max(maximum, float(series.max()))

                default = float(series.median())

                default = max(minimum, min(default, maximum))

            else:

                default = (minimum + maximum) / 2

            with (col1 if index % 2 == 0 else col2):

                values[feature] = st.slider(
                    label,
                    min_value=float(minimum),
                    max_value=float(maximum),
                    value=float(default),
                    step=float(step)
                )

        submitted = st.form_submit_button(
            "✨ Predict performance",
            type="primary",
            use_container_width=True
        )

    if submitted:

        student = pd.DataFrame([values])

        prediction = model.predict(student)[0]

        probabilities = model.predict_proba(student)[0]

        confidence = float(np.max(probabilities))

        st.write("")

        section_title(
            "Prediction result",
            "Review the estimate and its probability breakdown."
        )

        left, right = st.columns([0.9, 1.1])

        with left:

            st.markdown("**ESTIMATED PERFORMANCE**")

            st.markdown(
                f"""
                <h1 style="
                    color:#3155CC;
                    font-size:38px;
                    margin:10px 0;
                ">
                    {prediction}
                </h1>
                """,
                unsafe_allow_html=True
            )

            st.metric(
                "Prediction confidence",
                f"{confidence:.1%}"
            )

            st.progress(confidence)

            st.caption(
                "This is the model's estimated probability "
                "for its selected category, not overall accuracy."
            )

           

        with right:

            st.markdown("### Probability breakdown")

            probability_df = pd.DataFrame({
                "Performance category": classes,
                "Probability (%)": probabilities * 100
            }).set_index("Performance category")

            st.bar_chart(probability_df)

        section_title("Student profile summary")

        summary = pd.DataFrame({
            "Indicator": [
                ranges[feature][0]
                for feature in FEATURES
            ],
            "Entered value": [
                values[feature]
                for feature in FEATURES
            ]
        })

        st.dataframe(
            summary,
            use_container_width=True,
            hide_index=True
        )

        if prediction.lower() == "poor":

            st.warning(
                "Consider reviewing study routines, attendance, "
                "assessment feedback, and support needs with a teacher."
            )

        elif prediction.lower() == "average":

            st.info(
                "Consider choosing one or two learning habits "
                "to improve and reviewing progress over time."
            )

        else:

            st.success(
                "The model places this profile in the higher "
                "performance category. Maintain healthy learning habits."
            )

        st.caption(
            "Predictions may be wrong and should not be used alone "
            "to make important decisions about students."
        )


# =========================================================
# PAGE 3: STUDENT EXPLORER
# =========================================================

elif page == "Student Explorer":

    page_header(
        "Explore student records",
        "Filter the dataset, inspect academic indicators, "
        "and download records for further analysis.",
        "DATA EXPLORER"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Total records", f"{len(df):,}")

    c2.metric("Dataset columns", df.shape[1])

    c3.metric(
        "Missing values",
        int(df.isna().sum().sum())
    )

    section_title(
        "Filter records",
        "Choose one or more performance categories."
    )

    categories = sorted(df[TARGET].dropna().unique())

    selected_categories = st.multiselect(
        "Performance category",
        categories,
        default=categories
    )

    filtered_df = df[
        df[TARGET].isin(selected_categories)
    ]

    st.write(f"**Showing {len(filtered_df):,} records**")

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True,
        height=350
    )

    st.download_button(
        "⬇️ Download filtered dataset",
        data=filtered_df.to_csv(index=False).encode("utf-8"),
        file_name="edupredict_filtered_data.csv",
        mime="text/csv"
    )

    st.write("")

    left, right = st.columns(2)

    with left:

        section_title(
            "Explore an indicator",
            "Compare the distribution of an academic variable."
        )

        feature = st.selectbox(
            "Choose indicator",
            FEATURES,
            format_func=lambda x: ranges[x][0]
            if x in ranges else x
        )

        st.histogram if False else None

        st.bar_chart(
            df[feature]
            .dropna()
            .value_counts()
            .sort_index()
            .head(40)
        )

    with right:

        section_title(
            "Summary statistics",
            "Numerical overview of the dataset."
        )

        st.dataframe(
            df[FEATURES].describe().T.round(2),
            use_container_width=True
        )


# =========================================================
# PAGE 4: MODEL INSIGHTS
# =========================================================

elif page == "Model Insights":

    page_header(
        "Understand the model",
        "Explore the algorithm, feature importance, "
        "evaluation results, and responsible use.",
        "MODEL TRANSPARENCY"
    )

    c1, c2, c3 = st.columns(3)

    c1.metric("Algorithm", "Random Forest")

    c2.metric("Training data", "80%")

    c3.metric("Test data", "20%")

    section_title(
        "Feature importance",
        "Relative importance reported by the trained model. "
        "Importance does not establish causation."
    )

    classifier = model.named_steps["classifier"]

    importance_df = pd.DataFrame({
        "Indicator": FEATURES,
        "Importance": classifier.feature_importances_
    }).sort_values(
        "Importance",
        ascending=False
    ).set_index("Indicator")

    st.bar_chart(importance_df)

    section_title(
        "Model evaluation",
        f"Holdout accuracy: {accuracy:.1%}"
    )

    st.markdown("""
    <div class="card">
        <h3>How to interpret the score</h3>
        <p>
        Accuracy measures how many predictions matched the
        labels in the test split. It does not guarantee performance
        on future students.
        </p>

        <h3>Important dataset limitation</h3>
        <p>
        The 3,000-row dataset was created by adding sampled copies
        of records from the original 1,000-row dataset. A random
        split can put copies of the same record into both training
        and testing data, making accuracy overly optimistic.
        </p>

        <h3>Responsible use</h3>
        <p>
        This is an educational prototype. Do not use its predictions
        alone to rank students, punish students, or make high-stakes
        academic decisions.
        </p>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View classification report"):

        predictions = model.predict(X_test)

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0
        )

        st.dataframe(
            pd.DataFrame(report).T.round(3),
            use_container_width=True
        )


# =========================================================
# PAGE 5: ABOUT
# =========================================================

elif page == "About EduPredict":

    page_header(
        "Technology with a student-first purpose.",
        "EduPredict is an academic analytics prototype designed "
        "to explore how machine learning can support understanding "
        "of student performance.",
        "ABOUT THE PROJECT"
    )

    section_title("Project objectives")

    col1, col2 = st.columns(2)

    with col1:

        st.markdown("""
        <div class="card">
            <h3>🎯 Academic prediction</h3>
            <p>
            Estimate a performance category using attendance,
            assignments, internal marks, study hours, previous GPA,
            and class participation.
            </p>
        </div>

        <div class="card">
            <h3>📊 Data-driven insights</h3>
            <p>
            Explore the distribution of performance labels and
            understand patterns in the available records.
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:

        st.markdown("""
        <div class="card">
            <h3>🧠 Machine learning</h3>
            <p>
            Use a Random Forest classifier trained on the supplied
            dataset to estimate performance categories.
            </p>
        </div>

        <div class="card">
            <h3>🤝 Student support</h3>
            <p>
            Encourage reflection on learning habits and
            conversations with teachers, rather than treating
            predictions as fixed outcomes.
            </p>
        </div>
        """, unsafe_allow_html=True)

    section_title("Technology stack")

    st.markdown("""
    | Technology | Purpose |
    |---|---|
    | Python | Application logic |
    | Streamlit | Interactive website |
    | Pandas | Dataset processing |
    | NumPy | Numerical operations |
    | Scikit-learn | Machine learning |
    | Random Forest | Performance classification |
    """)

    st.info(
        "EduPredict is a prototype for educational exploration. "
        "Its predictions depend on the quality and representativeness "
        "of the supplied data."
    )


# =========================================================
# FOOTER
# =========================================================

st.markdown("---")

st.markdown(
    """
    <p style="
        text-align:center;
        color:#718096;
        font-size:13px;
    ">
        EDUPREDICT · Academic Intelligence Platform<br>
        Built for learning, exploration, and student support
    </p>
    """,
    unsafe_allow_html=True
)