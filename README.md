# EduPredict — Academic Performance Predictor

EduPredict is a Streamlit app for exploring student academic records and estimating a performance category from six indicators. It also provides dataset exploration, model insights, and downloadable filtered records.

> **Educational prototype:** Predictions may be wrong and should not be used alone to rank students, punish students, or make high-stakes academic decisions.

## Features

- **Overview:** Student-record counts, performance distribution, and holdout accuracy.
- **Performance Predictor:** Estimate a performance category from attendance, assignment score, internal mark, study hours, previous GPA, and class participation.
- **Student Explorer:** Filter and inspect records, view summary statistics, and download filtered data.
- **Model Insights:** Review feature importance, evaluation results, and model limitations.

The app trains a Random Forest classifier from the CSV dataset when it starts. The `.pkl` files in the project are not required by the current app.

## Requirements

- Python 3.9 or later
- Packages listed in `requirements.txt`
- `student_academic_performance_dataset_final.csv` in the same folder as `app.py`

The CSV must include the six feature columns and a `Performance` target column:

```text
Attendance_Percentage
Assignment_Score
Internal_Mark
Study_Hours_Per_Day
Previous_GPA
Class_Participation
Performance
```

## Run locally (Windows PowerShell)

From the project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit will print a local URL (usually `http://localhost:8501`) to open in your browser.

## Deploy on Streamlit Community Cloud

1. Push `app.py`, `requirements.txt`, and `student_academic_performance_dataset_final.csv` to a GitHub repository.
2. Sign in to [Streamlit Community Cloud](https://share.streamlit.io/) and choose **Create app**.
3. Select the repository and branch, set the app file path to `app.py`, and deploy.

The dataset needs to be present in the repository root beside `app.py`, because the app loads it from that location. Only publish the dataset if you have permission to share it and it contains no information that should remain private.

## Model and limitations

The app uses a scikit-learn pipeline with median imputation and a Random Forest classifier. It creates a stratified 80/20 train/test split at startup and reports accuracy on the held-out portion.

The dataset may include sampled copies of records. If copies of the same record appear in both the training and test sets, the reported accuracy can be overly optimistic and may not reflect performance on new students. Feature importance shows model associations, not causes.
