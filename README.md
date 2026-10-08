# Student Placement Prediction using Machine Learning

A machine learning-based system that predicts whether a student is likely to be placed based on academic performance, technical skills, internships, projects, aptitude, communication skills, and other employability-related factors.

The project compares multiple classical machine learning classification algorithms and provides an interactive interface for placement prediction.

## Project Overview

Student placement depends on multiple academic and skill-based factors. This project uses historical student data to identify patterns associated with placement outcomes and builds classification models to predict whether a student is likely to be placed.

The system takes student-related attributes as input and produces:

- Placement prediction
- Estimated prediction probability
- Model performance comparison
- Classification metrics
- Confusion matrices
- Exploratory data analysis

## Problem Statement

> Develop a machine learning system capable of predicting student placement outcomes based on academic performance, technical skills, internships, projects, aptitude performance, and other relevant student attributes.

## Objectives

- Analyze factors influencing student placement outcomes.
- Perform exploratory data analysis on student placement data.
- Preprocess and prepare the dataset for machine learning.
- Train multiple classification algorithms.
- Compare model performance using appropriate evaluation metrics.
- Select a suitable model for deployment.
- Develop an interactive web application for placement prediction.

## Machine Learning Models

The project evaluates the following models:

### 1. Logistic Regression

Used as a baseline classification model and provides interpretable probability-based predictions.

### 2. K-Nearest Neighbors

Classifies a student based on similarities with students in the training dataset.

### 3. Decision Tree

Creates interpretable decision rules based on student attributes.

### 4. Support Vector Machine

Finds an optimal decision boundary between placed and non-placed students.

**Random Forest is intentionally not used in this project.**

## Dataset Features

Depending on the dataset used, the model can consider features such as:

| Feature | Description |
|---|---|
| CGPA | Student's cumulative academic performance |
| 10th Percentage | Secondary school percentage |
| 12th Percentage | Higher secondary percentage |
| Backlogs | Number of academic backlogs |
| Internships | Number of internships completed |
| Projects | Number of academic/personal projects |
| Certifications | Number of relevant certifications |
| Coding Score | Technical/coding assessment score |
| Aptitude Score | Aptitude assessment score |
| Communication Score | Communication assessment score |
| Attendance | Academic attendance percentage |
| Placement Status | Target variable |

### Target Variable

```text
0 → Not Placed
1 → Placed
```

## Workflow

```text
                 Dataset
                    |
                    v
             Data Cleaning
                    |
                    v
             Exploratory Data
                 Analysis
                    |
                    v
           Feature Engineering
                    |
                    v
          Train/Test Split
                    |
          +---------+---------+
          |         |         |
          v         v         v
       Logistic    KNN    Decision Tree
      Regression
          |         |         |
          +---------+---------+
                    |
                    v
                   SVM
                    |
                    v
            Model Evaluation
                    |
                    v
          Best Model Selection
                    |
                    v
             Streamlit App
                    |
                    v
          Placement Prediction
```

## Evaluation Metrics

Since this is a classification problem, the models are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### Why multiple metrics?

Accuracy alone may not adequately represent model performance if the dataset contains an imbalance between placed and non-placed students.

Precision measures how many predicted placements were actually placed students, while recall measures how many actual placed students were correctly identified.

## Exploratory Data Analysis

The project includes analysis of relationships between placement outcomes and factors such as:

- CGPA
- Academic percentages
- Internships
- Projects
- Coding performance
- Aptitude performance
- Communication skills
- Backlogs
- Attendance

Visualizations can include:

- Distribution plots
- Count plots
- Box plots
- Correlation heatmaps
- Feature-wise placement comparisons
- Confusion matrices
- ROC curves

## Application

The project includes an interactive Streamlit application.

### Example Input

```text
CGPA                  : 8.2
10th Percentage       : 88
12th Percentage       : 84
Internships           : 2
Projects              : 4
Coding Score          : 78
Aptitude Score        : 82
Communication Score   : 75
Backlogs              : 0
```

### Example Output

```text
Prediction: PLACED

Estimated Model Probability: 82%
```

The displayed probability represents the model's estimated probability based on the patterns learned from the training data. It is not a guarantee of actual employment.

## Project Structure

```text
student-placement-prediction/
│
├── data/
│   └── placement.csv
│
├── notebooks/
│   └── analysis.ipynb
│
├── models/
│   ├── logistic_regression.pkl
│   ├── knn.pkl
│   ├── decision_tree.pkl
│   └── svm.pkl
│
├── src/
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Installation

Clone the repository:

```bash
git clone https://github.com/your-username/student-placement-prediction.git
```

Navigate to the project directory:

```bash
cd student-placement-prediction
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate the environment.

### Windows

```bash
.venv\Scripts\activate
```

### Linux/macOS

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the Streamlit application:

```bash
streamlit run app.py
```

The application will open in your browser.

## Technologies Used

- **Python**
- **Pandas**
- **NumPy**
- **Matplotlib**
- **Seaborn**
- **Scikit-learn**
- **Streamlit**
- **Joblib**

## Model Comparison

The project compares the trained models based on their performance on the test dataset.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | — | — | — | — | — |
| KNN | — | — | — | — | — |
| Decision Tree | — | — | — | — | — |
| SVM | — | — | — | — | — |

The values should be populated with the actual results obtained after training the models.

## Future Improvements

- Add more diverse student features.
- Implement hyperparameter optimization.
- Handle class imbalance using appropriate techniques.
- Add explainable AI techniques such as SHAP.
- Add personalized improvement recommendations.
- Integrate a database for storing prediction history.
- Deploy the application using Streamlit Cloud or another cloud platform.
- Evaluate the model on data from different institutions to assess generalization.

## Disclaimer

This project is intended for educational and research purposes.

Placement predictions are based on patterns present in the dataset and should not be interpreted as a definitive assessment of a student's employability or guarantee of placement.

## License

This project is available under the MIT License.
