# ML-Based Diabetes Prediction

An ML-based web application that predicts diabetes risk using health and lifestyle information. It also includes a separate hospital-associated UTI prediction option for female users.

## Project Features

- Gender-based user flow.
- Diabetes risk prediction using a calibrated machine learning model.
- Hospital-associated UTI prediction for female users.
- Option to view diabetes and UTI predictions together.
- SHAP-based explanations of predictions.
- Risk-related suggestions and guidance.
- Model performance and calibration information.

## Technologies Used

- Python
- Streamlit
- Pandas and NumPy
- Scikit-learn
- SHAP
- Matplotlib
- Joblib

## Datasets

### Diabetes Dataset

The diabetes model uses the CDC Diabetes Health Indicators dataset, available through the UCI Machine Learning Repository (Dataset ID: 891).

The dataset contains health and lifestyle indicators used to predict a survey-defined diabetes outcome.

### UTI Dataset

The UTI model uses the Danish hospital admission dataset available through Dryad:

https://doi.org/10.5061/dryad.6djh9w107

This model predicts hospital-associated UTI during admission. It is not designed to diagnose community-acquired UTI.

## Project Structure

```text
ML-Based-Diabetes-Prediction/
│
├── app/
│   └── streamlit_app.py
│
├── data/
├── models/
│   ├── diabetes_model.pkl
│   └── uti_model.pkl
│
├── reports/
├── src/
├── tests/
│
├── diabetes_dataset.csv
├── diabetes_cleaned.csv
├── uti_female.csv
├── prepare_diabetes.py
├── prepare_uti.py
├── train_diabetes.py
├── train_uti.py
├── requirements.txt
└── README.md
```

## Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/manasaa-04/ML-Based-Diabetes-Prediction.git
cd ML-Based-Diabetes-Prediction
```

### 2. Create a virtual environment

Windows:

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app/streamlit_app.py
```

The application will open in your browser.

## Model Evaluation

The models are evaluated using held-out test data and metrics such as accuracy, precision, recall, F1-score, and ROC-AUC.

The diabetes model also uses probability calibration. Calibration improves the reliability of predicted probabilities but does not necessarily improve detection performance at every classification threshold.

The UTI model has relatively low precision, so its predictions may include false positives. Its results should be interpreted cautiously.

## Important Disclaimer

This project is developed for educational and academic purposes. It is not a substitute for professional medical advice, diagnosis, or treatment. Predictions are not confirmed medical diagnoses.

## Author

Developed as a final-year academic project.
