# Heart Disease Prediction Using Random Forest

A machine-learning project that uses clinical attributes to estimate the likelihood of heart disease and presents the result through a Flask web application.

> **Disclaimer:** This project is for educational and research purposes only. It is not a medical device and must not be used to diagnose, treat, or rule out heart disease. Predictions should never replace assessment by a qualified healthcare professional.

## Project Overview

The application applies a Random Forest classifier to structured clinical data. The workflow covers dataset preparation, exploratory data analysis, feature engineering, model training and evaluation, and a web interface for prediction.

### Key Features

- Random Forest classification for heart-disease risk prediction
- Data cleaning, exploratory data analysis, and feature engineering
- Model evaluation using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix
- Flask-based web interface for clinical-parameter entry
- Probability-based result display and PDF report generation

## Technology Stack

- **Language:** Python
- **Machine learning:** scikit-learn
- **Data handling:** pandas, NumPy
- **Web application:** Flask
- **Model persistence:** joblib
- **Reporting:** ReportLab
- **Development:** Jupyter Notebook / VS Code

## Repository Structure

The repository contains model-development scripts, dataset analysis, a Flask application, templates/static assets, and project documentation. Some local runtime assets (such as the serialized model) may need to be generated or placed in the expected location before running the app.

## Getting Started

### 1. Clone the repository

```bash
git clone https://github.com/Sachijhon/AI-Based-Heart-Disease-Prediction-System-.git
cd AI-Based-Heart-Disease-Prediction-System-
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows:

```powershell
.venv\Scripts\Activate.ps1
```

Or on macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

Install the libraries used by the project:

```bash
pip install flask pandas numpy scikit-learn joblib reportlab matplotlib
```

### 4. Prepare the model and templates

Ensure the trained model file expected by the Flask application is present at the configured path and contains the expected model and feature-name objects. Ensure the required HTML templates and static assets are in their expected folders.

### 5. Run the application

```bash
python app.py
```

Open the local address printed by Flask in your browser. The application may require local configuration before it runs successfully.

## Model Evaluation

Use the evaluation scripts and notebooks in this repository to inspect the model's measured performance. Report metrics from the actual evaluation run rather than assuming a fixed accuracy. Evaluation results depend on the dataset, train/test split, and configuration.

## Data

The project includes a Cleveland heart-disease dataset file for academic analysis. Confirm the dataset's license, permitted use, and attribution requirements before redistribution or use beyond coursework.

## Security Notes

- Do not commit passwords, API keys, session secrets, or personal/sensitive data.
- Keep real credentials in environment variables or an untracked local configuration file.
- The development server and debug mode are not suitable for production deployment.
- Configure authentication, session security, input validation, and deployment settings before exposing the application publicly.

## Limitations

- The model was developed for an academic project and may not generalize to different populations or clinical settings.
- Predicted probabilities are model outputs, not calibrated clinical risk estimates unless calibration has been evaluated.
- The system has not been established as clinically validated.

## Authors

- Sachin B A
- Rohith S

Academic project — Bachelor of Computer Applications, 2025–2026.

## License

No license has been specified yet. Unless a license is added, reuse and redistribution permissions are not explicitly granted.