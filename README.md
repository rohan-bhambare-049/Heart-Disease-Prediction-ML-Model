# Heart Disease Prediction ML Model

Author: **Rohan Bhambare**  
Repository: [Heart-Disease-Prediction-ML-Model](https://github.com/rohan-bhambare-049/Heart-Disease-Prediction-ML-Model)

This repository contains a machine-learning project for predicting the presence of heart disease from clinical measurements. The main analysis is implemented in [`ML P1/heart.ipynb`](ML%20P1/heart.ipynb), and the trained model is exposed through a small Streamlit application in [`Full Project/app.py`](Full%20Project/app.py).

## Project contents

Only the two project folders are included in this repository:

```text
.
├── ML P1/
│   ├── heart.ipynb          # Main heart-disease analysis and modeling notebook
│   ├── heart.csv            # Heart-disease dataset used by the notebook
│   ├── insurance.ipynb      # Additional notebook from the ML P1 work
│   ├── insurance.csv        # Dataset for the additional notebook
│   ├── Logistic_Heart.pkl   # Serialized logistic-regression model
│   ├── scaler.pkl           # Serialized feature scaler
│   └── columns.pkl          # Model feature-column order
└── Full Project/
    ├── app.py               # Streamlit prediction interface
    ├── Logistic_Heart.pkl   # Model consumed by the application
    ├── scaler.pkl           # Scaler consumed by the application
    └── columns.pkl          # Feature order consumed by the application
```

The local `myenv` virtual environment is intentionally excluded from version control. It should be recreated locally from the required packages instead of uploaded to GitHub.

## Main notebook: `ML P1/heart.ipynb`

The notebook documents the full heart-disease prediction workflow:

1. **Load and inspect the data** — reads `heart.csv` with pandas and reviews columns, descriptive statistics, duplicated rows, class distribution, and missing values.
2. **Explore the data** — uses Seaborn visualizations for numeric distributions, category counts, cholesterol by target, and numeric correlations.
3. **Clean invalid values** — treats zero cholesterol and zero resting blood pressure as invalid measurements. Each is replaced with the mean of its respective non-zero observations.
4. **Encode features** — applies one-hot encoding with `drop_first=True` to categorical variables and converts the resulting fields to integer values.
5. **Scale numeric features** — standardizes `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, and `Oldpeak` with `StandardScaler`.
6. **Split the data** — separates `HeartDisease` as the target and uses an 80/20 train-test split with `random_state=42`.
7. **Compare classifiers** — evaluates Logistic Regression, K-Nearest Neighbors, Gaussian Naive Bayes, Decision Tree, and Support Vector Machine models using accuracy and F1 score.
8. **Export deployment artifacts** — saves the logistic-regression model, scaler, and ordered feature-column list with `joblib`.

### Dataset fields

The dataset includes age, sex, chest-pain type, resting blood pressure, cholesterol, fasting blood sugar, resting ECG, maximum heart rate, exercise-induced angina, oldpeak, ST slope, and the binary `HeartDisease` target.

The target distribution recorded in the notebook is:

| Target | Meaning | Records |
|---:|---|---:|
| 0 | No heart disease | 410 |
| 1 | Heart disease present | 508 |

### Recorded model comparison

The following values are the results stored in the notebook after evaluation on its test split:

| Model | Accuracy | F1 score |
|---|---:|---:|
| Logistic Regression | 0.8696 | 0.8857 |
| K-Nearest Neighbors | 0.8641 | 0.8815 |
| Gaussian Naive Bayes | 0.8533 | 0.8683 |
| Decision Tree | 0.7772 | 0.7980 |
| Support Vector Machine | 0.8478 | 0.8679 |

Logistic Regression is the selected model for the included application because it has the highest recorded accuracy and F1 score among the compared models.

## Run the Streamlit application

From the repository root, install the required packages and start the app:

```bash
pip install pandas scikit-learn streamlit joblib
streamlit run "Full Project/app.py"
```

The application loads the serialized model and preprocessing artifacts from the `Full Project` directory, accepts the clinical inputs through a form, applies the saved feature order and scaler, and displays a low-risk or high-risk prediction.

## Re-run the notebook

Open `ML P1/heart.ipynb` in Jupyter Notebook, JupyterLab, or VS Code with a Python environment containing pandas, NumPy, Matplotlib, Seaborn, scikit-learn, and joblib. Run the cells from top to bottom so that the cleaned data, fitted scaler, model comparisons, and serialized artifacts are produced in the expected order.

## Important note

This project is for educational and demonstration purposes only. Its output is a machine-learning prediction and is not medical advice, a diagnosis, or a substitute for evaluation by a qualified healthcare professional.
