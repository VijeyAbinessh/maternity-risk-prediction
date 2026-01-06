# 🩺 Pregnancy Risk Prediction System (Machine Learning)

## 📌 Project Overview

This project is a **machine learning–based pregnancy risk assessment system** that predicts the **risk percentage and risk level (Low / Moderate / High)** for a patient based on clinical parameters.
It combines **Logistic Regression** with a **rule-based explanation engine** to provide **interpretable medical insights** along with actionable health suggestions.

---

## 🎯 Objectives

* Predict pregnancy risk as a **percentage**
* Classify patients into **Low, Moderate, or High Risk**
* Provide **medical reasons** behind the prediction
* Suggest **preventive actions and recommendations**
* Visualize risk using an **intuitive risk gauge chart**

---

## 🧠 Model Used

* **Algorithm:** Logistic Regression
* **Why Logistic Regression?**

  * Produces true probability scores
  * Easy to interpret (medical domain friendly)
  * Lightweight and fast
  * Suitable for binary risk classification

---

## 📊 Features Used

The model uses the following clinical attributes:

* Age
* Systolic Blood Pressure
* Diastolic Blood Pressure
* Blood Sugar (BS)
* Body Mass Index (BMI)
* Previous Pregnancy Complications
* Preexisting Diabetes
* Gestational Diabetes
* Mental Health Status
* Heart Rate

---

## ⚙️ System Architecture

1. **Data Preprocessing**

   * Invalid value removal
   * Missing value handling using Median Imputation
   * Feature scaling using StandardScaler

2. **Model Training**

   * Train-test split (80–20)
   * Logistic Regression with class balancing

3. **Prediction Layer**

   * Risk probability calculation
   * Risk level categorization

4. **Explainability Layer**

   * Rule-based medical reason engine
   * Personalized health suggestions

5. **Visualization**

   * Risk gauge chart (0–100%)

---

## 📈 Model Performance

* **Accuracy:** ~98%
* **ROC-AUC Score:** ~0.99
  This indicates excellent discrimination between low-risk and high-risk cases.

---

## 📦 Files Generated

After training, the following files are saved:

* `risk_model.pkl` → Trained Logistic Regression model
* `imputer.pkl` → Missing value handler
* `scaler.pkl` → Feature scaler

These files are reused during prediction.

---

## 📊 Risk Interpretation Logic

| Risk Percentage | Risk Level    |
| --------------- | ------------- |
| < 30%           | Low Risk      |
| 30% – 60%       | Moderate Risk |
| > 60%           | High Risk     |

---

## 📌 Risk Visualization

The system includes a **risk gauge chart** that visually represents:

* Low Risk (Green)
* Moderate Risk (Orange)
* High Risk (Red)

This improves usability for **doctors and non-technical users**.

---

## 🧪 Sample Output

```text
Risk Percentage: 15.49%
Risk Level: Low Risk
Reasons: []
Suggestions: []
```

For high-risk patients, the system automatically provides:

* Medical reasons (e.g., high BP, diabetes)
* Preventive health suggestions

---

## 🛠️ Technologies Used

* Python
* Pandas & NumPy
* Scikit-learn
* Matplotlib
* Joblib

---

## 🚀 How to Run

1. Install dependencies:

```bash
pip install pandas numpy scikit-learn matplotlib joblib
```

2. Run the project:

```bash
python main.py
```

3. View:

* Console prediction output
* Risk visualization chart

---

## 🎓 Academic Value

* Demonstrates **applied machine learning**
* Combines **ML + rule-based explainability**
* Suitable for:

  * Final year projects
  * Health analytics demos
  * ML case studies
  * Viva and presentations

---

## 🔮 Future Enhancements

* Web dashboard using Flask or FastAPI
* Feature importance visualization
* Multi-class risk levels
* Integration with hospital systems
* Mobile app interface

---

## 👤 Author

Vijey Abinessh K

---


