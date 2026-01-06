import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score

import matplotlib.pyplot as plt
import numpy as np


def plot_risk_gauge(risk_percent):
    fig, ax = plt.subplots(figsize=(7, 4))

    # Gauge background
    ax.barh(0, 100, left=0, color="#e5e7eb")
    ax.barh(0, risk_percent, left=0, color="#ef4444" if risk_percent >= 60 else
            "#f59e0b" if risk_percent >= 30 else "#22c55e")

    # Labels
    ax.text(risk_percent + 2, 0,
            f"{risk_percent}%",
            va='center', fontsize=14, fontweight='bold')

    # Risk zones
    ax.text(15, -0.25, "Low", color="#22c55e", fontsize=11, fontweight="bold")
    ax.text(45, -0.25, "Moderate", color="#f59e0b", fontsize=11, fontweight="bold")
    ax.text(75, -0.25, "High", color="#ef4444", fontsize=11, fontweight="bold")

    # Styling
    ax.set_xlim(0, 100)
    ax.set_yticks([])
    ax.set_xlabel("Risk Percentage")
    ax.set_title("Pregnancy Risk Assessment", fontsize=14, fontweight="bold")

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.show()


# ======================================================
# 1. LOAD DATA
# ======================================================
DATA_PATH = r"C:\Users\vijey abinessh\Desktop\med\Dataset - Updated.csv"
df = pd.read_csv(DATA_PATH)


# ======================================================
# 2. BASIC CLEANING
# ======================================================
df = df[df["Age"] < 100]
df = df[df["BMI"] > 10]

df["Risk Level"] = df["Risk Level"].map({"Low": 0, "High": 1})
df = df.dropna(subset=["Risk Level"])


# ======================================================
# 3. FEATURE SELECTION
# ======================================================
FEATURES = [
    "Age",
    "Systolic BP",
    "Diastolic",
    "BS",
    "BMI",
    "Previous Complications",
    "Preexisting Diabetes",
    "Gestational Diabetes",
    "Mental Health",
    "Heart Rate"
]

X = df[FEATURES]
y = df["Risk Level"]


# ======================================================
# 4. HANDLE MISSING VALUES
# ======================================================
imputer = SimpleImputer(strategy="median")
X = pd.DataFrame(imputer.fit_transform(X), columns=FEATURES)


# ======================================================
# 5. FEATURE SCALING (IMPORTANT FOR LOGISTIC)
# ======================================================
scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=FEATURES)


# ======================================================
# 6. TRAIN TEST SPLIT
# ======================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)


# ======================================================
# 7. TRAIN LOGISTIC REGRESSION MODEL
# ======================================================
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)


# ======================================================
# 8. EVALUATION
# ======================================================
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print("\n=== MODEL PERFORMANCE ===")
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_prob))


# ======================================================
# 9. SAVE MODEL
# ======================================================
joblib.dump(model, "risk_model.pkl")
joblib.dump(imputer, "imputer.pkl")
joblib.dump(scaler, "scaler.pkl")

print("\n✅ Model, imputer, and scaler saved successfully")


# ======================================================
# 10. REASON ENGINE
# ======================================================
def generate_reasons(row):
    reasons = []

    if row["BS"] >= 7.8:
        reasons.append("Elevated blood sugar level")

    if row["Systolic BP"] >= 140 or row["Diastolic"] >= 90:
        reasons.append("High blood pressure")

    if row["BMI"] >= 25:
        reasons.append("BMI indicates overweight or obesity")

    if row["Preexisting Diabetes"] == 1:
        reasons.append("History of diabetes")

    if row["Previous Complications"] == 1:
        reasons.append("Previous pregnancy complications")

    if row["Mental Health"] == 1:
        reasons.append("Mental health concerns")

    if row["Heart Rate"] >= 85:
        reasons.append("Abnormal heart rate")

    return reasons


# ======================================================
# 11. SUGGESTION ENGINE
# ======================================================
def generate_suggestions(reasons):
    suggestions = []
    text = " ".join(reasons).lower()

    if "blood sugar" in text or "diabetes" in text:
        suggestions.append("Monitor blood sugar and follow diabetic diet")

    if "blood pressure" in text:
        suggestions.append("Reduce salt intake and monitor BP daily")

    if "bmi" in text or "obesity" in text:
        suggestions.append("Maintain balanced diet and light exercise")

    if "complications" in text:
        suggestions.append("Regular antenatal checkups recommended")

    if "mental health" in text:
        suggestions.append("Seek counseling or mental health support")

    if "heart rate" in text:
        suggestions.append("Consult physician for heart evaluation")

    return suggestions


# ======================================================
# 12. FINAL PREDICTION FUNCTION
# ======================================================
def predict_risk(input_dict):
    model = joblib.load("risk_model.pkl")
    imputer = joblib.load("imputer.pkl")
    scaler = joblib.load("scaler.pkl")

    # ORIGINAL (for reasons)
    original_df = pd.DataFrame([input_dict])
    original_df = pd.DataFrame(imputer.transform(original_df), columns=FEATURES)

    # SCALED (for model)
    scaled_df = pd.DataFrame(
        scaler.transform(original_df),
        columns=FEATURES
    )

    probability = model.predict_proba(scaled_df)[0][1]
    risk_percent = round(probability * 100, 2)

    if risk_percent < 30:
        risk_level = "Low Risk"
    elif risk_percent < 60:
        risk_level = "Moderate Risk"
    else:
        risk_level = "High Risk"

    reasons = generate_reasons(original_df.iloc[0])  # ✅ FIXED
    suggestions = generate_suggestions(reasons)

    return {
        "Risk Percentage": f"{risk_percent}%",
        "Risk Level": risk_level,
        "Reasons": reasons,
        "Suggestions": suggestions
    }


# ======================================================
# 13. TEST WITH SAMPLE INPUT
# ======================================================
if __name__ == "__main__":
    sample_patient = {
    "Age": 36,
    "Systolic BP": 152,
    "Diastolic": 96,
    "BS": 9.2,
    "BMI": 31,
    "Previous Complications": 1,
    "Preexisting Diabetes": 1,
    "Gestational Diabetes": 0,
    "Mental Health": 1,
    "Heart Rate": 92
}


    result = predict_risk(sample_patient)

    print("\n=== RISK PREDICTION RESULT ===")
    for k, v in result.items():
        print(f"{k}: {v}")

    # 🔥 VISUALIZE RISK
    risk_value = float(result["Risk Percentage"].replace("%", ""))
    plot_risk_gauge(risk_value)
