# 🎓 Field Recommendation System for Rwanda Polytechnic
### _AI-Powered Student Placement Using Machine Learning & Explainable AI (XAI)_

This repository contains a **complete end-to-end Machine Learning pipeline** for predicting the most suitable academic field for Rwanda Polytechnic (RP) students. The system is built using **Neural Networks, XGBoost, and Random Forest**, combined with **interpretability tools (SHAP & LIME)** and strict **no–data-leakage** preprocessing workflows.

The model analyzes **TVET combinations, examination boards, and subject-level marks** to generate accurate, fair, and explainable field recommendations.

---

## 🚀 Key Features  
### 🧠 Machine Learning Pipeline  
- Neural Network (Keras)  
- XGBoost  
- Random Forest  
- 5-Fold Stratified Cross-Validation  
- Overfitting & generalization analysis  

### 📊 Feature Engineering & Analysis  
- One-Hot Encoding (train-only)  
- Standard Scaling (train-only)  
- Mutual Information  
- Recursive Feature Elimination (RFE)  
- Permutation Importance  
- PCA (Dimensionality Reduction)  
- Feature Perturbation (robustness test)  

### 🔍 Explainability (XAI)  
- **SHAP values** for global feature impacts  
- **LIME** for local instance-based explanations  
- Subject–Field correlation analysis  
- Per-class performance insights  

### 📈 Visualizations Included  
- ROC curves (multi-class)  
- Confusion matrices (raw & normalized)  
- Heatmaps (MI, correlation, perturbation)  
- PCA scree plots  
- CV score boxplots  
- Confidence distribution histogram  
- Feature importance charts  

### 🎯 Deployable Recommendation System  
A reusable class:  
```python
FieldRecommendationSystem
```  
Allows real-time predictions with:  
- Predicted field  
- Confidence level  
- Top-3 recommended fields  
- Support for unseen boards/combos  

---

## 📁 Project Structure  

```
.
├── main.py                     # Full ML pipeline and analysis
├── results/
│   ├── field_recommendation_model.h5       # Saved best model
│   ├── field_recommendation_model.pkl
│   ├── encoders_scalers.pkl
│   ├── recommendation_system.pkl
│   ├── subject_columns.pkl
│   └── *.pkl (all transformers)
└── dataset/
    └── rp_merged_dataset_cleaned.json
```

---

## 📦 Installation  

### 1. Clone the repository  
```bash
[git clone https://github.com/your-username/field-recommendation-system.git](https://github.com/94etienne/ai_driven_field_recomendation_at_rp.git)
cd field-recommendation-system
```

### 2. Install requirements  
```bash
pip install -r requirements.txt
```

**requirements.txt example:**  
```
numpy
pandas
scikit-learn
xgboost
tensorflow
matplotlib
seaborn
joblib
lime
shap
statsmodels
tabulate
```

---

## ▶️ Run the Project

### Train models, generate visualizations, perform analysis:
```bash
python main.py
```

All results, models, and plots will be saved in the **results/** directory automatically.

---

## 🧪 Example Prediction (Real-Time Use)

```python
from joblib import load

system = load("results/recommendation_system.pkl")

prediction = system.predict(
    examination_board="RTB",
    combination="SWD",
    marks_dict={
        "Applied Mathematics B": 85,
        "Front-End Design and Development": 92,
        "DevOps and Software Testing": 88,
        "Back-End Development and Database": 90
    }
)

print(prediction)
```

**Output example:**
```json
{
  "predicted_field": "Software Engineering",
  "confidence": 0.94,
  "top_3_predictions": [
    ["Software Engineering", 0.94],
    ["Networking", 0.03],
    ["Information Systems", 0.02]
  ],
  "method": "xgboost",
  "examination_board": "RTB",
  "combination": "SWD"
}
```

---

## 🎯 Use Cases  
- Rwanda Polytechnic (RP) student field recommendation  
- TVET program placement  
- Automated academic advising  
- Early guidance for new applicants  
- Integration with RP systems (Moodle, MIS, LMS)  
- Government-level education analytics  

---

## 🧭 Architecture Overview  

```
Data → Preprocessing → Train/Test Split
     → Encoding & Scaling → Model Training
     → Cross-Validation → Model Selection
     → Interpretation (SHAP/LIME)
     → Recommendation System Export
     → Deployment (API/UI)
```

---

## 📊 Evaluation Metrics  

The system reports:
- Accuracy  
- Precision  
- Recall  
- F1-score  
- ROC–AUC (multi-class)  
- CV Mean & Std  
- Per-class accuracy  
- Overfitting gap  
- Statistical significance (McNemar test)  

---

## 🧩 Explainability Examples  

### ✔️ LIME:  
Explains specific predictions for individual students.

### ✔️ SHAP:  
Shows global influence of subjects like Mathematics, Physics, Programming, etc.

---

## 🌍 Deployment Options  

### Option A — Flask/FastAPI  
I can generate an API with an endpoint:  
```
POST /predict
```

### Option B — Streamlit Dashboard  
User-friendly interface for students and advisors.

### Option C — React + FastAPI Integration  
Works with your previous RP AI projects.

---

## 📝 License  
**MIT License**  
Free to modify, extend, and use for academic or government applications.

---

## 👨‍💻 Author  
**NTAMBARA Etienne**  
Assistant Lecturer & AI Researcher  
Rwanda Polytechnic – Huye College  
