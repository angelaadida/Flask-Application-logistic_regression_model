# 🌸 Iris Flower Classification Web App

**Python | Flask | Scikit-learn | Logistic Regression | Joblib**

A machine learning web application that classifies Iris flowers into 3 species (Setosa, Versicolor, Virginica) based on sepal and petal dimensions, built with Flask and Logistic Regression.

---

## 🚀 How It Works

```
User Input (Sepal & Petal dimensions)
        ↓
Flask Web App (app.py)
        ↓
Load Trained Model (logistic_regression_model.pkl)
        ↓
Logistic Regression Classification
        ↓
Display Predicted Iris Species
```

---

## 📊 Dataset — Iris Dataset

| Feature | Description |
|---------|-------------|
| Sepal Length | Length of sepal (cm) |
| Sepal Width | Width of sepal (cm) |
| Petal Length | Length of petal (cm) |
| Petal Width | Width of petal (cm) |
| **Target** | **Setosa / Versicolor / Virginica** |

- **150 samples** total — 50 per species
- Split: **70% training / 30% testing**

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|-----------|
| Web Framework | Flask |
| ML Model | Scikit-learn Logistic Regression |
| Model Serialization | Joblib |
| Frontend | HTML, CSS |
| Dataset | Iris (sklearn.datasets) |
| Language | Python |

---

## 📁 Project Structure

```
Flask-Application-logistic_regression_model/
│
├── app.py                          # Flask web app & routing
├── model.py                        # Model training & Joblib serialization
├── logistic_regression_model.pkl   # Trained Logistic Regression model
├── index.html                      # Frontend UI
└── style.css                       # CSS styling
```

---

## ⚙️ How to Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/angelaadida/Flask-Application-logistic_regression_model.git
cd Flask-Application-logistic_regression_model

# 2. Install dependencies
pip install flask scikit-learn numpy joblib

# 3. Train the model (optional — .pkl already included)
python model.py

# 4. Run the app
python app.py

# 5. Open browser
http://localhost:5000
```

---

## 🧠 Model Details

```python
# Model training (model.py)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris

model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Accuracy evaluation
accuracy = accuracy_score(y_test, y_pred)

# Save with Joblib
joblib.dump(model, 'logistic_regression_model.pkl')
```

---

## 📜 Key Concepts Demonstrated

- ✅ Multi-class classification with Logistic Regression
- ✅ Train/test split (70/30) with random state
- ✅ Model accuracy evaluation with accuracy_score
- ✅ Joblib model serialization & loading
- ✅ Flask API serving real-time predictions
- ✅ End-to-end ML web deployment

---

## 👩‍💻 About

Built by **Angela Nguyen Hao** — Data Scientist & BI Specialist

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Angela_Nguyen_Hao-0077B5?style=flat&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/angela-n-25945b338)
[![GitHub](https://img.shields.io/badge/GitHub-angelaadida-181717?style=flat&logo=github&logoColor=white)](https://github.com/angelaadida)
