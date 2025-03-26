# **🔬 Predicting % Silica Concentrate in Iron Ore Processing**

![Machine Learning](https://img.shields.io/badge/Machine%20Learning-Regression-blue.svg)
![Python](https://img.shields.io/badge/Python-3.8%2B-brightgreen.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-orange.svg)

## **📌 Project Overview**
This project applies **machine learning models** to predict the **% Silica Concentrate** in iron ore processing. High silica levels affect iron ore quality and energy efficiency, so optimizing silica content prediction is essential for industrial applications.

The project evaluates **Lasso, Ridge, Random Forest, and XGBoost**, using **feature engineering, hyperparameter tuning, and model evaluation** to find the best predictive model.

---

## **📂 Dataset Information**
- **Source**: [Iron Ore Quality Dataset on Kaggle](https://www.kaggle.com/code/kthxbao/eda-iron-ore-quality).
- **Features**: 24 parameters related to chemical composition, operational conditions, and environmental factors.
- **Target Variable**: `% Silica Concentrate` (Lower values preferred for high-quality iron ore).

**Key Features:**
- `% Iron Feed`, `% Silica Feed`, `Ore Pulp pH`, `Starch Flow`, `Amina Flow`, `Flotation Column Air Flow`.
- Feature **% Iron Concentrate_power2** was created to capture **non-linear effects**.

---

## **📦 Installation & Requirements**
### **🔧 Dependencies**
To run this project, install the following dependencies:
```bash
pip install numpy pandas scikit-learn matplotlib seaborn xgboost optuna
```

---

## **🚀 How to Run the Project**
1️⃣ **Clone the Repository**
```bash
git clone https://github.com/ishansilwal1/silica-prediction.git
cd silica-prediction
```

2️⃣ **Run the Model**
```bash
python silica_prediction.py
```

---

## **📊 Models Used & Justification**

| Model | Regularization | Best For | Limitation |
|--------------|--------------|----------------|----------------|
| **Lasso Regression** | L1 | Feature selection | Removes important features if λ is too high |
| **Ridge Regression** | L2 | Handling multicollinearity | Does not perform feature selection |
| **Random Forest** | None | Non-linear relationships | Computationally expensive |
| **XGBoost (Final Model)** | L1 & L2 | Best overall performance | Requires tuning |

### **✅ Why XGBoost was the Best Model?**
✔ **Corrects mistakes from previous trees (Boosting approach).**  
✔ **Uses L1 & L2 regularization to prevent overfitting.**  
✔ **Handles missing values automatically.**  
✔ **Computationally efficient & scalable.**  

---

## **🔬 Model Training & Hyperparameter Tuning**
- **Data Split**: 80% Train, 20% Test.
- **Feature Scaling**: StandardScaler for better optimization.
- **Hyperparameter Tuning**: GridSearchCV on `max_depth`, `learning_rate`, `n_estimators`.

```python
param_grid = {
    'n_estimators': [50, 100, 150],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}
grid_search = GridSearchCV(XGBRegressor(), param_grid, cv=5, scoring='neg_root_mean_squared_error')
grid_search.fit(X_train, y_train)
```

---

## **📈 Results & Evaluation**
| Model | Train RMSE | Test RMSE | Train R² | Test R² |
|--------|------------|-----------|----------|---------|
| Lasso Regression | 1.12 | 1.12 | 0.000 | 0.000 |
| Ridge Regression | 0.63 | 0.63 | 0.682 | 0.682 |
| Random Forest | 0.02 | 0.05 | 0.999 | 0.997 |
| **XGBoost (Final Model)** | **0.24** | **0.24** | **0.953** | **0.951** |

✅ **XGBoost achieved the best balance between accuracy and generalization.**

---

## **📤 Deployment**
You can save and deploy the model using Flask or FastAPI.

```python
import pickle
pickle.dump(xgb_model, open("silica_model.pkl", "wb"))
```

For API deployment, use Flask:
```python
from flask import Flask, request, jsonify
import pickle
app = Flask(__name__)
model = pickle.load(open("silica_model.pkl", "rb"))
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    prediction = model.predict([data['features']])
    return jsonify({'prediction': prediction.tolist()})
if __name__ == '__main__':
    app.run()
```

---

## **👨‍💻 Contributing**
Contributions are welcome! Feel free to **fork this repository**, make enhancements, and submit a pull request.

---

## **📜 License**
This project is licensed under the **MIT License**.

---

## **📞 Contact**
For any queries or suggestions, contact:
- **Name**: Ishan Silwal
- **GitHub**: [ishansilwal1](https://github.com/ishansilwal1/silica-prediction)
- **Email**: ishansilwal3@gmail.com

---

### **🌟 If you found this project helpful, please ⭐ this repository!** 🚀

