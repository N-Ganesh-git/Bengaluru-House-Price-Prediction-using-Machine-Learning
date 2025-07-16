# 🏠 Bengaluru House Price Prediction using XGBoost and Streamlit!


This project is a web-based machine learning application that predicts the price of a house in Bengaluru based on various input features. It uses a tuned XGBoost model for regression and is deployed using Streamlit.

## 🚀 Demo

<img width="1470" height="956" alt="Screenshot 2025-07-17 at 4 49 35 AM" src="https://github.com/user-attachments/assets/30b2c6d6-8d3c-44aa-8df5-a6fde200d436" />


## 📌 Features

- Interactive Streamlit web interface
- User inputs: Area Type, Location, Size, Square Feet, Bathrooms, Balconies
- XGBoost Regression with GridSearchCV for hyperparameter tuning
- Feature scaling and log transformation applied for better accuracy
- Model evaluation metrics: RMSE, MAE, R², Cross-validated R²
- Real-time price prediction in Indian Lakhs

---

## 📂 Dataset

Dataset: **Bengaluru House Price Data**  
Source: [Kaggle - Bengaluru House Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data)

### Important Features Used:

- `location`
- `area_type`
- `total_sqft`
- `size`
- `bath`
- `balcony`
- `price`

---

## 🧠 Machine Learning Model

- **Algorithm**: XGBoost Regressor
- **Hyperparameter Tuning**: GridSearchCV
- **Target Variable**: Log-transformed house price
- **Feature Scaling**: StandardScaler
- **Validation**: 5-fold Cross-validation

---

## 📊 Model Evaluation Metrics

- **RMSE (Root Mean Squared Error)**
- **MAE (Mean Absolute Error)**
- **R² Score**
- **Cross-validated R² Score**

---

## 🛠️ Tech Stack

- Python
- Streamlit
- Pandas
- NumPy
- Scikit-learn
- XGBoost

---

## 🔮 Future Improvements
- Save trained model using joblib or pickle for faster load
- Add EDA & visualizations for location-wise trends
- Use Streamlit Cloud or HuggingFace Spaces for deployment
- Implement filters and sorting for better UX
