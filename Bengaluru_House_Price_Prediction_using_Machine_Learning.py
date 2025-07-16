import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


st.set_page_config(page_title="Bengaluru House Price Prediction", layout="wide")
st.title("🏠 Bengaluru House Price Prediction")
st.write("Enter house details to predict the price using a tuned XGBoost model.")

@st.cache_data
def load_data():
    return pd.read_csv('/Users/ganeshn/Documents/Bengaluru_House_Data.csv')

def preprocess_data(df):
    df = df.dropna(subset=['price', 'total_sqft', 'bath', 'balcony', 'size'])

    def convert_sqft(sqft):
        try:
            if '-' in str(sqft):
                low, high = map(float, sqft.split('-'))
                return (low + high) / 2
            return float(sqft)
        except:
            return np.nan

    df['total_sqft'] = df['total_sqft'].apply(convert_sqft)
    df = df.dropna(subset=['total_sqft'])
    df['bedrooms'] = df['size'].str.extract('(\d+)').astype(float)
    df = df[df['bedrooms'] > 0]

    Q1 = df['price'].quantile(0.25)
    Q3 = df['price'].quantile(0.75)
    IQR = Q3 - Q1
    df = df[(df['price'] >= Q1 - 1.5 * IQR) & (df['price'] <= Q3 + 1.5 * IQR)]

    df['price_per_sqft'] = df['price'] * 100000 / df['total_sqft']

    location_price = df.groupby('location')['price'].mean().to_dict()
    df['location_encoded'] = df['location'].map(location_price)

    area_type_map = {key: idx for idx, key in enumerate(df['area_type'].unique())}
    df['area_type_encoded'] = df['area_type'].map(area_type_map)

    features = ['total_sqft', 'bath', 'balcony', 'bedrooms', 'price_per_sqft',
                'location_encoded', 'area_type_encoded']
    X = df[features]
    y = np.log1p(df['price'])

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, area_type_map, location_price, features, df

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 300],
        'learning_rate': [0.05, 0.1],
        'max_depth': [4, 5, 6]
    }

    grid = GridSearchCV(XGBRegressor(random_state=42), param_grid, cv=3, scoring='r2', verbose=1)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_

    y_pred = best_model.predict(X_test)
    y_pred_exp = np.expm1(y_pred)
    y_test_exp = np.expm1(y_test)

    rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
    mae = mean_absolute_error(y_test_exp, y_pred_exp)
    r2 = r2_score(y_test_exp, y_pred_exp)
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='r2')
    avg_cv_r2 = np.mean(cv_scores)

    return best_model, rmse, mae, r2, avg_cv_r2, X_test, y_test, y_pred

df = load_data()
X_scaled, y, scaler, area_type_map, location_price, features, processed_df = preprocess_data(df)
model, rmse, mae, r2, avg_cv_r2, X_test, y_test, y_pred = train_model(X_scaled, y)

st.header("📈 Model Performance")
col1, col2, col3, col4 = st.columns(4)
col1.metric("RMSE", f"{rmse:.2f}")
col2.metric("MAE", f"{mae:.2f}")
col3.metric("R² Score", f"{r2:.2f}")
col4.metric("Cross-Validated R²", f"{avg_cv_r2:.2f}")

st.header("🔍 Predict House Price")
with st.form("predict_form"):
    col1, col2 = st.columns(2)
    with col1:
        area_type = st.selectbox("Area Type", df['area_type'].unique())
        location = st.selectbox("Location", df['location'].unique())
        size = st.selectbox("Size", df['size'].unique())
    with col2:
        sqft = st.number_input("Total Square Feet", min_value=100.0, max_value=10000.0, value=1000.0)
        bath = st.number_input("Number of Bathrooms", 1, 10, 2)
        balcony = st.number_input("Number of Balconies", 0, 5, 1)

    submitted = st.form_submit_button("Predict Price")

if submitted:
    try:
        bedrooms = float(size.split()[0])
        pps = location_price.get(location, df['price'].mean()) * 100000 / sqft
        area_type_encoded = area_type_map.get(area_type, 0)
        location_encoded = location_price.get(location, df['price'].mean())

        input_df = pd.DataFrame([{
            'total_sqft': sqft,
            'bath': bath,
            'balcony': balcony,
            'bedrooms': bedrooms,
            'price_per_sqft': pps,
            'location_encoded': location_encoded,
            'area_type_encoded': area_type_encoded
        }])

        input_scaled = scaler.transform(input_df)
        prediction_log = model.predict(input_scaled)[0]
        prediction = np.expm1(prediction_log)

        st.success(f"💰 Predicted House Price: ₹{prediction:.2f} Lakhs")
    except Exception as e:
        st.error(f"❌ Error: {str(e)}")
