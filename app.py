import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_absolute_error

data=r"Bangalore.csv"

st.title("Real Estate Price Prediction System")

try:
    df=pd.read_csv(data)
    df.rename(columns = {"No. of Bedrooms":"BHK"},inplace = True)
    df.columns=df.columns.str.lower()
    df.columns=df.columns.str.replace(" ","_")

    df = df.drop_duplicates()

    df['original_price']=df['price']
    df['pricepersqft']=df['price']/df['area']

    # 🔥 REMOVE RARE LOCATIONS
    df['loc']=df['location']
    location_counts = df['loc'].value_counts()
    rare_locations = location_counts[location_counts < 10].index
    df['loc'] = df['loc'].replace(rare_locations, 'Other')

    # Scaling
    scaler = MinMaxScaler()
    df[['area']] = scaler.fit_transform(df[['area']])

    # One-hot encoding
    df = pd.get_dummies(df, columns=['loc'], drop_first=True)

    st.write(df.head(2))

    # ---------------- VISUALIZATION ---------------- #

    st.subheader("Price Distribution")
    fig,ax=plt.subplots()
    ax.hist(df['original_price'],bins=30)
    st.pyplot(fig)

    st.subheader("Area vs Price Relationship")
    fig,ax=plt.subplots()
    ax.scatter(df['area'],df['original_price'])
    st.pyplot(fig)

    # ---------------- FEATURE IMPORTANCE ---------------- #

    importance = df.corr(numeric_only=True)['original_price'].sort_values(ascending=False)

    st.subheader("Feature Importance")
    st.bar_chart(importance.head(15))

    # Select features
    important_features = importance[importance > 0.05].index.tolist()

    for col in ['price','pricepersqft','original_price']:
        if col in important_features:
            important_features.remove(col)

    # ---------------- MACHINE LEARNING ---------------- #

    X = df[important_features]
    y = np.log(df['original_price'])   # 🔥 LOG TRANSFORM

    X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.1,random_state=42)

    # Linear Regression
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)

    # Random Forest 🔥 IMPROVED
    rf_model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=5,
        random_state=42
    )
    rf_model.fit(X_train,y_train)
    y_pred_rf = rf_model.predict(X_test)

    # ---------------- SCORES ---------------- #

    st.subheader("Model Performance")

    col1,col2 = st.columns(2)

    with col1:
        st.write("### Linear Regression")
        st.write("R2:", r2_score(y_test,y_pred))
        st.write("MAE:", mean_absolute_error(np.exp(y_test), np.exp(y_pred)))

    with col2:
        st.write("### Random Forest")
        st.write("R2:", r2_score(y_test,y_pred_rf))
        st.write("MAE:", mean_absolute_error(np.exp(y_test), np.exp(y_pred_rf)))

    # ---------------- PREDICTION ---------------- #

    st.subheader("Enter your data to get the price Prediction")

    area = st.number_input("Enter area (sqft)", min_value=100.0, max_value=10000.0)
    bhk = st.number_input("Enter BHK", min_value=1, max_value=10)

    location_cols = [col for col in X.columns if col.startswith("loc_")]

    location = st.selectbox("Select Location", location_cols)

    input_data = pd.DataFrame(columns=X.columns)
    input_data.loc[0] = 0

    # scale area
    input_data['area'] = scaler.transform([[area]])[0][0]
    input_data['bhk'] = bhk

    if location in input_data.columns:
        input_data[location] = 1

    # Predict using Random Forest
    pred_log = rf_model.predict(input_data)[0]
    predict = np.exp(pred_log)   # 🔥 reverse log

    st.write(f"💰 Estimated Price: ₹ {int(predict):,}")

except Exception as e:
    st.error(f"Error Occurred: {e}")