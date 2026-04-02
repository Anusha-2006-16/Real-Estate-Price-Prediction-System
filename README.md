# 🏡 Real Estate Price Prediction System

An end-to-end Machine Learning project that predicts house prices based on features like location, area, and number of bedrooms. Built with an interactive web interface using Streamlit.

---

## 🚀 Project Overview

Predicting real estate prices is a complex problem influenced by multiple factors. This project uses Machine Learning models to estimate property prices and helps users make informed decisions.

---

## 🔍 Features

- 📊 Data preprocessing and cleaning
- 🏙️ Location handling using One-Hot Encoding
- 📉 Feature engineering (price per sqft)
- ⚡ Handling rare locations to reduce noise
- 📈 Visualization:
  - Price distribution
  - Area vs Price
  - Average price per location
  - Feature importance
- 🤖 Machine Learning Models:
  - Linear Regression
  - Random Forest Regressor
- 🎯 Model evaluation using R² Score and MAE
- 🌐 Interactive UI using Streamlit

---

## 🧠 Key Learnings

- Importance of **feature engineering**
- Handling **categorical variables effectively**
- Avoiding **data leakage**
- Improving model performance using **log transformation**
- Comparing baseline vs advanced models

---

## 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Streamlit

---

## 📂 Dataset

- Dataset: Bangalore Housing Dataset
- Contains features like:
  - Price
  - Area
  - Location
  - BHK
  - Amenities

---

## ⚙️ Installation & Setup

```bash
# Clone the repository
git clone https://github.com/your-username/real-estate-price-prediction.git

# Navigate to project folder
cd real-estate-price-prediction

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py

