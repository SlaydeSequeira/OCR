import subprocess
import sys

# Function to install packages
def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

# Install necessary packages
install('pandas')
install('streamlit')
install('scikit-learn')

import time
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Load and preprocess dataset
@st.cache_data
def load_data():
    file_path = "WineQT.csv"  # Update the file path as needed
    wine_data = pd.read_csv(file_path)
    wine_data = wine_data.drop(columns=["Id"])
    wine_data['quality'] = (wine_data['quality'] >= 6).astype(int)
    return wine_data

wine_data = load_data()

X = wine_data.drop(columns=["quality"])
y = wine_data["quality"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sidebar slider for l1_ratio
st.sidebar.title("Elastic Net Configuration")
l1_ratio = st.sidebar.slider("Set L1 Ratio (Elastic Net)", 0.0, 1.0, 0.5, 0.1)

# Feature selection using Lasso, Ridge, and Elastic Net
lasso_selector = SelectFromModel(LogisticRegression(penalty="l1", solver="saga", C=1.0, random_state=42))
lasso_selector.fit(X_train, y_train)
X_train_lasso = lasso_selector.transform(X_train)
X_test_lasso = lasso_selector.transform(X_test)

ridge_selector = SelectFromModel(RidgeClassifier(alpha=1.0, random_state=42))
ridge_selector.fit(X_train, y_train)
X_train_ridge = ridge_selector.transform(X_train)
X_test_ridge = ridge_selector.transform(X_test)

elastic_net_selector = SelectFromModel(
    SGDClassifier(loss="log", penalty="elasticnet", l1_ratio=l1_ratio, random_state=42)
)
elastic_net_selector.fit(X_train, y_train)
X_train_elastic_net = elastic_net_selector.transform(X_train)
X_test_elastic_net = elastic_net_selector.transform(X_test)

# Models
log_reg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# Function to compute accuracy and time
def compute_accuracy_and_time(model, X_train, X_test, y_train, y_test):
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time

    start_time = time.time()
    y_pred = model.predict(X_test)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test, y_pred)
    return accuracy, train_time, predict_time

# Compute metrics for Logistic Regression
logistic_lasso = compute_accuracy_and_time(log_reg, X_train_lasso, X_test_lasso, y_train, y_test)
logistic_ridge = compute_accuracy_and_time(log_reg, X_train_ridge, X_test_ridge, y_train, y_test)
logistic_elastic_net = compute_accuracy_and_time(log_reg, X_train_elastic_net, X_test_elastic_net, y_train, y_test)

# Compute metrics for Random Forest
rf_lasso = compute_accuracy_and_time(rf, X_train_lasso, X_test_lasso, y_train, y_test)
rf_ridge = compute_accuracy_and_time(rf, X_train_ridge, X_test_ridge, y_train, y_test)
rf_elastic_net = compute_accuracy_and_time(rf, X_train_elastic_net, X_test_elastic_net, y_train, y_test)

# Display results in a table
results = {
    "Model": ["Logistic Regression", "Logistic Regression", "Logistic Regression",
              "Random Forest", "Random Forest", "Random Forest"],
    "Feature Selection": ["Lasso", "Ridge", "Lasso + Ridge",
                           "Lasso", "Ridge", "Lasso + Ridge"],
    "Accuracy": [logistic_lasso[0], logistic_ridge[0], logistic_elastic_net[0],
                 rf_lasso[0], rf_ridge[0], rf_elastic_net[0]],
    "Train Time (s)": [logistic_lasso[1], logistic_ridge[1], logistic_elastic_net[1],
                       rf_lasso[1], rf_ridge[1], rf_elastic_net[1]],
    "Predict Time (s)": [logistic_lasso[2], logistic_ridge[2], logistic_elastic_net[2],
                         rf_lasso[2], rf_ridge[2], rf_elastic_net[2]]
}

results_df = pd.DataFrame(results)

st.title("Model Performance Metrics")
st.dataframe(results_df)
