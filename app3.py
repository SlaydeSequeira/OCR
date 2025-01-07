import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
import umap

# Function to load data using st.cache_data
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Sidebar: File Upload
st.sidebar.title("Upload Dataset")
uploaded_file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    # Load dataset
    wine_data = load_data(uploaded_file)
    st.write("##### Uniform Manifold Approximation and Projection")
    # Display dataset info
    st.write("### Dataset Head")
    st.dataframe(wine_data.head())

    # Select Features and Target
    features = wine_data.drop(columns=['quality', 'Id'])
    target = wine_data['quality']
    feature_names = features.columns

    # Standardize Features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Apply UMAP for Dimensionality Reduction
    reducer = umap.UMAP(n_components=2, random_state=42)
    reduced_features = reducer.fit_transform(scaled_features)

    # Split Data
    X_train_orig, X_test_orig, y_train, y_test = train_test_split(scaled_features, target, test_size=0.3, random_state=42)
    X_train_reduced, X_test_reduced, y_train_reduced, y_test_reduced = train_test_split(reduced_features, target, test_size=0.3, random_state=42)

    # Slider for Alpha Value
    alpha_value = st.sidebar.slider("Choose Alpha Value (Transparency)", 0.1, 1.0, 0.5)

    # Logistic Regression on Original Features
    start_time_orig = time.time()
    model_orig = LogisticRegression(max_iter=1000, random_state=42)
    model_orig.fit(X_train_orig, y_train)
    y_pred_orig = model_orig.predict(X_test_orig)
    accuracy_orig = accuracy_score(y_test, y_pred_orig)
    time_taken_orig = time.time() - start_time_orig

    # Logistic Regression on Reduced Features
    start_time_reduced = time.time()
    model_reduced = LogisticRegression(max_iter=1000, random_state=42)
    model_reduced.fit(X_train_reduced, y_train_reduced)
    y_pred_reduced = model_reduced.predict(X_test_reduced)
    accuracy_reduced = accuracy_score(y_test_reduced, y_pred_reduced)
    time_taken_reduced = time.time() - start_time_reduced

    # Display Results
    st.write("### Logistic Regression Results")
    st.write(f"Accuracy (Original Features): {accuracy_orig:.2f}")
    st.write(f"Time Taken (Original Features): {time_taken_orig:.2f} seconds")
    st.write(f"Accuracy (Reduced Features): {accuracy_reduced:.2f}")
    st.write(f"Time Taken (Reduced Features): {time_taken_reduced:.2f} seconds")

    # Plot Before and After Dimensionality Reduction
    st.write("### Before Dimensionality Reduction (Using First Two Features)")
    fig1, ax1 = plt.subplots()
    ax1.scatter(features.iloc[:, 0], features.iloc[:, 1], alpha=alpha_value)
    ax1.set_title("Before Reduction")
    ax1.set_xlabel(feature_names[0])
    ax1.set_ylabel(feature_names[1])
    st.pyplot(fig1)

    st.write("### After Dimensionality Reduction")
    fig2, ax2 = plt.subplots()
    ax2.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=alpha_value, color='orange')
    ax2.set_title("After Reduction")
    ax2.set_xlabel("UMAP Component 1")
    ax2.set_ylabel("UMAP Component 2")
    st.pyplot(fig2)
else:
    st.write("Please upload a CSV file to begin.")
