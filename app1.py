import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.decomposition import PCA
import time
import matplotlib.pyplot as plt
import numpy as np

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
    st.write("### Dataset")
    st.write("### Dataset Head")
    st.dataframe(wine_data.head())

    # Check class distribution to identify class imbalance
    st.write("### Target Class Distribution")
    st.write(wine_data['quality'].value_counts())

    # Select Features and Target
    features = wine_data.drop(columns=['quality', 'Id'])
    target = wine_data['quality']
    feature_names = features.columns

    # Check for missing values
    st.write("### Missing Values in Data")
    st.write(wine_data.isnull().sum())

    # Standardize Features
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(features)

    # Elbow Method: Evaluate PCA performance with varying components
    st.write("### Elbow Method for PCA")
    max_components = len(features.columns)
    pca = PCA()
    pca.fit(scaled_features)

    explained_variance = np.cumsum(pca.explained_variance_ratio_)

    # Plot explained variance
    fig, ax = plt.subplots()
    ax.plot(range(1, max_components + 1), explained_variance, marker='o', linestyle='--')
    ax.set_title('Elbow Method: Explained Variance vs Number of Components')
    ax.set_xlabel('Number of Principal Components')
    ax.set_ylabel('Cumulative Explained Variance')
    st.pyplot(fig)

    # Choose the number of components using a slider
    selected_components = st.sidebar.slider("Select Number of PCA Components", 1, max_components, 2)
    st.write(f"Selected Components: {selected_components}")

    # Apply PCA with the selected number of components
    pca = PCA(n_components=selected_components)
    reduced_features = pca.fit_transform(scaled_features)

    # Split Data
    X_train, X_test, y_train, y_test = train_test_split(reduced_features, target, test_size=0.3, random_state=42)

    # Train Random Forest Model
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    time_taken = time.time() - start_time

    # Display Results
    st.write("### Random Forest Results")
    st.write(f"Accuracy: {accuracy:.2f}")
    st.write(f"Time Taken: {time_taken:.2f} seconds")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("### Confusion Matrix")
    st.write(cm)

    # Display PCA Components
    st.write("### PCA Components Visualization")
    fig2, ax2 = plt.subplots()
    ax2.scatter(reduced_features[:, 0], reduced_features[:, 1], alpha=0.5, color='orange')
    ax2.set_title("Reduced Data (First Two Principal Components)")
    ax2.set_xlabel("Principal Component 1")
    ax2.set_ylabel("Principal Component 2")
    st.pyplot(fig2)

else:
    st.write("Please upload a CSV file to begin.")
