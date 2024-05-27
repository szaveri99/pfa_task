import streamlit as st
import pandas as pd
import pickle
from sklearn.cluster import KMeans
from sklearn.svm import SVC
import joblib

# Load pre-trained models and data
# Adjust the paths as per your setup
try:
    with open('kmeans_model.pkl', 'rb') as f:
        kmeans_model = joblib.load(f)
    with open('svm_model.pkl', 'rb') as f:
        svm_model = joblib.load(f)
    label_encoder = joblib.load('label_encoder.pkl')
    scaler = joblib.load('scaler_model.pkl')
except (OSError, pickle.UnpicklingError) as e:
    st.error(f"Error loading model: {e}")

# Load datasets
try:
    train_data = pd.read_excel('train.xlsx')
    test_data = pd.read_excel('test.xlsx')
    raw_data = pd.read_excel('rawdata.xlsx')
except FileNotFoundError as e:
    st.error(f"Error loading data: {e}")

st.title("Data Science Internship Tasks")

# Task 1: Clustering
st.header("Enter the Data for Task 1 Clustering and Task 2 Classification Output")
input_data = st.text_input("Enter data point (comma-separated values):")
# input_data = "-1.26609695, -2.18753632, -0.74022886, -0.05344765,  0.06017375,-1.04987638,  0.32616787, -0.23832541,  0.53095939, -1.01418655,-1.96270317, -1.46261002, -2.22088796, -1.98578267, -2.02874303,-0.71647747,  0.58627792,  0.65475827"
if input_data:
    try:
        data_point = [float(i) for i in input_data.split(',')]
        cluster = kmeans_model.predict([data_point])[0]
        st.write(f"The data point belongs to cluster: {cluster}")
    except ValueError as e:
        st.error(f"Invalid input: {e}")

# Task 2: Classification
# input_svm_data = "-1.26609695, -2.18753632, -0.74022886, -0.05344765,  0.06017375,-1.04987638,  0.32616787, -0.23832541,  0.53095939, -1.01418655,-1.96270317, -1.46261002, -2.22088796, -1.98578267, -2.02874303,-0.71647747,  0.58627792,  0.65475827"
if input_data:
    try:
        test_svm_data = [float(i) for i in input_data.split(',')]
        X_test_scaled = scaler.transform([test_svm_data])
        
        # Predict with the SVM model
        predictions = svm_model.predict(X_test_scaled)
        
        # Decode the numerical predictions to original labels
        predicted_labels = label_encoder.inverse_transform(predictions)
        
        st.write(f"SVM Predicted Class Labels: {predicted_labels}")
        
    except Exception as e:
        st.error(f"Error in prediction: {e}")


# Task 3: Python tasks
st.header("Task 3: Data Aggregation")
try:
    st.write("Datewise total duration for each 'inside' and 'outside':")
    raw_data.columns = raw_data.columns.str.strip()
    raw_data['datetime'] = pd.to_datetime(raw_data['date'].astype(str) + ' ' + raw_data['time'].astype(str), errors='coerce')
    raw_data = raw_data.dropna(subset=['datetime'])
    raw_data = raw_data.sort_values(by='datetime')
    raw_data['duration'] = raw_data['datetime'].diff().dt.total_seconds()
    raw_data['duration'] = raw_data['duration'].fillna(0)
    total_duration = raw_data.groupby(['date', 'position'])['duration'].sum().reset_index()
    st.write(total_duration)

    st.write("Datewise number of 'picking' and 'placing' activities:")
    activity_count = raw_data.groupby(['date', 'activity']).size().reset_index(name='count')
    st.write(activity_count)
except Exception as e:
    st.error(f"Error in data aggregation: {e}")

