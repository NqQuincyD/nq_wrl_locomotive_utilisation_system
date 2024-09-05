import streamlit as st
import pickle
from pathlib import Path
import streamlit_authenticator as stauth
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import datetime
import sqlite3
import random
from sklearn.multioutput import MultiOutputRegressor
import bcrypt
import re


# Initialize the database
def init_db():
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (username TEXT PRIMARY KEY, password TEXT, is_admin INTEGER)''')
    conn.commit()
    conn.close()

# Check if a user exists
def user_exists(username):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    user = c.fetchone()
    conn.close()
    return user is not None

# Add a new user
def add_user(username, password, is_admin):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    c.execute("INSERT INTO users (username, password, is_admin) VALUES (?, ?, ?)",
              (username, hashed_password, is_admin))
    conn.commit()
    conn.close()

# Verify user credentials
def verify_user(username, password, is_admin):
    conn = sqlite3.connect('users.db')
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username=? AND is_admin=?", (username, is_admin))
    user = c.fetchone()
    conn.close()
    if user and bcrypt.checkpw(password.encode('utf-8'), user[1]):
        return True
    return False

# Check password strength
def is_strong_password(password):
    if len(password) < 8:
        return False
    if not re.search(r"[A-Z]", password):
        return False
    if not re.search(r"[a-z]", password):
        return False
    if not re.search(r"\d", password):
        return False
    if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
        return False
    return True

# Load the dataset
def load_data():
    return pd.read_csv("Loco Data.csv")

# Multi-output performance prediction
def predict_performance(df, loco_type, loco_number, year):
    filtered_df = df[(df['LOCO_TYPE'] == loco_type) & (df['LOCO_NUMBER'] == loco_number)]
    
    if filtered_df.empty:
        return None
    
    X = filtered_df[['YEAR']]
    y = filtered_df[['Availability_Days', 'Train_kms', 'Train_km_per_day', 
                     'Failures_in_section', 'Reliabilty', 'Days_before_failure']]
    
    model = MultiOutputRegressor(LinearRegression())
    model.fit(X, y)
    
    prediction = model.predict(np.array([[year]]))
    return prediction[0]

# Generate recommendations based on predictions
def get_recommendation(predictions):
    availability_days = predictions[0]
    failures_in_section = predictions[3]
    
    if failures_in_section > 5:  # Arbitrary threshold for failures
        return "Not recommended to use this locomotive due to high failure predictions."
    elif availability_days < 100:  # Another threshold for availability
        return "Recommended to avoid using this locomotive as it shows low availability."
    else:
        return "Recommended to use this locomotive as it shows good performance indicators."

# Plot predictions
def plot_predictions(year, predictions):
    labels = ['Availability Days', 'Train Kms', 'Train Kms per Day', 
              'Failures in Section', 'Reliability', 'Days Before Failure']
    
    plt.figure(figsize=(10, 5))
    sns.barplot(x=labels, y=predictions)
    plt.title(f"Predictions for Year {year}")
    plt.ylabel("Predicted Values")
    st.pyplot(plt)

# Plot line graph
def plot_line_graph(years, predictions):
    labels = ['Availability Days', 'Train Kms', 'Train Kms per Day', 
              'Failures in Section', 'Reliability', 'Days Before Failure']
    
    plt.figure(figsize=(12, 6))
    for i, label in enumerate(labels):
        plt.plot(years, predictions[:, i], marker='o', label=label)
    
    plt.title("Predicted Values Over Years")
    plt.xlabel("Year")
    plt.ylabel("Predicted Values")
    plt.legend()
    plt.grid()
    st.pyplot(plt)

# Initialize the database
init_db()

# Streamlit app
st.title("Locomotive Utilisation System")

# Initialize session state for admin verification and user
if 'admin_verified' not in st.session_state:
    st.session_state.admin_verified = False
if 'user' not in st.session_state:
    st.session_state.user = None

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Login", "Sign Up", "Predict"])

if page == "Login":
    st.header("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    user_type = st.radio("Login as:", ("User", "Admin"))
    
    if st.button("Login"):
        is_admin = 1 if user_type == "Admin" else 0
        if verify_user(username, password, is_admin):
            st.session_state.user = username
            st.session_state.admin_verified = is_admin
            st.success(f"Logged in successfully as {user_type}")
        else:
            st.error("Invalid username or password")

elif page == "Sign Up":
    st.header("Sign Up")
    
    if not st.session_state.admin_verified:
        # Admin verification input
        admin_username = st.text_input("Admin Username (to access Sign Up)", placeholder="Enter admin username")
        admin_password = st.text_input("Admin Password", type="password")
        
        if st.button("Verify Admin"):
            if verify_user(admin_username, admin_password, is_admin=1):
                st.session_state.admin_verified = True
                st.success("Admin verified. You can now sign up users.")
            else:
                st.error("Admin verification failed.")
    else:
        # Show user creation form
        new_username = st.text_input("New Username")
        new_password = st.text_input("New Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")
        user_type = st.radio("User Type:", ("User", "Admin"))
        
        if st.button("Sign Up"):
            if new_password != confirm_password:
                st.error("Passwords do not match")
            elif not is_strong_password(new_password):
                st.error("Password is not strong enough. It should have at least 8 characters, including uppercase and lowercase letters, numbers, and special characters.")
            elif user_exists(new_username):
                st.error("Username already exists")
            else:
                is_admin = 1 if user_type == "Admin" else 0
                add_user(new_username, new_password, is_admin)
                st.success("User created successfully")
                # Clear input for new user
                new_username = st.text_input("New Username")  
                new_password = st.text_input("New Password", type="password")
                confirm_password = st.text_input("Confirm Password", type="password")

elif page == "Predict":
    if st.session_state.user:
        st.write(f"Welcome, {st.session_state.user}!")
        st.button("Logout", on_click=lambda: st.session_state.clear())
        
        df = load_data()
        loco_type = st.selectbox("Select Loco Type", df['LOCO_TYPE'].unique())
        loco_number = st.selectbox("Select Loco Number", df[df['LOCO_TYPE'] == loco_type]['LOCO_NUMBER'].unique())
        year = st.number_input("Predict Year", min_value=2025, max_value=2030, value=2025)
        
        if st.button("Predict"):
            # Prepare years for line graph
            years = np.arange(2025, 2031)  # Example range of years for predictions
            all_predictions = []

            for y in years:
                predictions = predict_performance(df, loco_type, loco_number, y)
                if predictions is not None:
                    all_predictions.append(predictions)
                else:
                    all_predictions.append([np.nan] * 6)  # Fill with NaN if no data

            all_predictions = np.array(all_predictions)  # Convert to NumPy array

            # Show predictions for the selected year
            current_predictions = all_predictions[0]  # Predictions for the selected year
            st.write(f"Predictions for {year}:")
            prediction_df = pd.DataFrame({
                'Metric': ['Availability Days', 'Train Kms', 'Train Kms per Day', 
                           'Failures in Section', 'Reliability', 'Days Before Failure'],
                'Predicted Value': current_predictions
            })
            st.table(prediction_df)  # Display predictions in table format
            
            # Generate and display recommendation
            recommendation = get_recommendation(current_predictions)
            st.write("Recommendation:", recommendation)
            
            # Plot bar chart and line graph
            plot_predictions(year, current_predictions)
            plot_line_graph(years, all_predictions)
    else:
        st.error("Please log in to access the prediction functionality.")

# Add some CSS to improve the app's appearance
st.markdown("""
<style>
    .stRadio > label {
        font-weight: bold;
        color: #4A4A4A;
    }
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-weight: bold;
    }
    .stTextInput > div > div > input {
        background-color: #F0F2F6;
    }
</style>
""", unsafe_allow_html=True)




