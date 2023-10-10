# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# Set the title and description of the app
st.title("Salary Prediction App")
st.write("Predict salary based on years of experience and education level")

# Upload dataset
st.write("Upload your dataset:")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)

    # Display the dataset
    st.write("Preview of the dataset:")
    st.write(data.head())

    # Data preprocessing
    X = data[['YearsExperience', 'EducationLevel']]
    y = data['Salary']

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

    # Build and train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test)

    # Calculate the Mean Squared Error (MSE) as a performance metric
    mse = mean_squared_error(y_test, y_pred)
    st.write(f"Mean Squared Error (MSE): {mse}")

    # Create a scatter plot of actual vs. predicted salaries
    fig, ax = plt.subplots()
    ax.scatter(y_test, y_pred)
    ax.set_xlabel('Actual Salary')
    ax.set_ylabel('Predicted Salary')
    ax.set_title('Actual vs. Predicted Salary')
    st.pyplot(fig)

    # Create a form for user input
    st.write("Enter the number of years of experience and education level to predict salary:")
    years_of_experience = st.number_input("Years of Experience", min_value=0, max_value=50, value=5)
    education_level = st.slider("Education Level", min_value=0, max_value=20, value=10)

    # Predict salary based on user input
    user_input = pd.DataFrame({'YearsExperience': [years_of_experience], 'EducationLevel': [education_level]})
    predicted_salary = model.predict(user_input)
    st.write(f"Predicted Salary: {predicted_salary[0]:.2f}")
