# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 2: Load the dataset
# Ensure 'Advertising.csv' is in the same directory as the script, or provide the full path if needed

# If running in a Jupyter notebook, make sure the file is in the current working directory
try:
    df = pd.read_csv('Advertising.csv')  # Replace with full path if necessary
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: The file 'Advertising.csv' was not found. Please check the file path.")
    exit()

# Display the first few rows of the dataset
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Step 3: Check for missing values and get summary statistics
print("\nChecking for missing values:")
print(df.isnull().sum())  # Check if there are any missing values

print("\nSummary statistics of the dataset:")
print(df.describe())  # Get summary statistics like mean, std, etc.

# Step 4: Visualize the relationships between features
# Plot pairplot to visualize relationships between the features and target
sns.pairplot(df)

# Step 5: Visualize the correlation matrix
correlation_matrix = df.corr()  # Calculate the correlation matrix
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")  # Heatmap to visualize correlations
plt.show()

# Step 6: Prepare data for modeling
X = df[['TV', 'Radio', 'Newspaper']]  # Features (independent variables)
y = df['Sales']  # Target variable (dependent variable)

# Step 7: Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape}, Test set size: {X_test.shape}")

# Step 8: Train the model (Linear Regression)
model = LinearRegression()  # Initialize the linear regression model
model.fit(X_train, y_train)  # Fit the model to the training data

# Step 9: Make predictions on the test set
y_pred = model.predict(X_test)  # Predict using the test data

# Step 10: Evaluate the model's performance

# Calculate Mean Squared Error (MSE)
mse = mean_squared_error(y_test, y_pred)
print(f"\nMean Squared Error (MSE): {mse}")

# Calculate R-squared value
r2 = r2_score(y_test, y_pred)
print(f"R-squared value: {r2}")

# Step 11: Visualize the results (Actual vs Predicted sales)
plt.scatter(y_test, y_pred)  # Scatter plot of actual vs predicted values
plt.xlabel('Actual Sales')
plt.ylabel('Predicted Sales')
plt.title('Actual vs Predicted Sales')
plt.show()
