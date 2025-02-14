import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load dataset (Ensure the file is in the same directory)
file_name = "Unemployment in India.csv"

try:
    df = pd.read_csv(file_name)
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found. Please check the filename and directory.")
    exit()

# Display first few rows
print("\nFirst 5 rows of the dataset:")
print(df.head())

# Check for missing values
missing_values = df.isnull().sum()
print("\nMissing values in each column:\n", missing_values)

# Convert 'Date' column to datetime format (if present)
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
else:
    print("\nWarning: 'Date' column not found in the dataset.")

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Display summary statistics
print("\nSummary Statistics:")
print(df.describe())

# Unemployment rate trend over time
if 'Date' in df.columns and 'Unemployment Rate' in df.columns:
    plt.figure(figsize=(12, 6))
    sns.lineplot(x=df['Date'], y=df['Unemployment Rate'], marker='o', color='b')
    plt.title('Unemployment Rate Over Time in India')
    plt.xlabel('Date')
    plt.ylabel('Unemployment Rate (%)')
    plt.xticks(rotation=45)
    plt.grid()
    plt.show()
else:
    print("\nError: 'Date' or 'Unemployment Rate' column is missing in the dataset.")

# Unemployment rate by region (if applicable)
if 'Region' in df.columns and 'Unemployment Rate' in df.columns:
    plt.figure(figsize=(10, 5))
    sns.barplot(x=df['Region'], y=df['Unemployment Rate'], palette='viridis')
    plt.title('Unemployment Rate by Region')
    plt.xticks(rotation=90)
    plt.xlabel('Region')
    plt.ylabel('Unemployment Rate (%)')
    plt.grid()
    plt.show()
else:
    print("\nError: 'Region' or 'Unemployment Rate' column is missing in the dataset.")
