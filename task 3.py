import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# Step 1: Load Dataset
file_path = "car data.csv"

# Check if the file exists
try:
    df = pd.read_csv(file_path)
    print("✅ Dataset loaded successfully!")
except FileNotFoundError:
    raise FileNotFoundError(f"❌ Error: Dataset file '{file_path}' not found. Please check the file path.")
except Exception as e:
    raise Exception(f"❌ Error loading the dataset: {e}")

# Step 2: Inspect column names
print("Columns in the dataset:", df.columns)

# Clean up column names (remove any leading/trailing spaces)
df.columns = df.columns.str.strip()

# Check if the required columns exist
required_columns = ['brand', 'fuel_type', 'transmission', 'mileage', 'horsepower', 'price']
missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    print(f"❌ Missing required columns: {missing_columns}")
    print("Available columns:", df.columns)
    raise KeyError("Please check the dataset and ensure that it contains all required columns.")

# Step 3: Preprocess Data (Handle missing values, categorical encoding)
df.dropna(inplace=True)  # Drop rows with missing values

# Step 4: One-Hot Encoding for categorical features
df = pd.get_dummies(df, columns=['brand', 'fuel_type', 'transmission'], drop_first=True)

# Step 5: Feature Selection and Scaling
df[['mileage', 'horsepower']] = StandardScaler().fit_transform(df[['mileage', 'horsepower']])  # Scale numeric columns

# Step 6: Define Target and Features
X = df.drop(columns=['price'])  # Features
y = df['price']  # Target

# Step 7: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 8: Train the Model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Step 9: Make Predictions
y_pred = model.predict(X_test)

# Step 10: Evaluate the Model
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"✅ Model Evaluation:")
print(f"Mean Absolute Error (MAE): {mae}")
print(f"Root Mean Squared Error (RMSE): {rmse}")
print(f"R² Score: {r2}")

# Step 11: Save the Model
joblib.dump(model, 'car_price_model.pkl')
print("✅ Model saved successfully as 'car_price_model.pkl'!")
