### Commit: Add core imports to setup California housing price model training pipeline

# Import standard data science libraries for model training workflow:
# 1. Scikit-learn tools: Load dataset, ML model, train-test split, and performance metrics
# 2. Pandas: For structured data handling and analysis
# 3. Joblib: For saving the trained model to disk after training completes

from sklearn.datasets import fetch_california_housing # built-in dataset collection
from sklearn.ensemble import RandomForestRegressor # ML model: Random Forest
from sklearn.model_selection import train_test_split # train-test split for model evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # performance metrics
import pandas as pd # data manipulation and analysis
import joblib # model serialization/deserialization for disk storage

#load dataset 
print("Loading the California Housing dataset...")
data = fetch_california_housing()
print("Dataset loaded successfully.")

x = pd.DataFrame(data.data,columns=data.feature_names)
y = pd.Series(data.target, name="price") # Fix: Use Series instead of DataFrame to avoid 1d array warning

print(f"total records: {x.shape[0]}")

## Split the dataset into training and testing sets 
print("Splitting the dataset into training and testing sets...")
x_train, x_test, y_train, y_test = train_test_split( x,y,
    test_size=0.2,
    random_state=42 # random seed -- 42 we used
)

## train the model 
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)
model.fit(x_train,y_train)
print("Model training completed.")

y_pred = model.predict(x_test)
mae = mean_absolute_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
print(f"average absolute error: {mae * 100000:.2f}") # Convert to dollars (dataset uses $100k units)
print(f"R^2 score: {r2:.2f}")
print("First 10 predictions:", y_pred[:10])

## Save the trained model to disk for future use or deployment
joblib.dump(model, "house_model.joblib")
joblib.dump(list(x.columns), "house_feature_names.joblib")
joblib.dump({"mae": mae}, "house_model_metadata.joblib")
print("Trained model saved successfully.")