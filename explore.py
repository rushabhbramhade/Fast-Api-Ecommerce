from sklearn.datasets import fetch_california_housing # built-in dataset collection
import pandas as pd

# Explanation of how this code loads the built-in dataset without external files:
# 1. The fetch_california_housing function is part of scikit-learn's built-in dataset collection
#    This dataset is pre-included in the sklearn library, so you don't need to download or provide any external files
# 2. When you call the function, it loads the pre-packaged California Housing dataset directly from the sklearn installation
data = fetch_california_housing()  # Loads the in-memory built-in dataset and returns a Bunch object (similar to a dictionary)
# data.data contains the raw feature matrix (numerical values for all housing features)
# data.feature_names contains the column names for each feature in the dataset
df = pd.DataFrame(data.data, columns=data.feature_names)  # Converts the raw feature matrix into a human-readable pandas DataFrame

df["price"] = data.target  # Adds the target variable (house price) as a new column in the DataFrame
print("Shape of the DataFrame :",df.shape)
print("data head:",df.head())
print("describe:",df.describe())# Provides a summary of the dataset, including mean, median, standard deviation, min, max, and percentiles
print("data info:",df.info())# Provides information about the dataset, including column types, non-null values, and memory usage
print("data describe:",df.describe())# Provides a summary of the dataset, including mean, median, standard deviation, min, max, and percentiles