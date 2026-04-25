"""
Customer Churn Prediction
Author: Ioanna Renta
Description: Analyze customer churn data and predict likelihood of churn
"""
 
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, classification_report
 
# Set up paths
DATA_DIR = Path("data")
CSV_FILE = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
#JSON_FILE = DATA_DIR / "titanic_processed.json"
 
# Create data directory if it doesn't exist
DATA_DIR.mkdir(exist_ok=True)
 
print("Project setup complete!")
print(f"Data directory: {DATA_DIR}")
print(f"CSV file location: {CSV_FILE}")

df = pd.read_csv(CSV_FILE)
df.head()
print(f"Dataset loaded successfully! Shape: {df.shape}") # Print the shape of the dataset
print(f"\nColumns: {list(df.columns)}") # Print the column names    
print(f"\nFirst few rows:")
print(df.head())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum())        

# Select numeric columns only
print("\ndf.describefunction:")
print(df.describe())

numeric_columns = df.select_dtypes(include=[np.number]) # Select only numeric columns for analysis  
print("\nNumeric Columns (using select_dtypes):")
print(numeric_columns.head()) 

#check data types
print("\nData Types:")
print(df.dtypes)

#check the target variable distribution
print("\nTarget Variable Distribution:")
print(df['Churn'].value_counts())

#drop Customer ID column as it is not useful for prediction
df = df.drop(columns=['customerID'])

#fix total charges column which has some non-numeric values
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median()) # Fill missing
print("\nTotalCharges column after fixing:")
print(df['TotalCharges'].head())

# convert churn to binary
df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})
print("\nChurn column after conversion:")
print(df['Churn'].head())

""" visualization of churn distribution """

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle('Key Features vs Churn', fontsize=16)

# Plot 1: Churn distribution
df['Churn'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Churn Distribution')
axes[0].set_xlabel('Churn')
axes[0].set_ylabel('Count')

# Plot 2: Monthly Charges by Churn
df.groupby('Churn')['MonthlyCharges'].mean().plot(kind='bar', ax=axes[1], color=['green', 'red'])
axes[1].set_title('Average Monthly Charges by Churn')
axes[1].set_xlabel('Churn')
axes[1].set_ylabel('Average Monthly Charges')

# Plot 3: Tenure by Churn
df.groupby('Churn')['tenure'].mean().plot(kind='bar', ax=axes[2], color=['green', 'red'])
axes[2].set_title('Average Tenure by Churn')
axes[2].set_xlabel('Churn')
axes[2].set_ylabel('Average Tenure (months)')

plt.tight_layout()
plt.savefig('data/churn_visualization.png')
plt.show()
print("Visualization saved!")

""" end of visualization section    """

# Encode categorical variables using one-hot encoding
# One-hot encode all remaining object columns
df = pd.get_dummies(df, drop_first=True)

# Verify
print(f"Shape after preprocessing: {df.shape}")
print(f"\nData types after encoding:")
print(df.dtypes)
print(f"\nFirst few rows:")
print(df.head())

# Separate features and target
X = df.drop(columns=['Churn'])
y = df['Churn']

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")
print(f"\nChurn distribution:")
print(y.value_counts())   

# Step 3: Split the data
print("\n" + "="*50)
print("SPLITTING DATA")
print("="*50)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print(f"Training set: {X_train.shape}")
print(f"Test set: {X_test.shape}")
print(f"\nTraining churn distribution:")
print(y_train.value_counts())
print(f"\nTest churn distribution:")
print(y_test.value_counts())

# Step 4: Train KNN Model
print("\n" + "="*50)
print("TRAINING KNN MODEL")
print("="*50)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

print(f"Model trained successfully!")
print(f"Training with K=5 and {X_train.shape[0]} samples")

# Step 5: Make Predictions and Evaluate
print("\n" + "="*50)
print("MODEL EVALUATION")
print("="*50)

# Make predictions
y_pred = knn.predict(X_test)

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")

# Confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# Classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Step 6: Experiment with different K values
print("\n" + "="*50)
print("FINDING BEST K VALUE")
print("="*50)

k_values = [1, 3, 5, 7, 9, 11, 15]
results = []

for k in k_values:
    # Train model with this K
    knn_k = KNeighborsClassifier(n_neighbors=k)
    knn_k.fit(X_train, y_train)
    
    # Make predictions
    y_pred_k = knn_k.predict(X_test)
    
    # Calculate metrics
    acc = accuracy_score(y_test, y_pred_k)
    prec = precision_score(y_test, y_pred_k)
    rec = recall_score(y_test, y_pred_k)
    
    results.append({'k': k, 'accuracy': acc, 'precision': prec, 'recall': rec})
    print(f"K={k:2d} | Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f}")

# Find best K by recall
best = max(results, key=lambda x: x['recall'])
print(f"\nBest K by recall: K={best['k']} with recall={best['recall']:.4f}")