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

# Challenge 1: Feature Scaling
from sklearn.preprocessing import StandardScaler

print("\n" + "="*50)
print("CHALLENGE 1: FEATURE SCALING")
print("="*50)

# Create scaler
scaler = StandardScaler()

# Fit on training data and transform both sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train KNN with scaled data
knn_scaled = KNeighborsClassifier(n_neighbors=7)
knn_scaled.fit(X_train_scaled, y_train)

# Make predictions
y_pred_scaled = knn_scaled.predict(X_test_scaled)

# Compare results
print("\nBefore scaling (K=7):")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred):.4f}")

print("\nAfter scaling (K=7):")
print(f"  Accuracy:  {accuracy_score(y_test, y_pred_scaled):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_scaled):.4f}")
print(f"  Recall:    {recall_score(y_test, y_pred_scaled):.4f}")


# Challenge 2: Feature Importance Analysis
print("\n" + "="*50)
print("CHALLENGE 2: FEATURE IMPORTANCE ANALYSIS")
print("="*50)

# Calculate correlation of all features with Churn
correlations = df.corr()['Churn'].drop('Churn')

# Sort by absolute value to find strongest relationships
correlations_sorted = correlations.abs().sort_values(ascending=False)

print("\nTop 10 features most correlated with Churn:")
print(correlations_sorted.head(10))

# Visualize top 15 correlations
plt.figure(figsize=(10, 8))
correlations.abs().sort_values(ascending=False).head(15).plot(kind='bar')
plt.title('Top 15 Features Correlated with Churn')
plt.xlabel('Feature')
plt.ylabel('Absolute Correlation')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('data/feature_importance.png')
print("Feature importance visualization saved!")

"""
df.corr()['Churn'] — calculates correlation between every feature and every other feature, then we pick just the Churn column — giving us one correlation value per feature.
.drop('Churn') — removes Churn's correlation with itself (which is always 1.0 — useless).
.abs() — takes the absolute value because we care about the strength of the relationship, not the direction. A correlation of -0.35 is just as strong as +0.35.
.sort_values(ascending=False) — sorts from strongest to weakest correlation.
"""


# Challenge 3: Other Algorithms
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

print("\n" + "="*50)
print("CHALLENGE 3: COMPARING ALGORITHMS")
print("="*50)

# Define all models
models = {
    'KNN (K=7, scaled)': KNeighborsClassifier(n_neighbors=7),
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVC': SVC(random_state=42)
}

# Train and evaluate each model
results = []
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred_model = model.predict(X_test_scaled)
    
    acc = accuracy_score(y_test, y_pred_model)
    prec = precision_score(y_test, y_pred_model)
    rec = recall_score(y_test, y_pred_model)
    
    results.append({'model': name, 'accuracy': acc, 'precision': prec, 'recall': rec})
    print(f"\n{name}:")
    print(f"  Accuracy:  {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall:    {rec:.4f}")

# Find best model by recall
best = max(results, key=lambda x: x['recall'])
print(f"\nBest model by recall: {best['model']} with recall={best['recall']:.4f}")

# Challenge 4: Advanced Evaluation
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import cross_val_score

print("\n" + "="*50)
print("CHALLENGE 4: ADVANCED EVALUATION")
print("="*50)

# Use Logistic Regression as our best model
best_model = LogisticRegression(max_iter=1000)
best_model.fit(X_train_scaled, y_train)

# ROC Curve
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

print(f"\nAUC Score: {roc_auc:.4f}")

# Plot ROC curve
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='red', linestyle='--', label='Random Guessing')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve - Logistic Regression')
plt.legend()
plt.tight_layout()
plt.savefig('data/roc_curve.png')
print("ROC curve saved!")

# Cross-validation
print("\nCross-validation (5 folds):")
cv_accuracy = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='accuracy')
cv_recall = cross_val_score(best_model, X_train_scaled, y_train, cv=5, scoring='recall')

print(f"Accuracy: {cv_accuracy.mean():.4f} (+/- {cv_accuracy.std():.4f})")
print(f"Recall:   {cv_recall.mean():.4f} (+/- {cv_recall.std():.4f})")

"""
predict_proba() — instead of returning 0/1 predictions, returns the probability of each class. [:, 1] takes the probability of churn (class 1) for each customer.
roc_curve() — takes the real labels and probabilities and calculates the false positive rate and true positive rate at every possible threshold.
+/- std — the standard deviation across the 5 folds. A small std means the model performs consistently — not just getting lucky on one particular split.
"""



# Challenge 5: Business Impact Analysis
print("\n" + "="*50)
print("CHALLENGE 5: BUSINESS IMPACT ANALYSIS")
print("="*50)

# Define business costs
MONTHLY_REVENUE = 64.76  # average monthly charges from our data
COST_FALSE_NEGATIVE = MONTHLY_REVENUE * 12  # lost customer = 12 months revenue
COST_FALSE_POSITIVE = MONTHLY_REVENUE * 1   # unnecessary retention = 1 month

# Get confusion matrix values
cm = confusion_matrix(y_test, best_model.predict(X_test_scaled))
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix breakdown:")
print(f"True Negatives  (correct no churn): {tn}")
print(f"False Positives (unnecessary retention): {fp}")
print(f"False Negatives (missed churners): {fn}")
print(f"True Positives  (caught churners): {tp}")

# Calculate costs
cost_missed = fn * COST_FALSE_NEGATIVE
cost_unnecessary = fp * COST_FALSE_POSITIVE
total_cost = cost_missed + cost_unnecessary

# Calculate cost of no model (missing all churners)
cost_no_model = (fn + tp) * COST_FALSE_NEGATIVE

print(f"\nBusiness Impact:")
print(f"Cost of missed churners: ${cost_missed:,.2f}")
print(f"Cost of unnecessary retention: ${cost_unnecessary:,.2f}")
print(f"Total cost with model: ${total_cost:,.2f}")
print(f"\nCost without any model: ${cost_no_model:,.2f}")
print(f"Money saved by using model: ${cost_no_model - total_cost:,.2f}")

# Find optimal threshold
print("\n" + "="*50)
print("OPTIMAL THRESHOLD ANALYSIS")
print("="*50)

thresholds_to_try = [0.3, 0.4, 0.5, 0.6, 0.7]
y_prob = best_model.predict_proba(X_test_scaled)[:, 1]

for threshold in thresholds_to_try:
    y_pred_threshold = (y_prob >= threshold).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_threshold)
    tn_t, fp_t, fn_t, tp_t = cm_t.ravel()
    
    cost = (fn_t * COST_FALSE_NEGATIVE) + (fp_t * COST_FALSE_POSITIVE)
    print(f"Threshold {threshold}: Total cost=${cost:,.2f} | Caught {tp_t} churners | Missed {fn_t}")

print(f"\nAverage monthly revenue per customer: ${MONTHLY_REVENUE}")
print(f"Cost per missed churner: ${COST_FALSE_NEGATIVE:,.2f}")
print(f"Cost per false alarm: ${COST_FALSE_POSITIVE:,.2f}")