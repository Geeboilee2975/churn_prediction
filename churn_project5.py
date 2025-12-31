import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier



# TASK 1: DATA PREPARATION
# ==========================================
# Aim: To load dataset, handle missing values, and encode variables[cite: 41, 44].
df = pd.read_csv('customer_churn.csv') 
df.dropna(inplace=True) 

# Encoding categorical variables [cite: 44, 48]
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])
print("Task 1: Data Preparation Complete.")

# ==========================================
# TASK 2: SPLITTING DATA FOR TRAINING AND TESTING
# ==========================================
# Aim: Divided data into 80% training and 20% testing sets[cite: 49, 51].
X = df.drop('Churn', axis=1) 
y = df['Churn'] 

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42) 
print(f"Task 2: Data Split (80/20) Complete.")

# ==========================================
# TASK 3: FEATURE SELECTION
# ==========================================
# Aim: Identify influential attributes like contract, charges, and tenure[cite: 57, 61].
selected_features = ['Contract', 'MonthlyCharges', 'tenure'] 

X_train_selected = X_train[selected_features] 
X_test_selected = X_test[selected_features] 
print(f"Task 3: Feature Selection Complete ({selected_features}).")

# ==========================================
# TASK 4: MODEL SELECTION
# ==========================================
# Aim: Choose a suitable binary classification algorithm[cite: 68, 69].
# We are selecting Random Forest as suggested in options[cite: 69].
model = RandomForestClassifier(n_estimators=100, random_state=42) 
print("Task 4: Model Selection (Random Forest) Complete.")

# ==========================================
# TASK 5: MODEL TRAINING
# ==========================================
# Aim: Train the model using the training dataset and features[cite: 78, 80].
model.fit(X_train_selected, y_train) 
print("Task 5: Model Training Complete.")




from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# ==========================================
# TASK 6: MODEL EVALUATION
# ==========================================
# Aim: Assess the model's performance on the testing dataset[cite: 85, 87].

# 1. Use the trained model to make predictions on the test set
y_pred = model.predict(X_test_selected)
y_pred_proba = model.predict_proba(X_test_selected)[:, 1] # Needed for ROC-AUC

# 2. Calculate and print the metrics 
print("\n--- Task 6: Model Evaluation Results ---")
print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_pred_proba):.4f}")

print("\nAll tasks completed successfully!")

import joblib

# 1. Save the model
joblib.dump(model, 'churn_model.pkl')

# 2. Because you used LabelEncoder on 'Contract', we need to save 
# how it mapped the words (e.g., 'One year' -> 1). 
# For 3 features, it's easier to just remember the mapping:
# Month-to-month = 0, One year = 1, Two year = 2
print("Model saved as churn_model.pkl")