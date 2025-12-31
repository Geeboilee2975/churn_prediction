import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# --- TASK 1: DATA PREPARATION [cite: 41] ---
# Load and preprocess the dataset, addressing missing values and encoding variables[cite: 44].
df = pd.read_csv('customer_churn.csv') # Changed to read_csv for your file

# Addressing missing values [cite: 44, 47]
df.dropna(inplace=True)

# Encoding categorical variables for machine learning readiness [cite: 44, 48]
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])
print("Task 1: Data Preparation Complete.")

# --- TASK 2: SPLIT DATA [cite: 49] ---
# Divide data into training (80%) and testing (20%)[cite: 51].
X = df.drop('Churn', axis=1) 
y = df['Churn'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
print("Task 2: Data Split 80/20 Complete.")

# --- TASK 4: MODEL SELECTION [cite: 68] ---
# Choose a suitable binary classification algorithm[cite: 69]. 
# We are using Random Forest as it is excellent for churn prediction[cite: 69].
model = RandomForestClassifier(random_state=42)
print("Task 4: Model (Random Forest) Selected.")

# --- TASK 5: MODEL TRAINING [cite: 78] ---
# Train the model using the training dataset and the "Churn" target variable[cite: 80, 81].
model.fit(X_train, y_train)

print("Task 5: Model Training Complete.")