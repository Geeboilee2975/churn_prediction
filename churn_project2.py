import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- Include Task 1 Logic (Necessary to have data to split) ---
df = pd.read_csv('customer_churn.csv') 
df.dropna(inplace=True)
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# --- TASK 2: SPLIT DATA ---
# Identify the target variable "Churn" [cite: 81]
X = df.drop('Churn', axis=1) 
y = df['Churn'] 

# Divide the data into training (80%) and testing (20%) 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

print("Task 2: Data split successfully.")
print(f"Training set size: {len(X_train)}")
print(f"Testing set size: {len(X_test)}")


# Test: Verify the 80/20 split
total_rows = len(df)
train_rows = len(X_train)
test_rows = len(X_test)

print(f"Total Rows in Dataset: {total_rows}")
print(f"Training Rows (Target 80%): {train_rows}")
print(f"Testing Rows (Target 20%): {test_rows}")

# Mathematical Verification
if abs((train_rows / total_rows) - 0.8) < 0.01:
    print("Success: Data split is exactly 80/20!")
else:
    print("Warning: Split ratio is incorrect.")