import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# --- TASK 1: DATA PREPARATION ---
df = pd.read_csv('customer_churn.csv') 
df.dropna(inplace=True)
le = LabelEncoder()
for col in df.select_dtypes(include=['object']).columns:
    df[col] = le.fit_transform(df[col])

# --- TASK 2: DATA SPLITTING ---
X = df.drop('Churn', axis=1) 
y = df['Churn'] 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# --- TASK 3: FEATURE SELECTION ---
# Identify and select relevant features influencing churn prediction [cite: 57, 60]
# Specific attributes required: contract type, monthly charges, and tenure [cite: 61]
selected_features = ['Contract', 'MonthlyCharges', 'tenure']

X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]

print("Task 3 Complete: Relevant features selected.")
print(f"Selected Attributes: {selected_features}")
