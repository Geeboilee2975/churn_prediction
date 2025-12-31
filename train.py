import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load data
df = pd.read_csv('customer_churn.csv') 

# 1. We ONLY keep these 3 columns for training
X = df[['Contract', 'MonthlyCharges', 'tenure']].copy()
y = df['Churn']

# 2. Encode the words into numbers
le = LabelEncoder()
X['Contract'] = le.fit_transform(X['Contract'])
y = le.fit_transform(y) # No=0, Yes=1

# 3. Train the model on ONLY these 3 features
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# 4. Save the new "3-Feature Brain"
joblib.dump(model, 'churn_model.pkl')
print("SUCCESS: Model re-trained on 3 features and saved!")