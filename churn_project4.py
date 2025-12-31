
# --- TASKS 1, 2, & 3 logic must be included at the top ---
# (Loading data, splitting 80/20, and selecting features)

from sklearn.ensemble import RandomForestClassifier

# --- TASK 4: MODEL SELECTION ---
# We are choosing a Random Forest Classifier as our binary classification algorithm.
# This choice is based on the dataset's characteristics (mix of data types).
model = RandomForestClassifier(n_estimators=100, random_state=42)

print("Task 4 Complete: Random Forest algorithm selected for binary classification.")

try:
    # Attempting to initialize the model (Task 4)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    print("Success: RandomForestClassifier imported and initialized correctly.")
except Exception as e:
    print(f"Error: {e}")