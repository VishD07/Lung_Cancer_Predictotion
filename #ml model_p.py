import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load the dataset
file_path = "C:\\Users\\vishw\\Downloads\\AutoML_Sample _Dataset_modified.csv"  # Update with actual dataset path
data = pd.read_csv(file_path)

# Retain only general features
selected_features = ['Age', 'Smoking_History', 'Packs_consumed_day', 'Alcohol_Consumption', 'TARGET']
data = data[selected_features]

# Encode categorical variables
label_encoders = {}
for col in ['Smoking_History', 'Alcohol_Consumption', 'TARGET']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le

# Split into features and target
X = data.drop(columns=['TARGET'])
y = data['TARGET']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model
model = RandomForestClassifier(random_state=42, n_estimators=100)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy}")
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save the retrained model
joblib.dump(model, 'C:\\Users\\vishw\\OneDrive\\Desktop\\Streamlit project\\retained_model.pkl')
print("Model saved as 'retrained_model.pkl'")
