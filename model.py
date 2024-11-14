import pandas as pd
import numpy as np
import warnings
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Load the dataset
data = pd.read_csv("dataset.csv")

# Map categorical/binary fields as needed, e.g., {'Yes': 1, 'No': 0} if applicable
# Assuming the dataset has columns matching those in the form

# Selecting features based on form inputs
X = data[['Own_car', 'Own_property', 'Work_phone', 'Unemployed', 
          'Num_children', 'Num_family', 'Total_income', 'Years_employed']]
y = data['Target']  # Assuming 'Target' is the name of the column for prediction

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(X_train)
x_test = scaler.transform(X_test)

# Train the Logistic Regression model
clf_lr = LogisticRegression()
clf_lr.fit(x_train, y_train)

# Make predictions and evaluate
predictions = clf_lr.predict(x_test)

# Display evaluation metrics
print("Confusion Matrix:\n", confusion_matrix(predictions, y_test))
print("\nClassification Report:\n", classification_report(predictions, y_test), "\n")

# Save the model and scaler using pickle
pickle.dump(clf_lr, open('model.pkl', 'wb'))
pickle.dump(scaler, open('scaler.pkl', 'wb'))

# Load model to confirm saving was successful
model = pickle.load(open('model.pkl', 'rb'))
print("Model loaded successfully:", model)
