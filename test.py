import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
import math

# Load the data
train_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")

# Automatically identify categorical columns based on dtype object
categorical_cols = train_data.select_dtypes(include=['object']).columns.tolist()
numerical_cols = train_data.select_dtypes(exclude=['object']).columns.tolist()

# Make sure 'patient_id' is not included in the numerical transformations
if 'patient_id' in numerical_cols:
    numerical_cols.remove('patient_id')

# Remove the target variable from numerical columns
if 'metastatic_diagnosis_period' in numerical_cols:
    numerical_cols.remove('metastatic_diagnosis_period')

# Handle categorical variables
encoders = {}
for col in categorical_cols:
    train_data[col] = train_data[col].fillna('Unknown')  # Fill missing values with 'Unknown'
    test_data[col] = test_data[col].fillna('Unknown')
    encoders[col] = LabelEncoder()  # Fit the encoder with all data including 'Unknown'
    all_values = pd.concat([train_data[col], test_data[col]], axis=0)  # Combine to capture all possible categories
    encoders[col].fit(all_values)
    train_data[col] = encoders[col].transform(train_data[col])
    test_data[col] = test_data[col].apply(lambda x: encoders[col].transform([x])[0])

# Standardize numerical variables
scaler = StandardScaler()
train_data[numerical_cols] = scaler.fit_transform(train_data[numerical_cols].fillna(train_data[numerical_cols].mean()))
test_data[numerical_cols] = scaler.transform(test_data[numerical_cols].fillna(test_data[numerical_cols].mean()))

# Split the training data into features and target
X_train = train_data.drop(['metastatic_diagnosis_period'], axis=1)
y_train = train_data['metastatic_diagnosis_period']

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model on the training data
y_train_pred = model.predict(X_train)
rmse_train = math.sqrt(mean_squared_error(y_train, y_train_pred))
print(f"Training RMSE: {rmse_train:.2f}")

# Perform cross-validation
cv_scores = cross_val_score(model, X_train, y_train, scoring='neg_root_mean_squared_error', cv=5)
rmse_cv = -cv_scores.mean()
print(f"Cross-validation RMSE: {rmse_cv:.2f}")

# Make predictions on the test set
X_test = test_data[X_train.columns]  # Use the same columns as the training data
y_pred = model.predict(X_test)

# Generate the submission file
submission = pd.DataFrame({'patient_id': test_data['patient_id'],
                           'metastatic_diagnosis_period': y_pred.astype(int)})
submission.to_csv('submission.csv', index=False)