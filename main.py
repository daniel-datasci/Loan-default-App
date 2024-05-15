# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# Load the data
data_path = 'Loan_default.csv'
df = pd.read_csv(data_path)

# Display the first few rows of the dataset to understand its structure
st.title("A Web Application For Loan Default Prediction")
st.write("""
         In the financial landscape, the ability to accurately predict loan defaults is of paramount importance. 
         This predictive capability not only mitigates risk for lenders but also aids in making informed credit decisions. 
         To address this critical need, this project embarks on the development of a loan default prediction application. 
         This application leverages machine learning techniques to analyze historical loan data and forecast the likelihood of future defaults.
         """)

# Data preprocessing
# Drop unnecessary columns
df.drop(['LoanID', 'Education', 'DTIRatio'], axis=1, inplace=True)

# Handle class imbalance
# Separate majority and minority classes
df_majority = df[df.Default == 0]
df_minority = df[df.Default == 1]

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                 replace=True,     # sample with replacement
                                 n_samples=len(df_majority),    # to match majority class
                                 random_state=42) # reproducible results

# Combine majority class with upsampled minority class
df_balanced = pd.concat([df_majority, df_minority_upsampled])

# Encode categorical variables if any
label_encoders = {}
for column in df_balanced.select_dtypes(include=['object']).columns:
    label_encoders[column] = LabelEncoder()
    df_balanced[column] = label_encoders[column].fit_transform(df_balanced[column])

# Split the data into features and target variable
X = df_balanced.drop('Default', axis=1)
y = df_balanced['Default']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build a Random Forest classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit app for predictions
st.title("Loan Default Prediction")

# Map original binary values back for user understanding
employment_type_map = {0: 'Unemployed', 1: 'Employed'}
marital_status_map = {0: 'Single', 1: 'Married'}
has_mortgage_map = {0: 'No', 1: 'Yes'}
has_dependents_map = {0: 'No', 1: 'Yes'}
loan_purpose_map = {0: 'Personal', 1: 'Business'}
has_cosigner_map = {0: 'No', 1: 'Yes'}

# Input fields for user to enter data
def user_input_features():
    inputs = {
        'LoanAmount': st.number_input('Enter Loan Amount', value=5000),
        'CreditScore': st.number_input('Enter Credit Score', value=650),
        'MonthsEmployed': st.number_input('Enter Months Employed', value=12),
        'NumCreditLines': st.number_input('Enter Number of Credit Lines', value=5),
        'InterestRate': st.number_input('Enter Interest Rate', value=10.0),
        'LoanTerm': st.number_input('Enter Loan Term', value=36),
        'Age': st.number_input('Enter Age', value=30),
        'Income': st.number_input('Enter Income', value=50000),
        'EmploymentType': st.selectbox('Select Employment Type', options=list(employment_type_map.values())),
        'MaritalStatus': st.selectbox('Select Marital Status', options=list(marital_status_map.values())),
        'HasMortgage': st.selectbox('Select Do you have Mortgage?', options=list(has_mortgage_map.values())),
        'HasDependents': st.selectbox('Select Do you have Dependents?', options=list(has_dependents_map.values())),
        'LoanPurpose': st.selectbox('Select What is the Loan Purpose?', options=list(loan_purpose_map.values())),
        'HasCoSigner': st.selectbox('Select Do you have a Cosigner?', options=list(has_cosigner_map.values())),
    }

    # Map back to original binary values
    inputs['EmploymentType'] = list(employment_type_map.keys())[list(employment_type_map.values()).index(inputs['EmploymentType'])]
    inputs['MaritalStatus'] = list(marital_status_map.keys())[list(marital_status_map.values()).index(inputs['MaritalStatus'])]
    inputs['HasMortgage'] = list(has_mortgage_map.keys())[list(has_mortgage_map.values()).index(inputs['HasMortgage'])]
    inputs['HasDependents'] = list(has_dependents_map.keys())[list(has_dependents_map.values()).index(inputs['HasDependents'])]
    inputs['LoanPurpose'] = list(loan_purpose_map.keys())[list(loan_purpose_map.values()).index(inputs['LoanPurpose'])]
    inputs['HasCoSigner'] = list(has_cosigner_map.keys())[list(has_cosigner_map.values()).index(inputs['HasCoSigner'])]

    return pd.DataFrame(inputs, index=[0])

input_df = user_input_features()

# Ensure the order of columns matches the training set
input_df = input_df[X_train.columns]

# Make prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

st.write("### Prediction")
st.write(f"The prediction is: {'Default' if prediction[0] else 'No Default'}")
st.write("### Prediction Probability")
st.write(f"Probability of Default: {prediction_proba[0][1]:.2f}")
st.write(f"### Model Accuracy: {accuracy:.2f}")

# Display classification report
st.write("### Classification Report")
st.write(classification_report(y_test, y_pred))

# Feature importance
feature_importances = pd.DataFrame(model.feature_importances_,
                                   index=X_train.columns,
                                   columns=['importance']).sort_values('importance', ascending=False)
st.write("### Feature Importances")
st.write(feature_importances)

# Visualizations using matplotlib
st.write("### Data Visualization")

# Correlation heatmap
st.write("#### Correlation Heatmap")
corr_matrix = df_balanced.corr()
plt.figure(figsize=(10, 8))
plt.imshow(corr_matrix, cmap='coolwarm', interpolation='none', aspect='auto')
plt.colorbar()
plt.xticks(range(len(corr_matrix)), corr_matrix.columns, rotation=90)
plt.yticks(range(len(corr_matrix)), corr_matrix.columns)
plt.title('Correlation Matrix')
st.pyplot(plt)

# Distribution of the target variable
st.write("#### Distribution of Default")
plt.figure(figsize=(6, 4))
plt.hist(df_balanced['Default'], bins=3, edgecolor='k', alpha=0.7)
plt.xlabel('Default')
plt.ylabel('Count')
plt.title('Distribution of Default')
st.pyplot(plt)

# Pairplot of important features
st.write("#### Pairplot of Important Features")
important_features = feature_importances.head(4).index.tolist()
pd.plotting.scatter_matrix(df_balanced[important_features + ['Default']], figsize=(12, 12), alpha=0.8, diagonal='kde')
plt.suptitle('Pairplot of Important Features', y=1.02)
st.pyplot(plt)
