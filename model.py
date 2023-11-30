import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

df = pd.read_csv('physical_data.csv')

# Display the first few rows of the dataset
print(df.head())
# Check for missing values
print(df.isnull().sum())
# Check the distribution of the target variable
print(df['status'].value_counts())


# Separate features (X) and target variable (y)
X = df.drop('status', axis=1)
y = df['status']
# Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# Example with adjusted parameters
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)


# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
# Display the classification report
print('Classification Report:')
print(classification_report(y_test, y_pred))
# Display the confusion matrix
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))



# Assuming you have already trained the model (steps provided in the previous responses)
import random 
# Create a new DataFrame with random data for testing (ensuring consistent feature order)
new_data = pd.DataFrame({
    'ID': [1], # Add 'ID' if it is required for the model to work
    'Steps': [random.randint(1000, 20000)],
    'HeartRate': [random.randint(40, 150)],
    'Sleep Duration': [round(random.uniform(2, 14), 2)],
    'Stress': [random.choice([1, 2, 3])], 
})
# Make predictions on the new data using the trained model
single_prediction = model.predict(new_data)
# Extract the single predicted value
single_value = single_prediction[0]
# Display the new data and the single predicted value
print('Data for Testing: ')
print(new_data)
print('\nStatus: ', single_value)


#make pickle file for our model
pickle.dump(model, open('model.pkl', 'wb'))
