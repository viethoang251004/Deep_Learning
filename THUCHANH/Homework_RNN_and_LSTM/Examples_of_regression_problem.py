import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression
from sklearn import metrics

# Load your dataset
df = pd.read_csv('AACG.csv')

# Assume that 'Close' is your target variable and 'Open', 'High', 'Low' are your features
features = df[['Open', 'High', 'Low']]
target = df['Close']

# Split the data into training and testing sets
features_train, features_test, target_train, target_test = train_test_split(features, target, test_size=0.2, random_state=0)

# Create and train the model
model = LinearRegression()
model.fit(features_train, target_train)

# Use the model to make predictions
target_pred = model.predict(features_test)

# You can also measure the accuracy of your model
print('Mean Absolute Error:', metrics.mean_absolute_error(target_test, target_pred))
