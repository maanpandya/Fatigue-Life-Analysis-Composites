import pandas as pd
from sklearn.model_selection import train_test_split

# Read the training data from NNData.csv
data = pd.read_csv('NNData.csv')

# Split the data into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2)

# Save the testing data to NNTestData.csv
train_data.to_csv('NNTrainingData.csv', index=False)
test_data.to_csv('NNTestData.csv', index=False)