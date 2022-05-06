import pandas as pd
import csv
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('./breast_cancer_updated.csv')

print(data.columns)

# print(data.head())

data.drop('IDNumber', axis=1, inplace=True)

print(data.columns)

print(data.isnull())
data = data.dropna()
data.reset_index(drop=True)
print('printing nulls after dropping')
print(data.isnull())
