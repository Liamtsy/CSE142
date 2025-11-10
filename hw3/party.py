import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from naivebayes import NaiveBayes
from sklearn.naive_bayes import CategoricalNB
from decisiontree import DecisionTree
from sklearn.tree import DecisionTreeClassifier

"""
Number of Instances: 435 (267 democrats, 168 republicans)

Number of Attributes: 16 + class name = 17 (all Boolean valued)

1. Class Name: 2 (democrat, republican)
2. handicapped-infants: 2 (y,n)
3. water-project-cost-sharing: 2 (y,n)
4. adoption-of-the-budget-resolution: 2 (y,n)
5. physician-fee-freeze: 2 (y,n)
6. el-salvador-aid: 2 (y,n)
7. religious-groups-in-schools: 2 (y,n)
8. anti-satellite-test-ban: 2 (y,n)
9. aid-to-nicaraguan-contras: 2 (y,n)
10. mx-missile: 2 (y,n)
11. immigration: 2 (y,n)
12. synfuels-corporation-cutback: 2 (y,n)
13. education-spending: 2 (y,n)
14. superfund-right-to-sue: 2 (y,n)
15. crime: 2 (y,n)
16. duty-free-exports: 2 (y,n)
17. export-administration-act-south-africa: 2 (y,n)

Missing Attribute Values: Denoted by "?"

NOTE: It is important to recognize that "?" in this database does not mean that the value of the attribute is unknown.
It means simply, that the value is not "yea" or "nay" (see "Relevant Information" section above).

Class Distribution: (2 classes)
1. 45.2 percent are democrat
2. 54.8 percent are republican
"""

def main():
  # 1. Data Load
  df = pd.read_csv('congressional+voting+records/house-votes-84.data', header = None, names = [
    "party",  # class label: democrat or republican
    "handicapped-infants",
    "water-project-cost-sharing",
    "adoption-of-the-budget-resolution",
    "physician-fee-freeze",
    "el-salvador-aid",
    "religious-groups-in-schools",
    "anti-satellite-test-ban",
    "aid-to-nicaraguan-contras",
    "mx-missile",
    "immigration",
    "synfuels-corporation-cutback",
    "education-spending",
    "superfund-right-to-sue",
    "crime",
    "duty-free-exports",
    "export-administration-act-south-africa"
    ])

  # 2. Feature and Label split
  X = df.iloc[:, 1:].copy() # 16 boolean features
  Y = df.iloc[:, 0].copy() # 1 class label

  # 3. Dealing with missing values
  for feature in X.columns:
    X[feature] = X[feature].replace("?", "missing")

  # 4. Ordinal Encoding
  encoder = OrdinalEncoder()
  X_encoded = encoder.fit_transform(X) # OrdinalEncoder.fit_transform() returns numpy array, X_encoded = np.ndarray

  # 5. Label Encoding
  Y_encoded = Y.map({'republican' : 1, 'democrat' : 0}).to_numpy()

  # 6. First split: 80% train, 20% temp
  #    Second split: 10% dev, 10% test (split temp 50:50), and shuffle

  X_train, X_temp, Y_train, Y_temp = train_test_split(
    X_encoded, Y_encoded, test_size = 0.2, random_state = 42, shuffle = True) # Shuffle True or False
  
  X_dev, X_test, Y_dev, Y_test = train_test_split(
    X_temp, Y_temp, test_size = 0.5, random_state = 42, shuffle = True)

  # 7. Model training
  model_NB = NaiveBayes()
  model_NB.fit(X_train, Y_train)
  model_DT = DecisionTree()
  model_DT.tree = model_DT.build_tree(X_train, Y_train)

  # 8. Prediction
  y_dev_pred_NB = model_NB.predict(X_dev)
  y_test_pred_NB = model_NB.predict(X_test)
  y_dev_pred_DT = model_DT.predict(X_dev)
  y_test_pred_DT = model_DT.predict(X_test)

  # 9. Accuracy
  print("=== Accuracy ===")
  print("Dev Accuracy NB:", accuracy_score(Y_dev, y_dev_pred_NB))
  print("Test Accuracy NB:", accuracy_score(Y_test, y_test_pred_NB))
  print("Dev Accuracy DT:", accuracy_score(Y_dev, y_dev_pred_DT))
  print("Test Accuracy DT:", accuracy_score(Y_test, y_test_pred_DT))

  # 10. Confusion Matrix
  print("\n=== Confusion Matrix ===")
  print("Dev Confusion Matrix NB:\n", confusion_matrix(Y_dev, y_dev_pred_NB))
  print("Test Confusion Matrix NB:\n", confusion_matrix(Y_test, y_test_pred_NB))
  print("Dev Confusion Matrix DT:\n", confusion_matrix(Y_dev, y_dev_pred_DT))
  print("Test Confusion Matrix DT:\n", confusion_matrix(Y_test, y_test_pred_DT))

  # 11. Scikit-learn Categorical NB for comparison
  model_sklearnNB = CategoricalNB(alpha = 1.0) # smoothing a = 1
  model_sklearnNB.fit(X_train, Y_train)
  model_sklearnDT = DecisionTreeClassifier()
  model_sklearnDT.fit(X_train, Y_train)

  y_dev_pred_NB2 = model_sklearnNB.predict(X_dev)
  y_test_pred_NB2 = model_sklearnNB.predict(X_test)
  y_dev_pred_DT2 = model_sklearnDT.predict(X_dev)
  y_test_pred_DT2 = model_sklearnDT.predict(X_test)

  print("Dev Accuracy NB:", accuracy_score(Y_dev, y_dev_pred_NB2))
  print("Test Accuracy NB:", accuracy_score(Y_test, y_test_pred_NB2))
  print("Confusion Matrix NB:\n", confusion_matrix(Y_test, y_test_pred_NB2))
  print("Dev Accuracy DT:", accuracy_score(Y_dev, y_dev_pred_DT2))
  print("Test Accuracy DT:", accuracy_score(Y_test, y_test_pred_DT2))
  print("Confusion Matrix DT:\n", confusion_matrix(Y_test, y_test_pred_DT2))

if __name__ == "__main__":
    main()