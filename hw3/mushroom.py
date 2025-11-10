import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from naivebayes import NaiveBayes
from sklearn.naive_bayes import CategoricalNB
from decisiontree import DecisionTree
from sklearn.tree import DecisionTreeClassifier

def main():
  # 1. Data Load
  df = pd.read_csv('mushroom/agaricus-lepiota.data', header = None, names = [
      "label",
      "cap-shape",
      "cap-surface",
      "cap-color",
      "bruises?",
      "odor",
      "gill-attachment",
      "gill-spacing",
      "gill-size",
      "gill-color",
      "stalk-shape",
      "stalk-root",
      "stalk-surface-above-ring",
      "stalk-surface-below-ring",
      "stalk-color-above-ring",
      "stalk-color-below-ring",
      "veil-type",
      "veil-color",
      "ring-number",
      "ring-type",
      "spore-print-color",
      "population",
      "habitat"
  ])

  # 2. Feature and Label split
  X = df.iloc[:, 1:].copy() # 22 categorical features
  Y = df.iloc[:, 0].copy() # 1 class label, e = edible & p = poisonous

  # 3. Dealing with missing values
  X["stalk-root"] = X["stalk-root"].replace("?", "missing")

  # 4. Ordinal Encoding
  encoder = OrdinalEncoder()
  X_encoded = encoder.fit_transform(X) # OrdinalEncoder.fit_transform() returns numpy array, X_encoded = np.ndarray

  # 5. Label Encoding
  Y_encoded = Y.map({'e' : 1, 'p' : 0}).to_numpy() # edible = 1, poisonous = 0

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