import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from perceptron import Perceptron

def evaluate_model(name, model, X, y):
    pred = model.predict(X)
    acc  = accuracy_score(y, pred)
    cm   = confusion_matrix(y, pred)
    print(f"\n[{name}]  accuracy={acc:.4f}\n{cm}")
    return acc

def main():
    # Data Load
    df = pd.read_csv('spambase/spambase.data', header=None)

    # Feature and Label Split
    X = df.iloc[:, :-1].values  # 57 continuous features
    y = df.iloc[:, -1].values   # 1 class label, 0(not spam) or 1(spam)

    # Dataset Split (80:10:10)
    # Split 10% as test set
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

    # Split the temp set 9:1 to get dev(10%) and train(80%)
    X_train, X_dev, y_train, y_dev = train_test_split(X_temp, y_temp, test_size=1/9, random_state=42, stratify=y_temp)

    # Data Preprocessing
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train) # fit the scaler depeding on the train set
    X_dev_scaled = scaler.transform(X_dev) # Using the train set's scaler, the standard depends on the train set
    X_test_scaled = scaler.transform(X_test) # Same as above
    X_90_scaled = np.vstack((X_train_scaled, X_dev_scaled))
    y_90 = np.hstack((y_train, y_dev))

    # Search for the best hyperparameters of perceptron using the dev set
    perceptron_grid = {"max_iter":[5,10,20,40,80,160],
                "eta":[1.0,0.5,0.1,0.01,0.001],
                "shuffle":[True, False]}

    best_config_p, best_accuracy_p= None, -1
    for config in ParameterGrid(perceptron_grid):
        p = Perceptron(**config, random_state=42).fit(X_train_scaled, y_train)
        acc = accuracy_score(y_dev, p.predict(X_dev_scaled))
        if acc > best_accuracy_p:
            best_accuracy_p, best_config_p = acc, config

    perceptron_best = Perceptron(**best_config_p, random_state=42).fit(X_90_scaled, y_90)
    evaluate_model(f"Perceptron {best_config_p}", perceptron_best, X_test_scaled, y_test)
    perceptron_best.print_weights_bias()

    # Logistic Regression
    log_grid = {
      "penalty": ["l1", "l2"], # Regularization penalty
      "C": [0.001, 0.01, 0.1, 1, 10, 100], # Inverse of regularization strength, smaller = stronger regularization
      "solver": ["liblinear", "saga"]
    }

    best_config_l, best_accuracy_l = None, -1
    for config in ParameterGrid(log_grid):
      log = LogisticRegression(max_iter = 500, random_state=42, **config)
      log.fit(X_train_scaled, y_train)
      acc = accuracy_score(y_dev, log.predict(X_dev_scaled))
      if acc > best_accuracy_l:
        best_accuracy_l, best_config_l = acc, config

    log_best = LogisticRegression(max_iter = 500, random_state = 42, **best_config_l)
    log_best.fit(X_90_scaled, y_90)
    evaluate_model(f"Logistic Regression {best_config_l}", log_best, X_test_scaled, y_test)

    # Linear SVM
    svm_grid = {"C":[0.01, 0.1, 1, 10, 100, 1000],
                "loss":["hinge","squared_hinge"],
                "class_weight": [None, "balanced"]}

    best_config_s, best_accuracy_s = None, -1
    for config in ParameterGrid(svm_grid):
      svm = LinearSVC(max_iter = 10000, random_state = 42, **config)
      svm.fit(X_train_scaled, y_train)
      acc = accuracy_score(y_dev, svm.predict(X_dev_scaled))
      if acc > best_accuracy_s:
        best_accuracy_s, best_config_s = acc, config

    svm_best = LinearSVC(max_iter = 10000, random_state = 42, **best_config_s)
    svm_best.fit(X_90_scaled, y_90)
    evaluate_model(f"Linear SVM {best_config_s}", svm_best, X_test_scaled, y_test)

if __name__ == "__main__":
    main()
