import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from naivebayes import NaiveBayes
from decisiontree import DecisionTree

def load_mushroom_data():
    # 1. Data Load
    df = pd.read_csv('mushroom/agaricus-lepiota.data', header=None, names=[
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
    X = df.iloc[:, 1:].copy()
    Y = df.iloc[:, 0].copy()

    # 3. Dealing with missing values
    X["stalk-root"] = X["stalk-root"].replace("?", "missing")

    # 4. Ordinal Encoding
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)

    # 5. Label Encoding
    Y_encoded = Y.map({'e': 1, 'p': 0}).to_numpy()

    return X_encoded, Y_encoded

def train_and_predict(X, Y):
    # Split data
    X_train, X_temp, Y_train, Y_temp = train_test_split(
        X, Y, test_size=0.2, random_state=42, shuffle=True)
    X_dev, X_test, Y_dev, Y_test = train_test_split(
        X_temp, Y_temp, test_size=0.5, random_state=42, shuffle=True)

    # Train models
    model_NB = NaiveBayes()
    model_NB.fit(X_train, Y_train)
    model_DT = DecisionTree()
    model_DT.tree = model_DT.build_tree(X_train, Y_train)

    # Make predictions
    y_test_pred_NB = model_NB.predict(X_test)
    y_test_pred_DT = model_DT.predict(X_test)

    return Y_test, y_test_pred_NB, y_test_pred_DT

def main():
    # Load and process data
    X_mushroom, Y_mushroom = load_mushroom_data()

    # Get predictions
    Y_test_mushroom, y_test_pred_NB_mushroom, y_test_pred_DT_mushroom = train_and_predict(X_mushroom, Y_mushroom)

    # Create visualization
    plt.figure(figsize=(15, 6))

    # Mushroom Dataset - Naive Bayes
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(Y_test_mushroom, y_test_pred_NB_mushroom), 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Mushroom Dataset - Naive Bayes')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Mushroom Dataset - Decision Tree
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(Y_test_mushroom, y_test_pred_DT_mushroom), 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Mushroom Dataset - Decision Tree')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 