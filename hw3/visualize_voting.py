import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from naivebayes import NaiveBayes
from decisiontree import DecisionTree

def load_voting_data():
    # 1. Data Load
    df = pd.read_csv('congressional+voting+records/house-votes-84.data', header=None, names=[
        "party",
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
    X = df.iloc[:, 1:].copy()
    Y = df.iloc[:, 0].copy()

    # 3. Dealing with missing values
    for feature in X.columns:
        X[feature] = X[feature].replace("?", "missing")

    # 4. Ordinal Encoding
    encoder = OrdinalEncoder()
    X_encoded = encoder.fit_transform(X)

    # 5. Label Encoding
    Y_encoded = Y.map({'republican': 1, 'democrat': 0}).to_numpy()

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
    X_voting, Y_voting = load_voting_data()

    # Get predictions
    Y_test_voting, y_test_pred_NB_voting, y_test_pred_DT_voting = train_and_predict(X_voting, Y_voting)

    # Create visualization
    plt.figure(figsize=(15, 6))

    # Voting Dataset - Naive Bayes
    plt.subplot(1, 2, 1)
    sns.heatmap(confusion_matrix(Y_test_voting, y_test_pred_NB_voting), 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Voting Dataset - Naive Bayes')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    # Voting Dataset - Decision Tree
    plt.subplot(1, 2, 2)
    sns.heatmap(confusion_matrix(Y_test_voting, y_test_pred_DT_voting), 
                annot=True, fmt='d', cmap='Blues')
    plt.title('Voting Dataset - Decision Tree')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main() 