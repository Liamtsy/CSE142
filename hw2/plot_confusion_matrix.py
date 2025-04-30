import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Spambase results
spambase_perceptron = np.array([[242, 37],
                               [25, 157]])

spambase_logreg = np.array([[260, 19],
                           [20, 162]])

spambase_svm = np.array([[261, 18],
                        [22, 160]])

# Universal Declaration results
ud_perceptron = np.array([[19, 1],
                         [0, 20]])

ud_logreg = np.array([[18, 2],
                     [0, 20]])

ud_svm = np.array([[18, 2],
                  [0, 20]])

def plot_confusion_matrices():
    # Set up the figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confusion Matrices for Spambase and Universal Declaration Datasets', fontsize=16)

    # Plot Spambase results
    spambase_results = [
        (spambase_perceptron, "Perceptron"),
        (spambase_logreg, "Logistic Regression"),
        (spambase_svm, "SVM")
    ]

    for idx, (cm, name) in enumerate(spambase_results):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, idx])
        axes[0, idx].set_title(f'Spambase - {name}')
        axes[0, idx].set_xlabel('Predicted')
        axes[0, idx].set_ylabel('Actual')
        axes[0, idx].set_xticklabels(['Not Spam', 'Spam'])
        axes[0, idx].set_yticklabels(['Not Spam', 'Spam'])

    # Plot Universal Declaration results
    ud_results = [
        (ud_perceptron, "Perceptron"),
        (ud_logreg, "Logistic Regression"),
        (ud_svm, "SVM")
    ]

    for idx, (cm, name) in enumerate(ud_results):
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, idx])
        axes[1, idx].set_title(f'Universal Declaration - {name}')
        axes[1, idx].set_xlabel('Predicted')
        axes[1, idx].set_ylabel('Actual')
        axes[1, idx].set_xticklabels(['Dutch', 'English'])
        axes[1, idx].set_yticklabels(['Dutch', 'English'])

    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    plt.close()

if __name__ == "__main__":
    plot_confusion_matrices()
    print("Confusion matrices have been saved to 'confusion_matrices.png'") 