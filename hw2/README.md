# CSE142 Homework 2: Machine Learning Classification

This repository contains two classification projects:

1. Spambase Email Classification
2. Universal Declaration Language Classification

## Environment Setup

1. Create a Python virtual environment:

```bash
python3 -m venv venv
```

2. Activate the virtual environment:

- On macOS/Linux:

```bash
source venv/bin/activate
```

- On Windows:

```bash
.\venv\Scripts\activate
```

3. Install required packages:

```bash
pip install numpy pandas scikit-learn matplotlib seaborn
```

## Project 1: Spambase Email Classification

This project implements and compares different classifiers for email spam detection using the Spambase dataset.

### Running the Code

Run the main script:

```bash
python spambase.py
```

The script will:

- Load and preprocess the Spambase dataset
- Split the data into training, development, and test sets
- Train and evaluate three classifiers:
  - Perceptron
  - Logistic Regression
  - Linear SVM
- Print performance metrics and confusion matrices for each model
- For Perceptron, print weight values and bias as well

## Project 2: Universal Declaration Language Classification

This project implements classifiers to distinguish between English and Dutch versions of the Universal Declaration of Human Rights.

### Running the Code

Run the main script:

```bash
python universal-declaration.py
```

The script will:

- Load and preprocess the English and Dutch text data
- Split the data into training, development, and test sets
- Train and evaluate three classifiers:
  - Perceptron
  - Logistic Regression
  - Linear SVM
- Print performance metrics and confusion matrices for each model
- For Perceptron, print weight values and bias as well

## Visualization

To visualize the confusion matrices for both datasets:

```bash
python plot_confusion_matrix.py
```

This will generate a 'confusion_matrices.png' file containing:

- 2x3 grid of confusion matrices
- Top row: Spambase dataset results
- Bottom row: Universal Declaration dataset results
- Each column represents a different model (Perceptron, Logistic Regression, SVM)

## Project Structure

```
hw2/
├── spambase/
│   ├── spambase.data
│   ├── spambase.DOCUMENTATION
│   ├── spambase.names
├── universal-declaration/
│   ├── english.txt
│   ├── english_dev.txt
│   ├── english_test.txt
│   ├── dutch.txt
│   ├── dutch_dev.txt
│   └── dutch_test.txt
├── spambase.py
├── universal-declaration.py
├── perceptron.py
├── plot_confusion_matrix.py
├── confusion_matrices.png
└── README.md
```

## Notes

- Both projects use the same Perceptron implementation (`perceptron.py`)
- The code includes hyperparameter tuning for each classifier
- Results are printed to the console, including accuracy scores and confusion matrices
- The random seed is set to 42 for reproducibility
- Visualization script requires seaborn package for better-looking plots
