# CSE142 Homework 3: Machine Learning Classification

This repository contains two classification projects:

1. Mushroom Classification
2. Congressional Voting Records Classification

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

## Project 1: Mushroom Classification

This project implements and compares different classifiers for mushroom edibility prediction using the Mushroom dataset.

### Running the Code

Run the main script:

```bash
python mushroom.py
```

The script will:

- Load and preprocess the Mushroom dataset
- Split the data into training, development, and test sets
- Train and evaluate two classifiers:
  - Naive Bayes
  - Decision Tree
- Print performance metrics and confusion matrices for each model
- Compare with scikit-learn implementations

## Project 2: Congressional Voting Records Classification

This project implements classifiers to predict the party affiliation (Democrat or Republican) of congressional representatives based on their voting records.

### Running the Code

Run the main script:

```bash
python party.py
```

The script will:

- Load and preprocess the Congressional Voting Records dataset
- Split the data into training, development, and test sets
- Train and evaluate two classifiers:
  - Naive Bayes
  - Decision Tree
- Print performance metrics and confusion matrices for each model
- Compare with scikit-learn implementations

## Visualization

To visualize the confusion matrices for both datasets:

For Mushroom dataset:

```bash
python visualize_mushroom.py
```

For Voting Records dataset:

```bash
python visualize_voting.py
```

These will generate visualizations containing:

- 1x2 grid of confusion matrices for each dataset
- Left: Naive Bayes results
- Right: Decision Tree results

## Project Structure

```
hw3/
├── mushroom/
│   ├── agaricus-lepiota.data
│   └── agaricus-lepiota.names
├── congressional+voting+records/
│   └── house-votes-84.data
├── mushroom.py
├── party.py
├── naivebayes.py
├── decisiontree.py
├── visualize_mushroom.py
├── visualize_voting.py
└── README.md
```

## Notes

- Both projects use the same Naive Bayes and Decision Tree implementations
- The code includes smoothing for Naive Bayes classifier
- Results are printed to the console, including accuracy scores and confusion matrices
- The random seed is set to 42 for reproducibility
- Visualization scripts require seaborn package for better-looking plots
- Each classifier is compared with its scikit-learn implementation
