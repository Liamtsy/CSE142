# CSE142 Homework 4: Machine Learning Clustering

This repository contains clustering projects implementing K-Means and Gaussian Mixture Models (GMM) algorithms.

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

## Project Structure

```
hw4/
├── hw4.py              # Main script for running clustering algorithms
├── kmeans.py           # K-Means clustering implementation
├── gmm.py             # Gaussian Mixture Model implementation
├── kmeans_experiment.py # Script for K-Means experiments
├── gmm_experiment.py   # Script for GMM experiments
├── compare_with_sklearn.py # Script to compare with scikit-learn implementations
├── hw4_dataset.csv    # Dataset for clustering
└── README.md
```

## Running the Code

### Main Clustering Script

Run the main script to perform clustering on the dataset:

```bash
python hw4.py
```

This will:

- Load and preprocess the dataset
- Perform K-Means clustering
- Visualize the clustering results
- Show centroids and cluster assignments

### Experiments

To run K-Means experiments with different parameters:

```bash
python kmeans_experiment.py
```

To run GMM experiments:

```bash
python gmm_experiment.py
```

### Comparison with scikit-learn

To compare our implementations with scikit-learn:

```bash
python compare_with_sklearn.py
```

## Features

- K-Means Clustering implementation with:

  - Random centroid initialization
  - Euclidean distance-based assignment
  - Iterative centroid updates
  - Convergence checking
  - Visualization capabilities

- Gaussian Mixture Model implementation with:
  - EM algorithm
  - Covariance matrix handling
  - Probability-based clustering
  - Visualization capabilities

## Notes

- The code includes proper error handling and convergence checks
- Results are visualized using matplotlib
- The random seed is set for reproducibility
- Each implementation is compared with its scikit-learn counterpart
- The dataset contains weight and length measurements for clustering analysis
