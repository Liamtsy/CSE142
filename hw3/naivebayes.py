import pandas as pd
import numpy as np


"""
Two binary classification
1) if the mushroom is edible or not
2) Determine the party(democrat or republican) of a specific congressional representative
Don't use any external libraries for this other than numpy and pandas for the classifiers
For visualization, feel free to use other libaries
All of the features are categorical although there are missing values.
We can treat a missing value as its own distinct category if I like.
"""

"""
For Naive Bayes, 단순한 확률 분포를 맞추는 방식
예시로, 이전 과제인 스팸 베이스를 들어보자.
특정 클래스(예: 스팸)에서 어떤 값(예: 단어 'offer')이 얼마나 자주 나왔는지를 세고(count)
전체에서 얼마나 나왔는지를 기준으로 나누어서(devide) 확률을 추정
즉, P(단어 = "offer" | 스팸) = 스팸에서 "offer" 등장 빈도 / 전체 스팸 이메일 수
모든 단어에 대해 이런 확률들을 곱해서 가장 확률이 높은 클래스(예: 스팸 or 스팸 아님)를 고름
"""

"""
Experiment: 80% training set | 10% dev set | 10% test set
1) Does it matter whether I shuffle my training data before fitting my models?
2) Would it help if I do any smoothing for the probability distributions?
Accuracy 와 Confusion Matrix 결과로 뽑아야함
"""

"""
당신이 구현한 분류기의 성능은 scikit-learn 라이브러리의 구현체들과 비교해서 어떠한가요?
sklearn.naive_bayes.CategoricalNB과 같은 데이터셋으로 비교, 정확도나 F1 Score같은 지표로 비교
"""
class NaiveBayes:
  def __init__(self):
    self.class_counts = None # The number of samples per class [count_0, count_1]
    self.class_priors = None  # The prior probability per class [P(0), P(1)]
    self.feature_value_counts = None # It's a table structure, dictionary
    self.num_feature_values = None # The number of unique values in each feature

  def fit(self, X_train, Y_train):
    # 1. Counting the number of samples per class
    class_counts = np.bincount(Y_train) # returns an array like [n, m]
    total_samples = len(Y_train) # The number of the total samples

    class_priors = class_counts / total_samples # it's array, we can divide elements in array with a single scaler thanks to NumPy.

    self.class_counts = class_counts
    self.class_priors = class_priors

    # 2. Building a count table for the conditional probability
    num_classes = len(np.unique(Y_train)) # returns a sorted array containing unique values in the previous array 
    num_features = X_train.shape[1]
    
    self.feature_value_counts = {
      c: [{} for _ in range(num_features)]
      for c in range(num_classes)
    }

    for i in range(len(X_train)):
      x = X_train[i]
      y = Y_train[i]

      for j in range(num_features):
        value = x[j]

        if value not in self.feature_value_counts[y][j]:
          self.feature_value_counts[y][j][value] = 0

        self.feature_value_counts[y][j][value] += 1

    # 3. Storing the number of unique values that each feature can have
    self.num_feature_values = []
    for j in range(num_features):
      unique_vals = np.unique(X_train[:, j])
      self.num_feature_values.append(len(unique_vals))

  def predict(self, X):
    predictions = []

    for sample in X: 
      log_probs_per_class = []

      for class_label in range(len(self.class_priors)): # 0 ~ 1
        log_class_prob = np.log(self.class_priors[class_label])

        feature_value_counts_for_class = self.feature_value_counts[class_label]
        total_samples_for_class = self.class_counts[class_label]

        for feature_index in range(len(sample)): # len(sample) = 22 in mushroom dataset, 0 ~ 21
          feature_val = sample[feature_index]
          value_count_dict = feature_value_counts_for_class[feature_index] # This is a dictionary per feature in a class

          # How many times did feature_val appear in this class?
          feature_val_count = value_count_dict.get(feature_val, 0)

          # Applying smoothing and the number of unique values (K)
          num_unique_vals = self.num_feature_values[feature_index]
          smoothed_conditional_prob = (feature_val_count + 1) / (total_samples_for_class + (1 * num_unique_vals))

          # Accumulating log probs
          log_class_prob += np.log(smoothed_conditional_prob)

        log_probs_per_class.append(log_class_prob)

      predicted_class = np.argmax(log_probs_per_class) # np.argmax returns a index of the biggest value in the array
      predictions.append(predicted_class)

    return np.array(predictions)



