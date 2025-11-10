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
For decision tree, we're gonna use "information gain".
"어떤 특성을 먼저 나누면 데이터를 잘 분리할 수 있을까?"
"특정 feature로 데이터를 나눴을 때, 클래스가 얼마나 잘 분리되는가?"
어떤 특성(예: "색상")으로 데이터를 분할 했을 때, 그 결과가 얼마나 깔끔하게(class가 섞이지 않게) 나뉘는지를 수치화한 것
계산 흐름:
1) 먼저 전체 데이터의 엔트로피(혼잡도)를 계산 -> 클래스가 섞여 있을수록 높고, 하나로 몰려 있으면 낮음
2) 어떤 특성(예: "색상")으로 분할한 후, 각 분할에서의 엔트로피를 구하고, 평균을 냄
3) 분할 전과 분할 후 엔트로피의 차이 = 정보 이득(information gain)
-> 정보 이득이 클수록 좋은 분할
클래스가 최대한 한쪽으로 몰리게 (섞이지 않게) 하는 분할이 목적
클래스를 분류하는데에 가장 큰 영향을 미치는 기준(어떤 특성)을 찾는게 목적! 트리에서 가장 큰 정보 이득을 주는 feature를 선택, 그 기준으로 분기
반복적으로 이 과정을 하면서 트리가 점점 뻗어나가고, 각 노드에서 데이터가 최대한 pure하게(명확하게 나눠지게) 나눠지도록 유도
"""

"""
Experiment: 80% training set | 10% dev set | 10% test set
For Decision Trees, you might want to try out some different
hyperparameters, in the form of different stopping criteria for recursively splitting
your data - you can try different stopping criteria by testing your classifier on
the development set.
Accuracy 와 Confusion Matrix 결과로 뽑아야함
"""

"""
당신이 구현한 분류기의 성능은 scikit-learn 라이브러리의 구현체들과 비교해서 어떠한가요?
sklearn.tree.DecisionTreeClassifier과 같은 데이터셋으로 비교, 정확도나 F1 Score같은 지표로 비교
"""

class DecisionTree:
  def __init__(self):
    pass

  def entropy(self, y):
    # 1. Counting the number of samples per class
    class_counts = np.bincount(y) # returns an array like [n, m]
    class_distribution = class_counts / np.sum(class_counts)

    entropy = 0.0
    for p in class_distribution:
      if p > 0:
        entropy -= p * np.log2(p)
    return entropy
  
  def information_gain(self, X_column, y):
    parent_entropy = self.entropy(y)
    unique_values = np.unique(X_column)
    weighted_child_entropy = 0.0

    for value in unique_values:
      indices = (X_column == value) # Boolean Mask Operation, returns like [True, False, True, False, ...]
      y_subset = y[indices]
      ratio = len(y_subset) / len(y)
      child_entropy = self.entropy(y_subset)
      weighted_child_entropy += ratio * child_entropy

    info_gain = parent_entropy - weighted_child_entropy
    return info_gain
  
  def best_split(self, X, y):
    best_info_gain = -1
    best_feature_index = -1

    num_features = X.shape[1]

    for i in range(num_features):
      X_column = X[:, i] # feature i
      info_gain = self.information_gain(X_column, y)

      if info_gain > best_info_gain:
        best_info_gain = info_gain
        best_feature_index = i

    return best_feature_index
  
  def build_tree(self, X, y):
    # 1. 만약 y가 전부 같은 클래스라면 → 리프 노드
    if np.all(y == y[0]): # np.all returns True if the condition for all elements is true
        return y[0]  # 예: 0 or 1

    # 2. 만약 feature가 더 이상 없으면 → 다수 클래스 반환
    if X.shape[1] == 0:
        return self.majority_class(y)

    # 3. 최적 분할 feature 선택
    best_feature = self.best_split(X, y)

    # 4. 트리 노드 생성
    tree = {"feature_index": best_feature, "branches": {}}

    # 5. 해당 feature의 고유값별로 분할
    values = np.unique(X[:, best_feature])
    for value in values:
        # 현재 값에 해당하는 샘플만 추출
        indices = (X[:, best_feature] == value)
        X_subset = X[indices]
        y_subset = y[indices]

        # 현재 feature 제거하고 넘김 (split에 쓴 feature는 버림)
        X_subset = np.delete(X_subset, best_feature, axis=1)

        # 재귀 호출로 자식 노드 구성
        subtree = self.build_tree(X_subset, y_subset)

        # 트리에 가지 추가
        tree["branches"][value] = subtree

    return tree
  
  def predict(self, X):
    return np.array([self._predict_sample(x, self.tree) for x in X])

  def _predict_sample(self, x, node):
    if not isinstance(node, dict):
      return node
    
    feature_index = node["feature_index"]
    feature_value = x[feature_index]

    if feature_value not in node["branches"]:
      return 0

    subtree = node["branches"][feature_value]

    return self._predict_sample(x, subtree)