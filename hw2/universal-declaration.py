from pathlib import Path
import numpy as np
import string, re, random
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import ParameterGrid
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from spambase import evaluate_model
from perceptron import Perceptron

FOLDER = Path("universal-declaration")
RANDOM_SEED = 42                             # Random seed for reproducibility

BILETTERS = ["th", "ch", "sh", "ij", "ee", "oo"]   # Bigrams
VOWEL_RE  = re.compile(r"[aeiou]{2,}", re.I)       # Repetitive vowels
ENG_STOP  = {"the", "of"}                          # English common articles
NLD_STOP  = {"de", "het"}                          # Dutch common articles

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# Vectorize the text into a 26-dimensional vector
def vec_letters26(text: str) -> np.ndarray: # text should be a string and return a numpy array.
    v = np.zeros(26)                      # Creating a 26-dimensional zero vector
    for ch in text.lower():               # Iterate through the lowercase characters
        if ch in string.ascii_lowercase:  # if the character is a lowercase letter a to z
            v[ord(ch) - 97] += 1          # Add 1 to the index of the character a = 0, b = 1, ..., z = 25
    return v

def vec_bigrams(text: str) -> np.ndarray:
    t = text.lower()
    return np.array([t.count(bg) for bg in BILETTERS]) # Count the number of bigrams in the text and return a numpy array.
    # Ex) "the" -> "th" so [1, 0, 0, 0, 0, 0]

def vectorize(text: str) -> np.ndarray:
    text_l = text.lower()
    letters = vec_letters26(text_l)       # 26-dimensional vector
    bigr    = vec_bigrams(text_l)         # 6-dimensional vector
    vowel_r = np.array([len(VOWEL_RE.findall(text_l))])  # 1-dimensional vector
    words   = set(text_l.split())
    stop    = np.array([                 # 2-dimensional vector
        int(bool(words & ENG_STOP)),     # any {the,of} present?
        int(bool(words & NLD_STOP))      # any {de,het} present?
    ])
    return np.hstack([letters, bigr, vowel_r, stop])     # 35-dimensional vector, concatenate the vectors horizontally

def load_and_split(lang: str, label: int):
    """
    lang   : "english" or "dutch"
    label  : English = 1, Dutch = -1
    Returns   : ((X_tr, y_tr), (X_dev, y_dev), (X_te, y_te))
    """
    # Load the training data
    train_path = FOLDER / f"{lang}.txt"
    with train_path.open(encoding="utf-8") as f:
        train_sentences = [ln.strip() for ln in f if len(ln.strip()) > 2]
    
    # Load the development data
    dev_path = FOLDER / f"{lang}_dev.txt"
    with dev_path.open(encoding="utf-8") as f:
        dev_sentences = [ln.strip() for ln in f if len(ln.strip()) > 2]
    
    # Load the test data
    test_path = FOLDER / f"{lang}_test.txt"
    with test_path.open(encoding="utf-8") as f:
        test_sentences = [ln.strip() for ln in f if len(ln.strip()) > 2]
    
    def to_pair(lst):
        X = [vectorize(s) for s in lst]             # vectorize the sentences
        y = [label]*len(lst) # label is 1(English) or -1(Dutch)
        return X, y
    
    return to_pair(train_sentences), to_pair(dev_sentences), to_pair(test_sentences)

def evaluate_model(name, model, X, y):
    pred = model.predict(X)
    acc  = accuracy_score(y, pred)
    cm   = confusion_matrix(y, pred)
    print(f"\n[{name}]  accuracy={acc:.4f}\n{cm}")
    return acc

def main():
    # 1. split each language
    (train_E_X, train_E_y), (dev_E_X, dev_E_y), (test_E_X, test_E_y) = load_and_split("english", 1)
    (train_D_X, train_D_y), (dev_D_X, dev_D_y), (test_D_X, test_D_y) = load_and_split("dutch", 0)

    # 2. merge splits from both languages
    X_train = np.vstack([train_E_X, train_D_X])
    y_train = np.hstack([train_E_y, train_D_y])
    X_dev   = np.vstack([dev_E_X, dev_D_X])
    y_dev   = np.hstack([dev_E_y, dev_D_y])
    X_test  = np.vstack([test_E_X, test_D_X])
    y_test  = np.hstack([test_E_y, test_D_y])

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_dev_scaled = scaler.transform(X_dev)
    X_test_scaled = scaler.transform(X_test)
    X_90_scaled = np.vstack([X_train_scaled, X_dev_scaled])
    y_90    = np.hstack([y_train, y_dev])

    # 3‑A Perceptron
    p_grid = {"max_iter": [10, 30, 50, 100, 200], "eta": [1.0, 0.1, 0.01, 0.001], "shuffle": [True, False]}
    best_cfg, best_acc = None, -1
    for cfg in ParameterGrid(p_grid):
        m = Perceptron(**cfg, random_state=RANDOM_SEED).fit(X_train_scaled, y_train)
        acc = accuracy_score(y_dev, m.predict(X_dev_scaled))
        if acc > best_acc:
            best_cfg, best_acc = cfg, acc
    perceptron_best = Perceptron(**best_cfg, random_state=RANDOM_SEED).fit(X_90_scaled, y_90)
    evaluate_model(f"Perceptron {best_cfg}", perceptron_best, X_test_scaled, y_test)
    perceptron_best.print_weights_bias()

    # 3‑B Logistic Regression
    log_grid = {"penalty": ["l1", "l2"], "C": [0.001,0.01, 0.1, 1, 10, 100],
                "solver": ["liblinear", "saga"]}
    best_cfg, best_acc = None, -1
    for cfg in ParameterGrid(log_grid):
        log = LogisticRegression(max_iter=500, random_state=RANDOM_SEED, **cfg).fit(X_train_scaled, y_train)
        acc = accuracy_score(y_dev, log.predict(X_dev_scaled))
        if acc > best_acc:
            best_cfg, best_acc = cfg, acc
    log_best = LogisticRegression(max_iter=500, random_state=RANDOM_SEED, **best_cfg).fit(X_90_scaled, y_90)
    evaluate_model(f"LogReg {best_cfg}", log_best, X_test_scaled, y_test)

    # 3‑C Linear SVM
    svm_grid = {"C": [0.01, 0.1, 1, 10, 100, 1000], "loss": ["hinge", "squared_hinge"],
                "class_weight": [None, "balanced"]}
    
    best_cfg, best_acc = None, -1
    for cfg in ParameterGrid(svm_grid):
        svm = LinearSVC(max_iter=20000, random_state=RANDOM_SEED, **cfg).fit(X_train_scaled, y_train)
        acc = accuracy_score(y_dev, svm.predict(X_dev_scaled))
        if acc > best_acc:
            best_cfg, best_acc = cfg, acc
    svm_best = LinearSVC(max_iter=20000, random_state=RANDOM_SEED, **best_cfg).fit(X_90_scaled, y_90)
    evaluate_model(f"SVM {best_cfg}", svm_best, X_test_scaled, y_test)

if __name__ == "__main__":
    main()