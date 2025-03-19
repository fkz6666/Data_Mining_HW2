import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

url1 = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"
column_names = ['Sample code number', 'Clump Thickness',
                'Uniformity of Cell Size', 'Uniformity of Cell Shape',
                'Marginal Adhesion', 'Single Epithelial Cell Size',
                'Bare Nuclei', 'Bland Chromatin',
                'Normal Nucleoli', 'Mitoses', 'Class']
data = pd.read_csv(url1, names=column_names)
data = data.replace('?', np.nan)
data = data.dropna()
X = data.drop(['Sample code number', 'Class'], axis=1)
y = data['Class'].replace({2: 0, 4: 1})
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
clf = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=5, max_depth=2, criterion='gini')
clf.fit(X_train, y_train)
feature = clf.tree_.feature[0]
threshold = clf.tree_.threshold[0]
parent_indices = np.arange(len(X_train))
left_child_indices = X_train.iloc[:, feature] <= threshold
right_child_indices = X_train.iloc[:, feature] > threshold
parent_labels = y_train
left_child_labels = y_train[left_child_indices]
right_child_labels = y_train[right_child_indices]

# calculate entropy
def calculate_entropy(y):
    p = np.mean(y)
    if p == 0 or p == 1:
        return 0
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)

#calculate Gini index
def calculate_gini(y):
    p = np.mean(y)
    return 1 - p**2 - (1 - p)**2

# calculate misclassification error
def calculate_misclassification_error(y):
    p = np.mean(y)
    return 1 - max(p, 1 - p)

# Calculate metrics
parent_entropy = calculate_entropy(parent_labels)
parent_gini = calculate_gini(parent_labels)
parent_error = calculate_misclassification_error(parent_labels)
left_entropy = calculate_entropy(left_child_labels)
right_entropy = calculate_entropy(right_child_labels)

# information gain
n_parent = len(parent_labels)
n_left = len(left_child_labels)
n_right = len(right_child_labels)
information_gain = parent_entropy - (n_left / n_parent * left_entropy + n_right / n_parent * right_entropy)
# output
print(f"Entropy of the parent node: {parent_entropy}")
print(f"Gini index of the parent node: {parent_gini}")
print(f"Misclassification error of the parent node: {parent_error}")
print(f"Information gain: {information_gain}")
print(f"Feature selected for the first split: {X.columns[feature]}")
print(f"Decision boundary value: {threshold}")