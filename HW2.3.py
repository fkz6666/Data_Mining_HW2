import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
column_names = ['ID', 'Diagnosis'] + [f'Feature_{i}' for i in range(1, 31)]
data = pd.read_csv(url, names=column_names)
X = data.drop(['ID', 'Diagnosis'], axis=1)
y = data['Diagnosis'].map({'M': 1, 'B': 0})
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)
pca_1 = PCA(n_components=1)
X_train_pca_1 = pca_1.fit_transform(X_train)
X_test_pca_1 = pca_1.transform(X_test)
pca_2 = PCA(n_components=2)
X_train_pca_2 = pca_2.fit_transform(X_train)
X_test_pca_2 = pca_2.transform(X_test)


def build_and_evaluate_tree(X_train, X_test, y_train, y_test):
    clf = DecisionTreeClassifier(min_samples_split=2, min_samples_leaf=5, max_depth=2, criterion='gini')
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    fpr = fp / (fp + tn)
    tpr = tp / (tp + fn)
    return f1, precision, recall, fp, tp, fpr, tpr


f1_original, precision_original, recall_original, fp_original, tp_original, fpr_original, tpr_original = \
    build_and_evaluate_tree(X_train, X_test, y_train, y_test)

f1_pca_1, precision_pca_1, recall_pca_1, fp_pca_1, tp_pca_1, fpr_pca_1, tpr_pca_1 = \
    build_and_evaluate_tree(X_train_pca_1, X_test_pca_1, y_train, y_test)

f1_pca_2, precision_pca_2, recall_pca_2, fp_pca_2, tp_pca_2, fpr_pca_2, tpr_pca_2 = \
    build_and_evaluate_tree(X_train_pca_2, X_test_pca_2, y_train, y_test)

# output
print("Original Data:")
print(f"F1 Score: {f1_original}, Precision: {precision_original}, Recall: {recall_original}")
print(f"FP: {fp_original}, TP: {tp_original}, FPR: {fpr_original}, TPR: {tpr_original}\n")

print("PCA (1 Component):")
print(f"F1 Score: {f1_pca_1}, Precision: {precision_pca_1}, Recall: {recall_pca_1}")
print(f"FP: {fp_pca_1}, TP: {tp_pca_1}, FPR: {fpr_pca_1}, TPR: {tpr_pca_1}\n")

print("PCA (2 Components):")
print(f"F1 Score: {f1_pca_2}, Precision: {precision_pca_2}, Recall: {recall_pca_2}")
print(f"FP: {fp_pca_2}, TP: {tp_pca_2}, FPR: {fpr_pca_2}, TPR: {tpr_pca_2}\n")

