from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, precision_score, f1_score
import pandas as pd

iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
X_train, X_test, y_train, y_test = train_test_split(df.iloc[:, :-1], df['target'], test_size=0.3, random_state=42)
results = []
for max_depth in range(1, 6):
    clf = DecisionTreeClassifier(
        max_depth=max_depth,
        min_samples_leaf=2,
        min_samples_split=5,
        random_state=42
    )

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    results.append({
        'max_depth': max_depth,
        'recall': recall,
        'precision': precision,
        'f1': f1
    })
results_df = pd.DataFrame(results)

print(results_df)

best_recall_depth = results_df.loc[results_df['recall'].idxmax(), 'max_depth']
worst_precision_depth = results_df.loc[results_df['precision'].idxmin(), 'max_depth']
best_f1_depth = results_df.loc[results_df['f1'].idxmax(), 'max_depth']
