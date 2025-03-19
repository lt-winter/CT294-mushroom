import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt
from sklearn import tree
raw_df = pd.read_csv("agaricus-lepiota.data", header=None)

# print(raw_df)

X = raw_df.iloc[:, 1:]
# Tìm giá trị xuất hiện nhiều nhất để thay thế "?" ở cột (b)
most_common = X.iloc[:, 10].mode()[0]  
X.iloc[:, 10] = X.iloc[:, 10].replace("?", most_common)
# print(X)

y = raw_df.iloc[:, 0]
# print(y)

from sklearn.preprocessing import LabelEncoder

# Mã hóa nhãn y

y = LabelEncoder().fit_transform(y)

# Mã hóa các đặc trưng X
X = X.apply(LabelEncoder().fit_transform)

def accurate():
    accuracies = []
    for i in range(0, 10):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        print(X_train)
        # print(y_train)
        tree_clf = DecisionTreeClassifier(criterion="entropy", random_state=40 + i, max_depth=5)
        tree_clf.fit(X_train, y_train)

        fn = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat']
        cn = ['poisonous', 'edible']

        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10), dpi=300)
        tree.plot_tree(tree_clf, feature_names=fn, class_names=cn, filled=True)

        fig.savefig('tree.png')

        tree_clf.score(X_test, y_test)

        accuracy = tree_clf.score(X_test, y_test)

        accuracies.append(accuracy)
    return accuracies

accuracies = accurate()
print("Độ chính xác trung bình ", sum(accuracies) / len(accuracies) * 100, "%")



