from matplotlib import pyplot as plt
from pandas import read_csv
from sklearn import neighbors
from sklearn.calibration import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 
    'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]
df = read_csv('agaricus-lepiota.data', names=columns);
print(df.head())
df.replace("?", df.mode().iloc[0], inplace=True)

# Tiền xử lý: Chuyển đổi các giá trị phân loại thành giá trị số
label_encoder = LabelEncoder();

for col in df.columns:
    df[col] = label_encoder.fit_transform(df[col])

print(df.head())
X = df.drop('class', axis=1)
y = df['class'];
for i in range(2):
    print(f"sample data of class {i}:")
    print(X[y==i].head())
    print()

Accuracies = [];
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=40+i)
    model_KNN = neighbors.KNeighborsClassifier(n_neighbors=15, p=2)
    model_KNN.fit(X_train, y_train)

    y_pred = model_KNN.predict(X_test)
    if(i==2):
        print("Print results for 20 test data points")
        print("Predicted labels :", ', '.join(map(str, y_pred[0:20])))
        print("Ground truth     :", ', '.join(map(str, y_test[0:20])))

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred);
    Accuracies.append(accuracy);
    print("Accurancy of KNN: %.2f %%" %(100 * accuracy))

mean_accuracy = sum(Accuracies) / len(Accuracies);
print("Average accuracy of KNN: %.2f %%" %(100 * mean_accuracy))
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

cm_df = pd.DataFrame(cm, index=['Edible', 'Poisonous'], columns=['Edible', 'Poisonous'])

plt.figure(figsize=(5, 4))
sns.heatmap(cm_df, annot=True, fmt="d", cmap="Blues")
plt.title('Confusion Matrix')
plt.ylabel('Actual Values')
plt.xlabel('Predicted Values')
plt.show()
