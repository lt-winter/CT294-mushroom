from pandas import read_csv
from sklearn.calibration import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import MinMaxScaler
columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 
    'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]
df = read_csv('agaricus-lepiota.data', names=columns);
# Tiền xử lý: Chuyển đổi các giá trị phân loại thành giá trị số
label_encoder = LabelEncoder();

for col in df.columns:
    df[col] = label_encoder.fit_transform(df[col])

X = df.drop('class', axis=1)
y = df['class'];
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
Accuracies = [];
for i in range(0,10):
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=40+i)
    model_Bayes = GaussianNB();
    model_Bayes.fit(X_train, y_train)
    y_pred = model_Bayes.predict(X_test)

    from sklearn.metrics import accuracy_score
    accuracy = accuracy_score(y_test, y_pred);
    Accuracies.append(accuracy);
    print("Accurancy of Bayes: %.2f %%" %(100 * accuracy))

mean_accuracy = sum(Accuracies) / len(Accuracies);
print("Average accuracy of KNN: %.2f %%" %(100 * mean_accuracy))

print("Average accuracy of Bayes: %.2f %%" %(100 * mean_accuracy))
