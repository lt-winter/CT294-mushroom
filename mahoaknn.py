from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

# Define column names
columns = [
    'class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor', 
    'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color', 'stalk-shape', 
    'stalk-root', 'stalk-surface-above-ring', 'stalk-surface-below-ring', 
    'stalk-color-above-ring', 'stalk-color-below-ring', 'veil-type', 'veil-color', 
    'ring-number', 'ring-type', 'spore-print-color', 'population', 'habitat'
]

# Load dataset
df = read_csv('hehe.data', names=columns)

# Handle missing values by replacing '?' with the mode of the respective column
df.replace("?", df.mode().iloc[0], inplace=True)

# Encode categorical values
label_encoder = LabelEncoder()
for col in df.columns:
    df[col] = label_encoder.fit_transform(df[col])

# Save the processed data to a new CSV file
encoded_file_path = "encoded_data.csv"
df.to_csv(encoded_file_path, index=False)
