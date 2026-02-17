import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io
from google.colab import files
import warnings

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

warnings.filterwarnings('ignore')

print("Heart Failure Prediction System")

# Upload Dataset 1 (299 rows) heart_failure_clinical_records_dataset
print("First, please upload 'heart_failure_clinical_records_dataset.csv'")
uploaded1 = files.upload()
file_name1 = list(uploaded1.keys())[0]
data1 = pd.read_csv(io.BytesIO(uploaded1[file_name1]))
print(f"Successfully uploaded {file_name1}\n")

# Upload Dataset 2 (918 rows) heart
print("Second, please upload 'heart.csv'")
uploaded2 = files.upload()
file_name2 = list(uploaded2.keys())[0]
data2 = pd.read_csv(io.BytesIO(uploaded2[file_name2]))
print(f"Successfully uploaded {file_name2}\n")


# Preprocessing and Evaluating first dataset

print("PART 1: Project 'DEATH_EVENT' (299-row Dataset)")


# Data Exploration
print("Statistical Summary (Dataset 1)")
print(data1.describe())

# Preprocessing
print("\nPreprocessing (Dataset 1)")
# This dataset is all numerical, so we only need to scale it.
X1 = data1.drop('DEATH_EVENT', axis=1)
y1 = data1['DEATH_EVENT']

# 80/20 split for training and testing
X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y1, test_size=0.2, random_state=42)
print(f"Dataset 1: {X1_train.shape[0]} training samples, {X1_test.shape[0]} testing samples.")

# Scale the data
scaler1 = StandardScaler()
X1_train_scaled = scaler1.fit_transform(X1_train)
X1_test_scaled = scaler1.transform(X1_test)
print("Dataset 1 preprocessed and scaled.")

# Defining Models
models_1 = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector (SVC)": SVC(random_state=42)
}

# Training and Evaluating All Models
print("\nModel Training & Evaluation (Dataset 1)")

for name, model in models_1.items():
    print(f"\n--- Results for: {name} ---")

    # Training
    model.fit(X1_train_scaled, y1_train)

    # Predicting
    y_pred1 = model.predict(X1_test_scaled)

    # Evaluating
    accuracy = accuracy_score(y1_test, y_pred1)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y1_test, y_pred1, target_names=['Did Not Die (0)', 'Died (1)']))

    print("Confusion Matrix:")
    cm1 = confusion_matrix(y1_test, y_pred1)
    print(cm1)

    # Plot heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm1, annot=True, fmt='g', cmap='Blues',
                xticklabels=['Predicted: No Death', 'Predicted: Death'],
                yticklabels=['Actual: No Death', 'Actual: Death'])
    plt.title(f"Confusion Matrix for {name} (Dataset 1)")
    plt.show()


# Preprocessing and Evaluating first dataset

print("PART 2: Project 'HeartDisease' (918-row Dataset)")

# Data Exploration
print("\nStatistical Summary (Dataset 2)")
print(data2.describe())

# Preprocessing
print("\nPreprocessing (Dataset 2)")
# This dataset has mixed (numerical and categorical) data
X2 = data2.drop('HeartDisease', axis=1)
y2 = data2['HeartDisease']

# Identify column types
numeric_features = ['Age', 'RestingBP', 'Cholesterol', 'FastingBS', 'MaxHR', 'Oldpeak']
categorical_features = ['Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', 'ST_Slope']

# Create the preprocessor
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])

# 80/20 split for training and testing
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y2, test_size=0.2, random_state=42)
print(f"Dataset 2: {X2_train.shape[0]} training samples, {X2_test.shape[0]} testing samples.")

# Apply the preprocessor
X2_train_processed = preprocessor.fit_transform(X2_train)
X2_test_processed = preprocessor.transform(X2_test)
print("Dataset 2 preprocessed, encoded, and scaled.")

# Define Models
models_2 = {
    "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "Support Vector (SVC)": SVC(random_state=42)
}

# Training and Evaluating All Models
print("\nModel Training & Evaluation (Dataset 2)")

for name, model in models_2.items():
    print(f"\nResults for: {name}")

    # Training
    model.fit(X2_train_processed, y2_train)

    # Predicting
    y_pred2 = model.predict(X2_test_processed)

    # Evaluating
    accuracy = accuracy_score(y2_test, y_pred2)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y2_test, y_pred2, target_names=['No Disease (0)', 'Has Disease (1)']))

    print("Confusion Matrix:")
    cm2 = confusion_matrix(y2_test, y_pred2)
    print(cm2)

    # Plot heatmap
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm2, annot=True, fmt='g', cmap='Greens',
                xticklabels=['Predicted: No Disease', 'Predicted: Disease'],
                yticklabels=['Actual: No Disease', 'Actual: Disease'])
    plt.title(f"Confusion Matrix for {name} (Dataset 2)")
    plt.show()

