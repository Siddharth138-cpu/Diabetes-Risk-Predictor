import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# 1. Load the dataset
df = pd.read_csv(r'D:\projects\sid\siddd.venv\kaggle_diabetes.csv')

# 2. Basic inspection
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("Data types:\n", df.dtypes)
print("Info:")
df.info()
print("Summary stats:\n", df.describe().T)
print("Any nulls present?", df.isnull().any().to_dict())

# 3. Rename column for convenience
df = df.rename(columns={'DiabetesPedigreeFunction': 'DPF'})
print("\nRenamed Columns:", df.columns.tolist())

# 4. Plot outcome distribution
plt.figure(figsize=(10, 7))
sns.countplot(x='Outcome', data=df)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.xlabel('Has Diabetes')
plt.ylabel('Count')
plt.title('Outcome Distribution')
plt.show()

# 5. Handle missing-ish values (zeros are invalid for these features)
df_copy = df.copy(deep=True)

cols_to_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df_copy[cols_to_fix] = df_copy[cols_to_fix].replace(0, np.nan)  # Use lowercase np.nan per NumPy 2.x :contentReference[oaicite:0]{index=0}
print("\nMissing values per column after replace:\n", df_copy.isnull().sum())

# 6. Histograms before imputation
df_copy.hist(figsize=(15, 15))
plt.suptitle("Before Imputation", y=0.92)
plt.show()

# 7. Impute according to distribution characteristics
df_copy['Glucose'].fillna(df_copy['Glucose'].mean(), inplace=True)
df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean(), inplace=True)
df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median(), inplace=True)
df_copy['Insulin'].fillna(df_copy['Insulin'].median(), inplace=True)
df_copy['BMI'].fillna(df_copy['BMI'].median(), inplace=True)
print("\nMissing values after imputation:\n", df_copy.isnull().sum())

# 8. Histograms after imputation
df_copy.hist(figsize=(15, 15))
plt.suptitle("After Imputation", y=0.92)
plt.show()

# 9. Prepare features and labels
X = df_copy.drop(columns='Outcome')
y = df_copy['Outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0
)
print(f"\nX_train size: {X_train.shape}, X_test size: {X_test.shape}")

# 10. Feature scaling
sc = StandardScaler()
X_train_scaled = sc.fit_transform(X_train)
X_test_scaled = sc.transform(X_test)

# 11. Define and tune models
def find_best_model(X, y):
    models = {
        'logistic_regression': {
            'model': LogisticRegression(solver='lbfgs', multi_class='auto', max_iter=1000),
            'parameters': {'C': [1, 5, 10]}
        },
        'decision_tree': {
            'model': DecisionTreeClassifier(),
            'parameters': {'criterion': ['gini', 'entropy'], 'max_depth': [5, 10]}
        },
        'random_forest': {
            'model': RandomForestClassifier(),
            'parameters': {'n_estimators': [10, 20, 50, 100]}
        },
        'svm': {
            'model': SVC(gamma='auto'),
            'parameters': {'C': [1, 10, 20], 'kernel': ['rbf', 'linear']}
        }
    }
    cv_shuffle = ShuffleSplit(n_splits=5, test_size=0.20, random_state=0)
    scores = []
    for name, mp in models.items():
        gs = GridSearchCV(mp['model'], mp['parameters'], cv=cv_shuffle, return_train_score=False)
        gs.fit(X, y)
        scores.append({
            'model': name,
            'best_parameters': gs.best_params_,
            'score': gs.best_score_
        })
    return pd.DataFrame(scores, columns=['model', 'best_parameters', 'score'])

print("\nBest Models Found:")
print(find_best_model(X_train_scaled, y_train))

# 12. Cross-validation on a tuned Random Forest
rf = RandomForestClassifier(n_estimators=20, random_state=0)
cv_scores = cross_val_score(rf, X_train_scaled, y_train, cv=5)
print(f"\nRandom Forest CV Average Accuracy: {cv_scores.mean():.4f}")

# 13. Final model training and evaluation
classifier = rf.fit(X_train_scaled, y_train)

# Evaluation on test set
y_pred = classifier.predict(X_test_scaled)
cm_test = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm_test, annot=True, cmap="Blues", fmt='g')
plt.title('Confusion Matrix - Test Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

acc_test = accuracy_score(y_test, y_pred) * 100
print(f"\nTest Set Accuracy: {acc_test:.2f}%")
print("Classification Report (Test Set):\n", classification_report(y_test, y_pred))

# Evaluation on training set
y_train_pred = classifier.predict(X_train_scaled)
cm_train = confusion_matrix(y_train, y_train_pred)

plt.figure(figsize=(10, 7))
sns.heatmap(cm_train, annot=True, cmap="Blues", fmt='g')
plt.title('Confusion Matrix - Training Set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

acc_train = accuracy_score(y_train, y_train_pred) * 100
print(f"\nTraining Set Accuracy: {acc_train:.2f}%")
print("Classification Report (Training Set):\n", classification_report(y_train, y_train_pred))

# 14. Prediction helper function
def predict_diabetes(Pregnancies, Glucose, BloodPressure,
                     SkinThickness, Insulin, BMI, DPF, Age):
    features = np.array([[int(Pregnancies),
                          float(Glucose),
                          float(BloodPressure),
                          float(SkinThickness),
                          float(Insulin),
                          float(BMI),
                          float(DPF),
                          int(Age)]])
    features_scaled = sc.transform(features)
    return classifier.predict(features_scaled)[0]

# Sample predictions
for sample in [
    (2, 81, 72, 15, 76, 30.1, 0.547, 25),
    (1, 117, 88, 24, 145, 34.5, 0.403, 40),
    (5, 120, 92, 10, 81, 26.1, 0.551, 67)
]:
    prediction = predict_diabetes(*sample)
    print("\nInput:", sample)
    print("Result:", "Oops! You have diabetes." if prediction else "Great! You don't have diabetes.")
