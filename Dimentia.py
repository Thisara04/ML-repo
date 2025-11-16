from google.colab import drive
drive.mount('/content/drive',force_remount=True)

import os
project_path = "/content/drive/MyDrive/Phase 1 - Resources"
print("Files will be saved here:", project_path)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import seaborn as sns

file="/content/drive/MyDrive/Phase 1 - Resources/Dataset/Dementia Prediction Dataset.csv"
df=pd.read_csv(file)
df.head()

selected_features=['SEX','HISPANIC','HISPOR','RACE','RACESEC','RACETER','PRIMLANG','EDUC','MARISTAT','NACCLIVS','INDEPEND','RESIDENC','HANDED','NACCAGE','NACCAGEB','INBIRYR','NEWINF','INEDUC','INRELTO','INKNOWN','INLIVWTH','INRELY','NACCFAM','NACCMOM','NACCDAD','ANYMEDS','NACCAMD','TOBAC100','SMOKYRS','PACKSPER','QUITSMOK','ALCFREQ','CVHATT','HATTMULT','CVBYPASS','CVPACE','CVHVALVE','CBSTROKE','TBIBRIEF','TBIEXTEN','DEP2YRS','DEPOTHR','NACCTBI','HEIGHT','WEIGHT','NACCBMI','VISION','VISCORR','VISWCORR','HEARING','HEARAID','HEARWAID','HXSTROKE','HALL','HALLSEV','APP','APPSEV','BILLS','TAXES','SHOPPING','GAMES','STOVE','MEALPREP','EVENTS','PAYATTN','REMDATES','TRAVEL','CDRGLOB']
df=df[selected_features]
df.head()

df['Dementia_state'] = df['CDRGLOB'].apply(lambda x: 1 if x >= 0.5 else 0)
df.head()

df['HISPANIC']=df['HISPANIC'].replace({9:pd.NA }).astype('Int64')
df['HISPOR']=df['HISPOR'].replace({50:7,88:8,99:pd.NA,-4:pd.NA}).astype('Int64')
df['RACE']=df['RACE'].replace({50:6,99:pd.NA}).astype('Int64')
df['RACESEC']=df['RACESEC'].replace({50:6,88:pd.NA,99:pd.NA}).astype('Int64')
df['RACETER']=df['RACETER'].replace({50:6,88:pd.NA,99:pd.NA}).astype('Int64')
df['PRIMLANG']=df['PRIMLANG'].replace({9:pd.NA}).astype('Int64')
df['EDUC']=df['EDUC'].replace({99:pd.NA}).astype('Int64')
df['MARISTAT']=df['MARISTAT'].replace({9:pd.NA}).astype('Int64')
df['NACCLIVS']=df['NACCLIVS'].replace({9:pd.NA}).astype('Int64')
df['INDEPEND']=df['INDEPEND'].replace({9:pd.NA}).astype('Int64')
df['RESIDENC']=df['RESIDENC'].replace({9:pd.NA}).astype('Int64')
df['HANDED']=df['HANDED'].replace({9:pd.NA}).astype('Int64')

columns = ['HISPANIC', 'HISPOR', 'RACE', 'RACESEC', 'RACETER', 'PRIMLANG', 'EDUC', 'MARISTAT', 'NACCLIVS', 'INDEPEND', 'RESIDENC', 'HANDED']

for col in columns:
    print(f"{col} value counts:")
    print(df[col].value_counts(dropna=False))  # dropna=False includes NaN counts
    print("-"*30)


df['INBIRYR']=df['INBIRYR'].replace({9999:pd.NA,-4:pd.NA}).astype('Int64')
df['NEWINF']=df['NEWINF'].replace({9:pd.NA}).astype('Int64')
df['INEDUC']=df['INEDUC'].replace({99:pd.NA,-4:pd.NA}).astype('Int64')
df['INRELTO']=df['INRELTO'].replace({-4:pd.NA}).astype('Int64')
df['INKNOWN']=df['INKNOWN'].replace({999:pd.NA,-4:pd.NA}).astype('Int64')
df['INLIVWTH']=df['INLIVWTH'].replace({-4:pd.NA}).astype('Int64')
df['INRELY']=df['INRELY'].replace({-4:pd.NA}).astype('Int64')

df['NACCFAM']=df['NACCFAM'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['NACCMOM']=df['NACCMOM'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['NACCDAD']=df['NACCDAD'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['ANYMEDS']=df['ANYMEDS'].replace({-4:pd.NA}).astype('Int64')
df['NACCAMD']=df['NACCAMD'].replace({-4:pd.NA}).astype('Int64')

df['TOBAC100']=df['TOBAC100'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['SMOKYRS']=df['SMOKYRS'].replace({88:pd.NA,99:pd.NA,-4:pd.NA}).astype('Int64')
df['PACKSPER']=df['PACKSPER'].replace({88:pd.NA,99:pd.NA,-4:pd.NA}).astype('Int64')
df['QUITSMOK']=df['QUITSMOK'].replace({888:pd.NA,999:pd.NA,-4:pd.NA}).astype('Int64')
df['ALCFREQ']=df['ALCFREQ'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')

df['CVHATT']=df['CVHATT'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['HATTMULT']=df['HATTMULT'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')
df['CVBYPASS']=df['CVBYPASS'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['CVPACE']=df['CVPACE'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['CVHVALVE']=df['CVHVALVE'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['CBSTROKE']=df['CBSTROKE'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')

df['TBIBRIEF']=df['TBIBRIEF'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['TBIEXTEN']=df['TBIEXTEN'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['DEP2YRS']=df['DEP2YRS'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['DEPOTHR']=df['DEPOTHR'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['NACCTBI']=df['NACCTBI'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')

df['HEIGHT']=df['HEIGHT'].replace({88.8:pd.NA,-4:pd.NA})
df['HEIGHT'] = df['HEIGHT'].astype('Float64')
df['HEIGHT'] = df['HEIGHT'] * (2.54/100)
df['HEIGHT'] = df['HEIGHT'].round(2)

df['WEIGHT']=df['WEIGHT'].replace({888:pd.NA,-4:pd.NA})
df['WEIGHT'] = df['WEIGHT'].astype('Float64')
df['WEIGHT'] = df['WEIGHT'] * 0.45359237
df['WEIGHT'] = df['WEIGHT'].round(2)

df['NACCBMI']=df['NACCBMI'].replace({888.8:pd.NA,-4:pd.NA})
df['NACCBMI'] = df['NACCBMI'].astype('Float64')
df['NACCBMI'] = df['NACCBMI']

print(df[['HEIGHT','WEIGHT','NACCBMI']].sample(10))

df['VISION']=df['VISION'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['VISCORR']=df['VISCORR'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['VISWCORR']=df['VISWCORR'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')

df['HEARING']=df['HEARING'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['HEARAID']=df['HEARAID'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['HEARWAID']=df['HEARWAID'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')

columns = ['VISION', 'VISCORR', 'VISWCORR', 'HEARING', 'HEARAID', 'HEARWAID']
for col in columns:
    print(f"{col} unique values:")
    print(df[col].unique())
    print("-"*30)

df['HXSTROKE']=df['HXSTROKE'].replace({-4:pd.NA}).astype('Int64')
df['HALL']=df['HALL'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['HALLSEV']=df['HALLSEV'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')
df['APP']=df['APP'].replace({9:pd.NA,-4:pd.NA}).astype('Int64')
df['APPSEV']=df['APPSEV'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')

df['BILLS']=df['BILLS'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')
df['TAXES']=df['TAXES'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')
df['SHOPPING']=df['SHOPPING'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')
df['GAMES']=df['GAMES'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')
df['STOVE']=df['STOVE'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')
df['MEALPREP']=df['MEALPREP'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')
df['EVENTS']=df['EVENTS'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')
df['PAYATTN']=df['PAYATTN'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')
df['REMDATES']=df['REMDATES'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')
df['TRAVEL']=df['TRAVEL'].replace({8:pd.NA,9:pd.NA,-4:pd.NA}).astype('Int64')

df = df.drop(['CDRGLOB'], axis=1)

import pandas as pd

with pd.option_context('display.max_columns', None):
    display(df.head())

import matplotlib.pyplot as plt
import seaborn as sns

# Compute correlation matrix
corr = df.corr()

plt.figure(figsize=(20, 15))
sns.heatmap(
    corr,
    cmap='coolwarm',
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    square=True
)

plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(rotation=0, fontsize=10)
plt.tight_layout()
plt.show()


corr = df.corr()

pd.set_option('display.max_rows', None)
corr_with_dementia = corr['Dementia_state'].sort_values(ascending=False)
# Display full list
print(corr_with_dementia)

import matplotlib.pyplot as plt

corr_with_target = df.corr()['Dementia_state'].sort_values()

# Plot vertical bar chart
plt.figure(figsize=(20, 8))
corr_with_target.plot(
    kind='bar',
    color='skyblue',
    edgecolor='black'
)
plt.ylabel('Correlation with Dementia_state', fontsize=12)
plt.xlabel('Features', fontsize=12)
plt.title('Correlation of Features with Dementia_state', fontsize=16)
plt.xticks(rotation=90, fontsize=10)  # Rotate x-axis labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

numeric_features = df.select_dtypes(include=['int64', 'Int64', 'float64', 'Float64']).columns.tolist()
numeric_features.remove('Dementia_state')

# Fill NaN with 0 or any imputation
X = df[numeric_features].fillna(0)

# Apply PCA
pca = PCA(n_components=2)
reduced = pca.fit_transform(X)

# Plot
plt.figure(figsize=(8,6))
plt.scatter(reduced[:,0], reduced[:,1], c=df['Dementia_state'], cmap='coolwarm', alpha=0.6)
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA of Numeric Features')
plt.colorbar(label='Dementia_state')
plt.show()

row_count = len(df)
print(f"row_count: {row_count}")
print("\n")
for col in df.columns:
    print(f"{col} - {df[col].isnull().sum()} ")

df=df.drop(['RACESEC','RACETER','INKNOWN','INEDUC','QUITSMOK','ALCFREQ','HATTMULT','HEARWAID','HALLSEV','APPSEV'],axis=1)
#remove as they have too much null values

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
!pip install catboost
from catboost import CatBoostClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.metrics import log_loss
import warnings
warnings.filterwarnings('ignore')

X = df.drop(['Dementia_state'], axis=1)
y = df['Dementia_state']

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4,shuffle=True)

from sklearn.impute import SimpleImputer

# Use median for numeric features
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

#scale data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

#define hyperparameter grids for each model
param_grids = {
    'Decision Tree': {'max_depth': [3, 5, 7, 12, None]},
    'Random Forest': {'n_estimators': [50, 100, 200], 'max_depth': [3, 5, 7, 12, None]},
    'K-Nearest Neighbors': {'n_neighbors': [3, 5, 7]},
    'Logistic Regression': {'C': [0.1, 1, 10]},
    'XGBoost': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1], 'max_depth': [3, 5, 7]},
    'CatBoost': {'iterations': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
}

#instantiate classification models with default parameters
models = {
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'Logistic Regression': LogisticRegression(multi_class='multinomial', solver='lbfgs',max_iter=1000),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss'),
    'CatBoost': CatBoostClassifier(verbose=0)
}

#Train and evaluate each model
for name, model in models.items():
    # Scale data for specific models
    if name in ['K-Nearest Neighbors', 'Support Vector Machine', 'Logistic Regression']:
        X_train_used = X_train_scaled
        X_test_used = X_test_scaled
      #use non-scaled data
    else:
        X_train_used = X_train
        X_test_used = X_test

    print(f"Training {name}...")

    grid_search = GridSearchCV(model, param_grids[name], cv=5, scoring='f1_macro', n_jobs=-1)
    grid_search.fit(X_train_used, y_train)

    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test_used)

    print(f'\n{ name } Classification Report:')
    print(classification_report(y_test, y_pred))
    print(f'Best Parameters: {grid_search.best_params_}\n{"-"*70}\n')

from sklearn.ensemble import VotingClassifier
from sklearn.metrics import classification_report, accuracy_score

#tune the data and train
best_rf = RandomForestClassifier(max_depth=None, n_estimators=200)

best_xgb = XGBClassifier(
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    use_label_encoder=False,
    eval_metric='mlogloss'
)

best_cat = CatBoostClassifier(
    iterations=200,
    learning_rate=0.01,
    verbose=0
)


best_rf.fit(X_train, y_train)
best_xgb.fit(X_train, y_train)
best_cat.fit(X_train, y_train)

#create voting classifier
voting_clf = VotingClassifier(
    estimators=[
        ('rf', best_rf),
        ('xgb', best_xgb),
        ('cat', best_cat)
    ],
    voting='soft'
)

# Train the ensemble
voting_clf.fit(X_train, y_train)

# Predict
y_pred = voting_clf.predict(X_test)

# Evaluate
print("Ensemble Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import joblib
from sklearn.calibration import CalibratedClassifierCV

# Save the scaler
joblib.dump(scaler, "/content/scaler.pkl")
print("Scaler saved successfully!")

# Calibrate the voting classifier
calibrated_model = CalibratedClassifierCV(
    voting_clf,
    cv='prefit',
    method='sigmoid'
)

# Fit calibration on the test set
calibrated_model.fit(X_test, y_test)

# Save the calibrated model
joblib.dump(calibrated_model, "/content/ensemble_calibrated.pkl",compress=3)
print("Calibrated ensemble model saved successfully!")

from google.colab import files

files.download("/content/scaler.pkl")
files.download("/content/ensemble_calibrated.pkl")

importances = best_rf.feature_importances_
feature_names = X_train.columns

# Create a DataFrame
feat_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feat_df = feat_df.sort_values(by='Importance', ascending=True)  # ascending for horizontal barplot

# Plot
plt.figure(figsize=(10,12))
sns.barplot(x='Importance', y='Feature', data=feat_df, palette='viridis')
plt.title("Feature Importances", fontsize=16)
plt.xlabel("Importance", fontsize=14)
plt.ylabel("Features", fontsize=14)
plt.tight_layout()
plt.show()

with pd.option_context('display.max_columns', None):
    display(df.head())

input_df1 = {
    "SEX": 1, "HISPANIC": 0, "HISPOR": 8, "RACE": 1, "PRIMLANG": 1, "EDUC": 16, "MARISTAT": 1,
    "NACCLIVS": 4, "INDEPEND": 1, "RESIDENC": 1, "HANDED": 2, "NACCAGE": 70, "NACCAGEB": 70,
    "INBIRYR": 1962, "NEWINF": -4, "INRELTO": 1, "INLIVWTH": 1, "INRELY": 0, "NACCFAM": 1,
    "NACCMOM": 0, "NACCDAD": 0, "ANYMEDS": 1, "NACCAMD": 13, "TOBAC100": 0, "SMOKYRS": 0,
    "PACKSPER": 0, "CVHATT": 0, "CVBYPASS": 0, "CVPACE": 0, "CVHVALVE": 0, "CBSTROKE": 0,
    "TBIBRIEF": 0, "TBIEXTEN": 0, "DEP2YRS": 0, "DEPOTHR": 0, "NACCTBI": 0, "HEIGHT": 1.8,
    "WEIGHT": 105.23, "NACCBMI": np.nan, "VISION": 0, "VISCORR": 1, "VISWCORR": np.nan,
    "HEARING": 1, "HEARAID": np.nan, "HXSTROKE": np.nan, "HALL": 0, "APP": 0, "BILLS": 0,
    "TAXES": 1, "SHOPPING": 0, "GAMES": 1, "STOVE": 0, "MEALPREP": 1, "EVENTS": 0, "PAYATTN": 1,
    "REMDATES": 1, "TRAVEL": 1
}

# Convert dictionary to DataFrame (shape: 1 row)
input_df1 = pd.DataFrame([input_df1])

#using the voting classifier
prediction = voting_clf.predict(input_df1)
prediction_prob = voting_clf.predict_proba(input_df1)

prediction_label = {0: "Non-Dementia", 1: "Risk of Dementia"}

print("Predicted Dementia State:", prediction_label[prediction[0]])
print("Prediction Probabilities:", prediction_prob[0])

input_df2 = {
    "SEX": 0, "HISPANIC": 1, "HISPOR": 7, "RACE": 2, "PRIMLANG": 1, "EDUC": 12, "MARISTAT": 2,
    "NACCLIVS": 3, "INDEPEND": 0, "RESIDENC": 1, "HANDED": 1, "NACCAGE": 75, "NACCAGEB": 75,
    "INBIRYR": 1945, "NEWINF": -4, "INRELTO": 3, "INLIVWTH": np.nan, "INRELY": 1, "NACCFAM": 0,
    "NACCMOM": 0, "NACCDAD": 0, "ANYMEDS": 1, "NACCAMD": 5, "TOBAC100": 1, "SMOKYRS": 25,
    "PACKSPER": 1, "CVHATT": 1, "CVBYPASS": 0, "CVPACE": np.nan, "CVHVALVE": 0, "CBSTROKE": 0,
    "TBIBRIEF": 0, "TBIEXTEN": 0, "DEP2YRS": 0, "DEPOTHR": 0, "NACCTBI": 0, "HEIGHT": 1.65,
    "WEIGHT": 68, "NACCBMI": 25.0, "VISION": 1, "VISCORR": 0, "VISWCORR": 1, "HEARING": 1,
    "HEARAID": 0, "HXSTROKE": np.nan, "HALL": 0, "APP": 1, "BILLS": 0, "TAXES": 0, "SHOPPING": 1,
    "GAMES": 0, "STOVE": 0, "MEALPREP": 1, "EVENTS": 0, "PAYATTN": 1, "REMDATES": 0, "TRAVEL": 1
}

# Convert to DataFrame
input_df2 = pd.DataFrame([input_df2])

#the voting classifier
prediction2 = voting_clf.predict(input_df2)
prediction_prob2 = voting_clf.predict_proba(input_df2)

prediction_label = {0: "Non-Dementia", 1: "Risk of Dementia"}

print("Predicted Dementia State:", prediction_label[prediction[0]])
print("Prediction Probabilities:", prediction_prob[0])

low_risk_sample = X_train[y_train == 0].iloc[0]  # first sample with Dementia_state = 0
input_df_low_risk = pd.DataFrame([low_risk_sample])

# Predict
prediction_low_risk = voting_clf.predict(input_df_low_risk)
prediction_prob_low_risk = voting_clf.predict_proba(input_df_low_risk)

prediction_label = {0: "Non-Dementia", 1: "Risk of Dementia"}

print("Predicted Dementia State:", prediction_label[prediction[0]])
print("Prediction Probabilities:", prediction_prob[0])

input_df3 = {
    "SEX": 0, "HISPANIC": 0, "HISPOR": 1, "RACE": 1, "PRIMLANG": 1, "EDUC": 16, "MARISTAT": 1,
    "NACCLIVS": 0, "INDEPEND": 1, "RESIDENC": 0, "HANDED": 1, "NACCAGE": 65, "NACCAGEB": 65,
    "INBIRYR": 1958, "NEWINF": 0, "INRELTO": 0, "INLIVWTH": 1, "INRELY": 0, "NACCFAM": 1,
    "NACCMOM": 0, "NACCDAD": 0, "ANYMEDS": 0, "NACCAMD": 0, "TOBAC100": 0, "SMOKYRS": 0,
    "PACKSPER": 0, "CVHATT": 0, "CVBYPASS": 0, "CVPACE": 0, "CVHVALVE": 0, "CBSTROKE": 0,
    "TBIBRIEF": 0, "TBIEXTEN": 0, "DEP2YRS": 0, "DEPOTHR": 0, "NACCTBI": 0, "HEIGHT": 1.70,
    "WEIGHT": 70, "NACCBMI": 24, "VISION": 1, "VISCORR": 1, "VISWCORR": 1, "HEARING": 1,
    "HEARAID": 0, "HXSTROKE": 0, "HALL": 0, "APP": 0, "BILLS": 0, "TAXES": 0, "SHOPPING": 0,
    "GAMES": 0, "STOVE": 0, "MEALPREP": 0, "EVENTS": 0, "PAYATTN": 0, "REMDATES": 0, "TRAVEL": 0
}

input_df3 = pd.DataFrame([input_df3])

prediction = voting_clf.predict(input_df3)
prediction_prob = voting_clf.predict_proba(input_df3)[0]

# Map
prediction_label = {0: "Non-Dementia", 1: "Risk of Dementia"}

# Estimated risk as percentage
risk_percentage = prediction_prob[1] * 100  # probability of dementia

print("Predicted Dementia State:", prediction_label[prediction[0]])

print(f"Probability of Non-Dementia: {prediction_prob[0]*100:.2f}%")
print(f"Probability of Risk of Dementia: {prediction_prob[1]*100:.2f}%")

from sklearn.metrics import accuracy_score, classification_report

# Predict for the test set
y_pred = voting_clf.predict(X_test)
y_pred_prob = voting_clf.predict_proba(X_test)

# Map numeric predictions to labels
prediction_label = {0: "Non-Dementia", 1: "Risk of Dementia"}
y_pred_labels = [prediction_label[p] for p in y_pred]

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {accuracy:.2f}")

#classification report
print("Classification Report:\n", classification_report(y_test, y_pred))


