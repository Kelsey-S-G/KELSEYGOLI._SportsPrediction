#%% md
# Data Preparation and Feature Extraction
#%%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

players = pd.read_csv('players_22.csv')
male_legacy = pd.read_csv('male_players (legacy).csv')
#%%
male_legacy
#%%
male_legacy.info()
#%%
legacy_columns = set(male_legacy.columns)
players_22_columns = set(players.columns)

columns_to_drop = legacy_columns - players_22_columns

male_legacy = male_legacy.drop(columns=columns_to_drop)
#%%
L = []
L_less = []
for i in male_legacy.columns:
    if(male_legacy[i].isnull().sum())<(0.4*(male_legacy.shape[0])):
        L.append(i)
    else:
        L_less.append(i)
#%%
male_legacy = male_legacy[L]
#%%
numeric_data = male_legacy.select_dtypes(include=np.number)
non_numeric = male_legacy.select_dtypes(include=['object'])
#%%
y = numeric_data['overall']
#%%
n_nm = non_numeric['preferred_foot']
#%%
n_nm, values = pd.factorize(n_nm)
#%%
n_nm = pd.get_dummies(n_nm).astype(int)
#%%
n_nm.columns = ['Left', 'Right']
#%%
x = pd.concat([numeric_data, n_nm], axis=1)
#%%
x = pd.DataFrame(x, columns=x.columns)
#%%
x.fillna(x.mean(), inplace=True)
#%% md
# Correlation Analysis
#%%
correlation = x.corr()
#%%
correlation
#%%
correlation_target = correlation['overall']
selected_features = correlation_target[correlation_target.abs() > 0.4].index

x = x[selected_features]
#%%
x.drop('overall', axis=1, inplace=True)
#%% md
# Model Training with Cross-Validation
#%%
from sklearn.preprocessing import StandardScaler
scaler= StandardScaler()
x = scaler.fit_transform(x)
#%%
xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)
#%%
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
#%%
nb = GaussianNB()
sv = SVC(probability=True)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
#%%
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer, mean_absolute_error
import pickle as pkl
#%%
for model in (nb, sv, rf, gb):
    model.fit(xtrain, ytrain)
    pkl.dump(model, open('C:\\Users\\User\\OneDrive - Ashesi University\\Desktop\\Introduction to AI\\Jupyter Codes'  + model.__class__.__name__ + '.pkl', 'wb'))
    y_pred = model.predict(xtest)
    scorer = make_scorer(mean_absolute_error, greater_is_better=False)
    print(model.__class__.__name__, cross_val_score(model, xtrain, ytrain, cv=2, scoring=scorer))
#%% md
# Model Evaluation and Optimization
#%%
from sklearn.model_selection import GridSearchCV

parameters = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 20, 30]
}

grid_search = GridSearchCV(RandomForestClassifier(random_state=42), parameters, cv=5, scoring=scorer)
grid_search.fit(xtrain, ytrain)

print(f'Best parameters: {grid_search.best_params_}')
print(f'Best CV MAE: {-grid_search.best_score_}')

best_model = grid_search.best_estimator_

# Evaluate on validation set
y_pred = best_model.predict(xtest)
mae = mean_absolute_error(ytest, y_pred)
print(f'Validation MAE: {mae}')
#%% md
# Testing on a Different Dataset
#%%
xtest = players.drop('overall', axis=1)
ytest = players['overall']

y_test_pred = best_model.predict(xtest)
test_mae =mean_absolute_error(ytest, y_test_pred)
print(f'Test MAE: {test_mae}')
#%% md
# Deploying the Model on a Web Page
#%%
from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('best_model.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_data = pd.DataFrame([data])  
    
    # Predict using the loaded model
    prediction = model.predict(input_data)
    
    response = {
        'rating': prediction[0],
        'confidence': 0.95  
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)