#Importing Libraries
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # or other relevant metrics
from sklearn.datasets import make_classification # for example data
import warnings

#Ignoring the warning
warnings.filterwarnings("ignore")

#Prepare Data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Initialize and Train Model
model = xgb.XGBClassifier(objective='binary:logistic', eval_metric='logloss', random_state=42)
model.fit(X_train, y_train)

#Make Predictions
y_pred = model.predict(X_test)

#Evaluate Model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")