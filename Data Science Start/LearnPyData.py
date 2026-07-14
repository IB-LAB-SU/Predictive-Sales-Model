import numpy as np
from sklearn import tree
from sklearn.linear_model import LinearRegression  
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression  # Geïmporteerd
from sklearn.metrics import accuracy_score          # R² vervangen door Accuracy


# 1. Dataset [lengte, gewicht, schoenmaat]
X = [[181,80,44], [177,70,43], [160, 60, 38],
 [190,90,47], [175,64,39], [177,70,40], [181,85,43], [201, 85, 37], [175, 62, 39]]

# 2. Categorieën voor de Decision Tree -> Exact 9 labels
Y_tree = ['male', 'female', 'female', 'female', 'male', 'male', 'female', 'female', 'male']

# 3. Getallen voor de Linear Regression -> Exact 9 labels (0 = female, 1 = male)
Y_reg = [1, 0, 0, 0, 1, 1, 0, 0, 1]

# --- DEEL 1: DECISION TREE (Classificatie) ---
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y_tree)

prediction = clf.predict([[185, 60, 36]])
print(f"Decision Tree Voorspelling: {prediction}")

# --- DEEL 2: LINEAR REGRESSION (Regressie) ---
model = LinearRegression()
model.fit(X, Y_reg)            
y_pred = model.predict(X)      

# --- DEEL 3: LOGISTIC REGRESSION (Classificatie via Logistieke Regressie) ---
log_model = LogisticRegression()
log_model.fit(X, Y_reg)            # Traint direct op de tekstlabels!
y_pred_log = log_model.predict(X)     # Voorspelt 'male' of 'female'

# Nieuwe voorspelling voor de testpersoon
test_person = [[165, 60, 36]]
pred_log_person = log_model.predict(test_person)
# Bekijk ook de exacte kansberekening (bijv. 85% kans op male, 15% female)
pred_proba = log_model.predict_proba(test_person)

# --- RESULTATEN ---
print(f"Logistic Regression Voorspelling voor: {pred_log_person}")
print(f"Kansverdeling [Female, Male]: {pred_proba[0]}")
print(f"Model Nauwkeurigheid (Accuracy Score): {accuracy_score(Y_reg, y_pred_log):.4f}")

print(f"Helling (Slope) voor eigenschappen: {model.coef_}")
print(f"Snijpunt (Intercept): {model.intercept_:.4f}")
print(f"Model Nauwkeurigheid (R² score): {r2_score(Y_reg, y_pred):.4f}")