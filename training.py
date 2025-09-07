import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

df = pd.read_csv("../Herma.csv")
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print('Training Logistic Regression with K-Fold CV...')

log_reg = LogisticRegressionCV(
    Cs=[0.01, 0.1, 1, 10, 100],  
    cv=5,  
    penalty='l2',  
    solver='liblinear',  
    scoring='accuracy',
    max_iter=1000
)
log_reg.fit(X_train, y_train)

print(f'Best C: {log_reg.C_}')
print(f'Training score: {log_reg.score(X_train, y_train)}')
print(f'Test score: {log_reg.score(X_test, y_test)}')

with open('Herma.pkl', 'wb') as pipeline_file:
    pickle.dump(log_reg, pipeline_file)

print("Model saved as 'Herma.pkl'.")