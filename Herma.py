<<<<<<< HEAD
import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "https://darttgoblin.github.io"}})

with open('Herma.pkl', 'rb') as pipeline_file:
    herma = pickle.load(pipeline_file)

@app.route('/', methods=['POST'])
def handle_request():
    if request.method == 'OPTIONS':
        # Handle preflight request
        response = jsonify({'success': True})
        response.headers.add('Access-Control-Allow-Origin', 'https://darttgoblin.github.io')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type')
        response.headers.add('Access-Control-Allow-Methods', 'POST')
        return response
    
    data = request.get_json()  
    user_responses = data.get('user_responses')  
    
    result = herma.predict([user_responses])[0]
    probabilities = herma.predict_proba([user_responses])[0]

    result = True if result == 1 else False
    result_data = {
        'success': True,
        'result': result,
        'probabilities': probabilities.tolist()
    }

    return jsonify(result_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3069)
=======
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV

# Load dataset
df = pd.read_csv("../Herma.csv")

# Split features and labels
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # Last column

# Split dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Logistic Regression with L2 regularization and K-Fold Cross-Validation
print('Training Logistic Regression with K-Fold CV...')
log_reg = LogisticRegressionCV(
    Cs=[0.01, 0.1, 1, 10, 100],  # Range of C values
    cv=5,  # 5-Fold Cross-Validation
    penalty='l2',  # L2 Regularization
    solver='liblinear',  # Suitable for small datasets
    scoring='accuracy',
    max_iter=1000
)

# Train model
log_reg.fit(X_train, y_train)

# Evaluate model
print(f'Best C: {log_reg.C_}')
print(f'Training score: {log_reg.score(X_train, y_train)}')
print(f'Test score: {log_reg.score(X_test, y_test)}')

# Save model
with open('Herma.pkl', 'wb') as pipeline_file:
    pickle.dump(log_reg, pipeline_file)

print("Model saved as 'Herma.pkl'.")
>>>>>>> c33b84a654203803deef3b96ed69dcca72b7ce37
