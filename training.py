import numpy as np
import pickle
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

with open('Herma.pkl', 'rb') as pipeline_file:
    herma = pickle.load(pipeline_file)

@app.route('/', methods=['POST'])
def handle_request():
    data = request.get_json()  
    user_responses = data.get('user_responses')  
    print(user_responses)
    
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
