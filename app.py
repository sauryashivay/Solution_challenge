from flask import Flask, render_template, request, jsonify
import joblib
from src.preprocess import PREPROCESSING
from src.predictor import PREDICTOR
from src.llm_engine import CustomerData,CreditLLMEngine
import pandas as pd
LLM_ENGINE = CreditLLMEngine()

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get data from the frontend (this is a dictionary)
        user_input = request.json
        
        # 2. ML Preprocessing & Prediction
        processor = PREPROCESSING(user_input)
        processed_data = processor.run()
        predictor = PREDICTOR(processed_data)
        ml_prediction = predictor.run()[0] 

        # 4. Prepare LLM Input (Merge the dicts)
        # This creates a single dictionary containing all user_input PLUS the Risk
        llm_payload = {**user_input, "Risk": ml_prediction}

        # 5. Get LLM Analysis
        # Pydantic will now receive a clean dictionary with 'good' or 'bad' for Risk
        customer = CustomerData(**llm_payload)
        llm_text = LLM_ENGINE.get_description(customer)
        
        return jsonify({
            'ml_output': "High Risk" if ml_prediction=="bad" else "Low Risk",
            'llm_output': llm_text,
            'status': 'success'
        })
        
    except Exception as e:
        print(f"Log Error: {e}") # This helps you see the trace in your terminal
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True)