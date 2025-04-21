
from flask import Flask, render_template, request
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import RobustScaler
import joblib
import pandas as pd

app = Flask(__name__)

# Load the XGBoost model and scaler
model = xgb.Booster()
model.load_model('xgboost_gps_spoofing_model.json')
scaler = joblib.load('scaler.pkl')

# Define expected value ranges for validation
VALID_RANGES = {
    'PRN': (1, 32),
    'DO': (-4000000, 4000),
    'PD': (173687.0, 173687.3),
    'CP': (0, 1e10),
    'EC': (0, 1e10),
    'LC': (0, 1e10),
    'PC': (0, 1e10),
    'PIP': (1, 1e10),
    'PQP': (0, 1e10),
    'TCD': (0, 1e10),
    'CN0': (10, 60)
}

def engineer_features(input_data):
    """Engineer features from input data"""
    df = pd.DataFrame([input_data])

    df['CP_diff'] = 0
    df['CP_consistency'] = 0
    df['signal_quality'] = df['CN0']
    df['PC_PIP_ratio'] = df['PC'] / df['PIP']
    df['EC_LC_ratio'] = df['EC'] / df['LC']
    df['CP_rate'] = 0
    df['PD_rate'] = 0
    df['quality_score'] = ((df['CN0'] >= 45.0) & (np.abs(df['PC'] - df['PIP']) < 1000.0)).astype(int)

    return df

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Parse form inputs
            input_features = {
                key: float(request.form[key]) for key in VALID_RANGES
            }

            # Validate ranges
            out_of_range = []
            for key, val in input_features.items():
                low, high = VALID_RANGES[key]
                if not (low <= val <= high):
                    out_of_range.append(f"{key}: {val} (Expected: {low}–{high})")

            if out_of_range:
                message = "⚠️ The following values are out of expected range:\n" + "\n".join(out_of_range)
                return render_template('index.html', prediction_text=message)

            # Feature engineering
            df_processed = engineer_features(input_features)

            features = ['PRN', 'DO', 'PD', 'CP', 'EC', 'LC', 'PC', 'PIP', 'PQP', 'TCD', 'CN0',
                        'CP_consistency', 'signal_quality', 'PC_PIP_ratio', 'EC_LC_ratio',
                        'CP_rate', 'PD_rate', 'quality_score']

            features_scaled = scaler.transform(df_processed[features])
            dtest = xgb.DMatrix(features_scaled)

            # Predict
            pred_probs = model.predict(dtest)
            prediction = np.argmax(pred_probs[0])

            labels = {
                0: "Authentic GPS Signal (Unspoofed)",
                1: "Spoofed GPS Signal (Type 1)",
                2: "Spoofed GPS Signal (Type 2)",
                3: "Spoofed GPS Signal (Type 3)"
            }

            result = labels.get(prediction, f"Unknown Prediction (Raw Output: {prediction})")
            confidence = float(pred_probs[0][prediction]) * 100

            return render_template('index.html', 
                                   prediction_text=f'{result}',
                                   confidence=f'Confidence: {confidence:.2f}%')

        except ValueError:
            return render_template('index.html', 
                                   prediction_text="Invalid input! Please enter numerical values.")
        except Exception as e:
            return render_template('index.html', 
                                   prediction_text=f"An error occurred: {str(e)}")

if __name__ == '__main__':
    app.run(debug=True)
