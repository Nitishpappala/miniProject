import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
import xgboost as xgb
import joblib

def create_and_save_scaler():
    """Create a sample dataset, fit the scaler, and save it"""
    # Create sample data with the expected features
    sample_data = pd.DataFrame({
        'PRN': np.random.uniform(1, 32, 100),
        'DO': np.random.uniform(-5000, 5000, 100),
        'PD': np.random.uniform(20000000, 25000000, 100),
        'CP': np.random.uniform(-1e7, 1e7, 100),
        'EC': np.random.uniform(0, 100, 100),
        'LC': np.random.uniform(0, 1000, 100),
        'PC': np.random.uniform(-1e7, 1e7, 100),
        'PIP': np.random.uniform(-1e7, 1e7, 100),
        'PQP': np.random.uniform(0, 100, 100),
        'TCD': np.random.uniform(0, 1000, 100),
        'CN0': np.random.uniform(30, 60, 100)
    })

    # Add engineered features
    sample_data['CP_consistency'] = np.random.uniform(0, 10, 100)
    sample_data['signal_quality'] = np.random.uniform(30, 60, 100)
    sample_data['PC_PIP_ratio'] = sample_data['PC'] / sample_data['PIP']
    sample_data['EC_LC_ratio'] = sample_data['EC'] / sample_data['LC']
    sample_data['CP_rate'] = np.random.uniform(-100, 100, 100)
    sample_data['PD_rate'] = np.random.uniform(-100, 100, 100)
    sample_data['quality_score'] = np.random.randint(0, 2, 100)

    # Create and fit the scaler
    scaler = RobustScaler()
    scaler.fit(sample_data)

    # Save the scaler
    joblib.dump(scaler, 'scaler.pkl')
    print("Scaler has been created and saved as 'scaler.pkl'")

    # Create and save a dummy XGBoost model if it doesn't exist
    # This is just for testing - you should replace it with your actual trained model
    if not os.path.exists('xgboost_gps_spoofing_model.json'):
        # Create dummy data
        X = sample_data
        y = np.random.randint(0, 4, 100)  # 4 classes
        
        # Create DMatrix
        dtrain = xgb.DMatrix(X, label=y)
        
        # Set parameters
        params = {
            'objective': 'multi:softprob',
            'num_class': 4,
            'max_depth': 6,
            'eta': 0.1
        }
        
        # Train a basic model
        model = xgb.train(params, dtrain, num_boost_round=10)
        
        # Save the model
        model.save_model('xgboost_gps_spoofing_model.json')
        print("Dummy XGBoost model has been created and saved as 'xgboost_gps_spoofing_model.json'")

if __name__ == "__main__":
    import os
    create_and_save_scaler()