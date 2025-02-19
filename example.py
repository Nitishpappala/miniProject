import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
df = pd.read_excel("GPS_Data_Simplified_2D_Feature_Map.xlsx")

def engineer_advanced_features(df):
    """Enhanced feature engineering with focus on spoofing detection"""
    df = df.copy()
    
    # Signal quality metrics
    df['CN0_variance'] = df['CN0'].rolling(window=3).var().fillna(0)
    df['signal_strength_score'] = (df['CN0'] >= 45.0).astype(int)
    
    # Phase coherence metrics
    df['CP_jump'] = df['CP'].diff().abs()
    df['CP_stability'] = df['CP_jump'].rolling(window=5).mean().fillna(0)
    df['phase_anomaly'] = (df['CP_jump'] > df['CP_jump'].mean() + 2*df['CP_jump'].std()).astype(int)
    
    # Pseudorange consistency
    df['PD_PC_diff'] = np.abs(df['PD'] - df['PC'])
    df['range_consistency'] = (df['PD_PC_diff'] < df['PD_PC_diff'].mean() + 2*df['PD_PC_diff'].std()).astype(int)
    
    # Doppler shift analysis
    df['DO_variation'] = df['DO'].diff().abs()
    df['doppler_anomaly'] = (df['DO_variation'] > df['DO_variation'].mean() + 2*df['DO_variation'].std()).astype(int)
    
    # Cross-validation features
    df['PC_PIP_ratio'] = df['PC'] / df['PIP']
    df['EC_LC_ratio'] = df['EC'] / df['LC']
    
    # Signal quality score (complex)
    df['quality_score'] = (
        (df['CN0'] >= 45.0).astype(int) +
        (df['PD_PC_diff'] < df['PD_PC_diff'].median()).astype(int) +
        (df['CP_jump'] < df['CP_jump'].median()).astype(int) +
        (df['DO_variation'] < df['DO_variation'].median()).astype(int)
    )
    
    # Time-based features
    df['TCD_variation'] = df['TCD'].diff().abs()
    df['time_consistency'] = (df['TCD_variation'] < df['TCD_variation'].mean() + 2*df['TCD_variation'].std()).astype(int)
    
    return df

# Prepare the data
print("Engineering features...")
df_processed = engineer_advanced_features(df)

# Define features
features = [
    'PRN', 'DO', 'PD', 'CP', 'EC', 'LC', 'PC', 'PIP', 'PQP', 'TCD', 'CN0',
    'CN0_variance', 'signal_strength_score', 'CP_jump', 'CP_stability', 
    'phase_anomaly', 'PD_PC_diff', 'range_consistency', 'DO_variation',
    'doppler_anomaly', 'PC_PIP_ratio', 'EC_LC_ratio', 'quality_score',
    'TCD_variation', 'time_consistency'
]

X = df_processed[features]
y = df_processed['Output']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to DMatrix
dtrain = xgb.DMatrix(X_train_scaled, label=y_train)
dtest = xgb.DMatrix(X_test_scaled, label=y_test)

# Define XGBoost parameters with focus on spoofing detection
params = {
    'objective': 'multi:softprob',
    'num_class': len(y.unique()),
    'max_depth': 8,
    'eta': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.2,
    'scale_pos_weight': 1,
    'tree_method': 'hist',
    'eval_metric': ['mlogloss', 'merror'],
    'lambda': 1.5,  # L2 regularization
    'alpha': 0.5    # L1 regularization
}

# Train model with early stopping
print("Training model...")
num_rounds = 500
watchlist = [(dtrain, 'train'), (dtest, 'test')]
model = xgb.train(
    params, 
    dtrain, 
    num_rounds, 
    watchlist, 
    early_stopping_rounds=50,
    verbose_eval=100
)

# Save the model and scaler
model.save_model('xgboost_gps_spoofing_model.json')
import joblib
joblib.dump(scaler, 'scaler.pkl')

# Make predictions
y_pred = model.predict(dtest)
y_pred_labels = np.argmax(y_pred, axis=1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred_labels))

# Plot confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred_labels)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Feature importance analysis
importance_scores = model.get_score(importance_type='gain')
importance_df = pd.DataFrame.from_dict(importance_scores, orient='index', columns=['importance'])
importance_df = importance_df.sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(importance_df.head(10))