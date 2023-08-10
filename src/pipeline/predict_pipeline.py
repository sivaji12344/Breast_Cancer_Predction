import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
import joblib

def load_data_from_csv(filename):
    data = pd.read_csv(filename)
    return data

if __name__ == "__main__":
    # Load engineered test data from CSV file
    test_data_engineered = load_data_from_csv('test_data_engineered.csv')

    # Split features and target
    X_test = test_data_engineered.drop(columns=['target'])
    y_test = test_data_engineered['target']

    # Load the fitted preprocessing steps and model from disk
    scaler = joblib.load('scaler.pkl')
    model = joblib.load('model.pkl')

    # Create a prediction pipeline
    prediction_pipeline = Pipeline([
        ('scaler', scaler),
        ('model', model)
    ])

    # Make predictions on test data
    y_pred = prediction_pipeline.predict(X_test)

    # Print predictions and actual labels
    print("Predictions:", y_pred)
    print("Actual Labels:", y_test.tolist())
