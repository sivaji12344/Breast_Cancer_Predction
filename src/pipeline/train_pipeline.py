import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

def load_data_from_csv(filename):
    data = pd.read_csv(filename)
    return data

if __name__ == "__main__":
    # Load engineered train and test data from CSV files
    train_data_engineered = load_data_from_csv('train_data_engineered.csv')

    # Split features and target
    X_train = train_data_engineered.drop(columns=['target'])
    y_train = train_data_engineered['target']

    # Define the preprocessing steps
    scaler = StandardScaler()

    # Define the model and hyperparameters
    model = SVC()
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    }

    # Create a pipeline
    pipeline = Pipeline([
        ('scaler', scaler),
        ('model', GridSearchCV(model, param_grid, cv=3))
    ])

    # Fit the pipeline on the training data
    pipeline.fit(X_train, y_train)

    # Save the scaler and model as pickle files
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(pipeline.named_steps['model'].best_estimator_, 'model.pkl')

    print("Scaler and model saved as pickle files.")
