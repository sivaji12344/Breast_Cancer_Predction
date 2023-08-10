import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

def load_data_from_csv(filename):
    data = pd.read_csv(filename)
    return data

if __name__ == "__main__":
    # Load engineered train and test data from CSV files
    train_data_engineered = load_data_from_csv('train_data_engineered.csv')
    test_data_engineered = load_data_from_csv('test_data_engineered.csv')

    # Split features and target
    X_train = train_data_engineered.drop(columns=['target'])
    y_train = train_data_engineered['target']
    X_test = test_data_engineered.drop(columns=['target'])
    y_test = test_data_engineered['target']

    # Define the model (SVM as an example)
    model = SVC()

    # Define hyperparameters to search
    param_grid = {
        'C': [0.1, 1, 10],
        'kernel': ['linear', 'rbf', 'poly']
    }

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X_train, y_train)

    # Get the best model
    best_model = grid_search.best_estimator_

    # Make predictions on test data
    y_pred = best_model.predict(X_test)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy:", accuracy)
