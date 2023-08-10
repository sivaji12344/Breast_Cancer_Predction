import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data_from_csv(filename):
    data = pd.read_csv(filename)
    return data

def perform_feature_engineering(data):
    # Example: Standardize the feature columns
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(data.drop(columns=['target']))
    scaled_data = pd.DataFrame(scaled_features, columns=data.columns[:-1])
    scaled_data['target'] = data['target']
    return scaled_data

if __name__ == "__main__":
    # Load train and test data from CSV files
    train_data = load_data_from_csv('train_data.csv')
    test_data = load_data_from_csv('test_data.csv')

    # Perform feature engineering
    train_data_engineered = perform_feature_engineering(train_data)
    test_data_engineered = perform_feature_engineering(test_data)

    # Save the engineered train and test data to CSV files
    train_data_engineered.to_csv('train_data_engineered.csv', index=False)
    test_data_engineered.to_csv('test_data_engineered.csv', index=False)

    print("Feature engineering and transformation completed.")
