import pymongo
import pandas as pd
from src.logger import logging
from sklearn.model_selection import train_test_split


def load_dataset_from_mongodb(database_name,collection_name):
    client=pymongo.MongoClient('mongodb+srv://sivajisiva49:sivajisiva49@cluster0.6hyzjnn.mongodb.net/?retryWrites=true&w=majority')
    db=client[database_name]
    collection=db[collection_name]

    cursor=collection.find({})
    data=pd.DataFrame(list(cursor))
    return data

if __name__=="__main__":
    database='Dataset'
    collection='Breast_cancer_dataset'

    data=load_dataset_from_mongodb(database,collection)    
    logging.info('Data Ingestion method starts')

    features = data.drop(columns=['_id', 'target'])  # Assuming '_id' is the ObjectId generated by MongoDB
    target = data['target']
    train_features, test_features, train_target, test_target = train_test_split(features, target, test_size=0.2, random_state=42)

    # Save train and test data to CSV files
    train_data = pd.concat([train_features, train_target], axis=1)
    test_data = pd.concat([test_features, test_target], axis=1)
    
    train_data.to_csv('train_data.csv', index=False)
    test_data.to_csv('test_data.csv', index=False)

    print("Train and test data saved to CSV files.")

    

