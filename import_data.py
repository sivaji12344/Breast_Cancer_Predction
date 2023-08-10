import pymongo
from sklearn.datasets import load_breast_cancer
from src.logger import logging



def insert_data_into_Mongo(database_name,collection_name,data):
    client=pymongo.MongoClient('mongodb+srv://sivajisiva49:sivajisiva49@cluster0.6hyzjnn.mongodb.net/?retryWrites=true&w=majority')
    db=client[database_name]
    collecton=db[collection_name]

    collecton.insert_many(data.to_dict(orient='records'))

if __name__=="__main__":
    breast_cancer_data=load_breast_cancer()

    #converting the loaded dataset into pandas dataframe
    import pandas as pd
    data = pd.DataFrame(breast_cancer_data.data,columns=breast_cancer_data.feature_names)
    data['target']=breast_cancer_data.target

    #replacing values in target variable wth class names
    data['target']=data['target'].replace({0:'malignant',1:'Benign'})

    database_name='Dataset'
    collection_name='Breast_cancer_dataset'

    insert_data_into_Mongo(database_name,collection_name,data)
    print("Data inserted into MongoDB successfully.")
    logging.info('Data imported succefully into MongoDB')
