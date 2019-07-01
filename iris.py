from sklearn import datasets
from sklearn import svm
from sklearn.semi_supervised import label_propagation

import numpy as np
import pandas as pd 

class IrisClassifier:
    def __init__(self):
        self.train_simple()

    def get_train_data(self):
        # data,target = datasets.load_iris(return_X_y=True)
        # X = data
        # y = target
        # return X,y

        data=pd.read_csv('iris_training.csv')
        feature=data.iloc[:,:4].values
        target=data.iloc[:,4:].values.flatten()
        return feature,target


    def train_simple(self):
        X,y=self.get_train_data()
        svc=svm.SVC(gamma='auto')
        self.svc=svc.fit(X,y)

    def predict(self,data):
        result=self.svc.predict(data)
        return result

if __name__ == "__main__":
    classifier=IrisClassifier()
    print(classifier.predict([[5.9 ,3.,  5.1 ,1.8]]))