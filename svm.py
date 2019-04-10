import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix  

def linear_svm():
    # download data set: https://drive.google.com/file/d/13nw-uRXPY8XIZQxKRNZ3yYlho-CYm_Qt/view
    # info: https://archive.ics.uci.edu/ml/datasets/banknote+authentication

    # load data
    bankdata = pd.read_csv("bill_authentication.csv")  

    # see the data
    bankdata.shape  

    # see head
    bankdata.head()  

    # data processing
    X = bankdata.drop('Class', axis=1)  
    y = bankdata['Class']  

    from sklearn.model_selection import train_test_split  
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)  

    # train the SVM
    svclassifier = SVC(kernel='linear')  
    svclassifier.fit(X_train, y_train)  

    # predictions
    y_pred = svclassifier.predict(X_test)  

    # Evaluate model
    print(confusion_matrix(y_test,y_pred))  
    print(classification_report(y_test,y_pred))  


# Iris dataset  https://archive.ics.uci.edu/ml/datasets/iris4
def import_iris():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

    # Assign colum names to the dataset
    colnames = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'Class']

    # Read dataset to pandas dataframe
    irisdata = pd.read_csv(url, names=colnames) 

    # process
    X = irisdata.drop('Class', axis=1)  
    y = irisdata['Class']  

    # train
    from sklearn.model_selection import train_test_split  
    return train_test_split(X, y, test_size = 0.20)  

def polynomial_kernel():
    # TODO
    # Trains, predicts and evaluates the model
    
    # Train
    X_train, X_test, y_train, y_test = import_iris()
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    poly_kernel_svm_clf = Pipeline([("scaler", StandardScaler()), ("svm_clf", SVC(kernel="poly", degree=8, coef0=1, C=5))])
    poly_kernel_svm_clf.fit(X_train, y_train)
    
    # Predict
    poly_kernel_pred = poly_kernel_svm_clf.predict(X_test)
    
    # Evaluate model
    print("=================")
    print("Polynomial Kernel")
    print("=================")
    print(confusion_matrix(y_test, poly_kernel_pred))  
    print(classification_report(y_test, poly_kernel_pred)) 

def gaussian_kernel():
    # TODO
    # Trains, predicts and evaluates the model
    
    # Train
    X_train, X_test, y_train, y_test = import_iris()
    gaussian_kernel_svm_clf = SVC(kernel="rbf")
    gaussian_kernel_svm_clf.fit(X_train, y_train)
    
    # Predict
    gaussian_kernel_pred = gaussian_kernel_svm_clf.predict(X_test)
    
    # Evaluate model
    print("===============")
    print("Gaussian Kernel")
    print("===============")
    print(confusion_matrix(y_test, gaussian_kernel_pred))  
    print(classification_report(y_test, gaussian_kernel_pred)) 

def sigmoid_kernel():
    # TODO
    # Trains, predicts and evaluates the model
    
    # Train
    X_train, X_test, y_train, y_test = import_iris()
    sigmoid_kernel_svm_clf = SVC(kernel="sigmoid")
    sigmoid_kernel_svm_clf.fit(X_train, y_train)
    
    # Predict
    sigmoid_kernel_pred = sigmoid_kernel_svm_clf.predict(X_test)
    
    # Evaluate model
    print("==============")
    print("Sigmoid Kernel")
    print("==============")
    print(confusion_matrix(y_test, sigmoid_kernel_pred))  
    print(classification_report(y_test, sigmoid_kernel_pred)) 

def test():
    import_iris()
    polynomial_kernel()
    gaussian_kernel()
    sigmoid_kernel()

test()