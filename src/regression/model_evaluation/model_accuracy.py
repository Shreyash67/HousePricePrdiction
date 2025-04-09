import pandas as pd
from src.regression.model_evaluation.model_train import model_train
from src.regression.train_test.train_model import train_test
from sklearn.metrics import r2_score

def accuracy(df):
    
    y_pred = model_train(df)

    x_train, x_test, y_train, y_test = train_test(df)

    result = r2_score(y_test,y_pred)

    return f"{result*100}%"

# df = pd.read_csv('H:\\linear_regression\\new_data\\data.csv')
# result = accuracy(df)
# print(result)