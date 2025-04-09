import pandas as pd
from src.regression.train_test.train_model import train_test
from sklearn.linear_model import LinearRegression

def model_train(df):
    x_train, x_test, y_train, y_test = train_test(df)

    model = LinearRegression()

    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)

    return y_pred

def model_saving(df):
    x_train, x_test, y_train, y_test = train_test(df)

    model = LinearRegression()

    model.fit(x_train,y_train)

    model_name = model.__class__.__name__

    return model_name, model


# df = pd.read_csv('H:\\linear_regression\\new_data\\data.csv')
# y_pred = model_train(df)
# print(y_pred)