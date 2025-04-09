import pandas as pd
from src.regression.split.spliting import split_x_y
from sklearn.model_selection import train_test_split

def train_test(df):
    x,y = split_x_y(df)

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20,random_state=42)

    return x_train, x_test, y_train, y_test

# df = pd.read_csv('H:\\linear_regression\\new_data\\data.csv')
# x_train, x_test, y_train, y_test = train_test(df)
# print(x_train)