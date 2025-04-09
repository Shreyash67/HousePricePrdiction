import pandas as pd

def split_x_y(df):
    x = df.iloc[:,:3]
    y = df.iloc[:,3:]

    return x,y

# df = pd.read_csv('new_data\data.csv')
# x,y = split_x_y(df)

# print(x)
