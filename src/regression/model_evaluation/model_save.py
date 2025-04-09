from src.regression.model_evaluation.model_train import model_saving
from src.regression.model_evaluation.model_accuracy import accuracy
import pandas as pd
import pickle

df = pd.read_csv('H:\\linear_regression\\new_data\\data.csv')
model_name,model = model_saving(df)
model_accuracy = accuracy(df)

print(f"Model Name : {model_name}")
print(f"Model Accuracy : {model_accuracy}")
with open('model\Regression_model.pkl','wb') as file:
    pickle.dump(model,file)
    print("Successfully Save the Model!")

