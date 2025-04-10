from flask import Flask,render_template,request
import pandas as pd
import pickle

app = Flask(__name__)

with open('H:\linear_regression\model\Regression_model.pkl','rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
