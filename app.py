from flask import Flask,render_template,request
import numpy as np
import pickle

app = Flask(__name__)

with open('H:\linear_regression\model\Regression_model.pkl','rb') as file:
    model = pickle.load(file)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['GET','POST'])
def predict():
    if request.method == 'GET':
        return render_template('index.html')
    else:
        try:
            Area = float(request.form['Area'])
            BHK = float(request.form['BHK'])
            Bathroom = float(request.form['Bathroom'])

            input_data = np.array([[Area,BHK,Bathroom]])
            result = model.predict(input_data)
            predict_price = float(result[0])

            return render_template('index.html',result = f"Predicted House Price: {predict_price:.2f}$")
        except Exception as e:
            return render_template('index.html',result=f"Error: {e}")

if __name__ == '__main__':
    app.run(debug=True)
