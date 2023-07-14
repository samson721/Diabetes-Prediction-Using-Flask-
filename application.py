from flask import Flask, request, app, render_template
from flask import Response
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
import pickle


application = Flask(__name__)
app = application

standardscaler = pickle.load(open('models/scaler.pkl','rb'))
logmodel = pickle.load(open('models/modelforprediction.pkl','rb'))

## routing

@app.route("/")
def index():
    return render_template('index.html')


@app.route("/predict",methods = ['GET','POST'])
def predict():
    result=''

    if request.method == "POST":
        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=standardscaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=logmodel.predict(new_data)
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'
            
        return render_template('single_prediction.html',result=result)
    else:
        return render_template('home.html')


if __name__=="__main__":
    app.run(host="0.0.0.0")
