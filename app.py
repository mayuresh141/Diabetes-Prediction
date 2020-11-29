from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open('random_forest_classifier_model1.pkl', 'rb'))

@app.route("/")
def home():
    return render_template("base.html")

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose =int(request.form['glucose'])
        bp = int(request.form['blood_pressure'])
        st = int(request.form['skin_thickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])

        data = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        my_prediction = model.predict(data)
    return render_template("result.html", prediction=my_prediction)
