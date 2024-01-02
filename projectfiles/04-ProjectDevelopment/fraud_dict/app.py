from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the machine learning model and scaler
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))


@app.route('/predict')
def about():
    return render_template('predict1.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from the form
    step = int(request.form.get('step'))
    trans_type = request.form.get('Type')
    amount = float(request.form.get('amount'))
    old_balance_org = float(request.form.get('oldbalanceOrg'))
    new_balance_orig = float(request.form.get('newbalanceOrig'))
    old_balance_dest = float(request.form.get('oldbalanceDest'))
    new_balance_dest = float(request.form.get('newbalanceDest'))

    # Prepare the input data for prediction
    input_data = [[step, trans_type, amount, old_balance_org, new_balance_orig, old_balance_dest, new_balance_dest]]
    x = scaler.transform(input_data)
    prediction = model.predict(x)

    # Assuming 'prediction' is a binary result (0 or 1)
    result = "Fraud Transaction" if prediction[0] == 1 else "Legitimate Transaction"

    return render_template('predict1.html', prediction=result)

if __name__ == "__main__":
    app.run()
