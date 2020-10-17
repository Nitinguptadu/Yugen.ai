import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
import config


app = Flask(__name__)
model = pickle.load(open('Pickle_RL_Model1.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)

    output = round(prediction[0], 2)

    return render_template('index.html', prediction_text='[One Repersent Yes & Zero Repersent  No] Your Result  :-- {}'.format(output))


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=config.PORT, debug=config.DEBUG_MODE)
