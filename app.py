#import libraries
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

#Initialize the flask App
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
vectorizer = pickle.load(open('vectorizer.pkl', 'rb'))


#default page of our web-app
@app.route('/')
def home():
    return render_template('index.html')

#To use the predict button in our web-app
# @app.route('/predict',methods=['POST'])
@app.route('/',methods=['POST'])
def predict():
    '''
    For rendering results on HTML GUI
    '''
    # features = [float(x) for x in request.form.values()]
    # final_features = [np.array(int_features)]
    # prediction = model.predict(final_features)

    features = [x for x in request.form.values()]
    
    try:
        vectorized_input = vectorizer.transform(features)    
        output = model.predict(vectorized_input)[0]

    except:
        output='Unable to predict'

    return render_template('index.html', prediction_text='Detected Class is: {}'.format(output))

if __name__ == "__main__":
    app.run(debug=False)
