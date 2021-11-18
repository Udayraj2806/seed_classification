from flask import Flask,request,jsonify
import numpy as np
import pickle

model = pickle.load(open('model.pkl','rb'))

app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/classify',methods=['POST'])
def predict():

    area = request.form.get('area')
    perimeter = request.form.get('perimeter')
    compactness = request.form.get('compactness')
    lengthOfKernel = request.form.get('lengthOfKernel')
    widthOfKernel = request.form.get('widthOfKernel')
    asymmetryCoefficient = request.form.get('asymmetryCoefficient')
    lengthOfKernelGroove = request.form.get('lengthOfKernelGroove')




    input_query = np.array([[area,perimeter,compactness,lengthOfKernel,widthOfKernel,asymmetryCoefficient,lengthOfKernelGroove]])

    result = model.predict(input_query)[0]

    return jsonify({'seed_type':str(result)})

if __name__ == '__main__':
    app.run(debug=True)