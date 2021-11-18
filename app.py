from flask import Flask,request,jsonify
import numpy as np
import main
from main import Decision_Node,classify,Leaf,Question
import pickle

model = pickle.load(open('model.pkl','rb'))








app = Flask(__name__)

@app.route('/')
def index():
    return "Hello world"

@app.route('/classify',methods=['POST'])
def classify():

    area = request.form.get('area')#[10.87, 9.65, 8.8648, 7.139, 6.463, 5.696, 4.967]
    perimeter = request.form.get('perimeter')
    compactness = request.form.get('compactness')
    lengthOfKernel = request.form.get('lengthOfKernel')
    widthOfKernel = request.form.get('widthOfKernel')
    asymmetryCoefficient = request.form.get('asymmetryCoefficient')
    lengthOfKernelGroove = request.form.get('lengthOfKernelGroove')




   # input_query = [float(area),float(perimeter),float(compactness),float(lengthOfKernel),float(widthOfKernel),float(asymmetryCoefficient),float(lengthOfKernelGroove)]
    a=float(area)
    b=float(perimeter)
    c=float(compactness)
    d=float(lengthOfKernel)
    x=float(widthOfKernel)
    y=float(asymmetryCoefficient)
    z=float(lengthOfKernelGroove)
    p=[a, b, c, d, x, y, z]
    model.row=p
    model.new();
    e=model.print_leaf()


    return jsonify({'seed_type':str(e)})
    model.row=[]

if __name__ == '__main__':
    app.run(debug=True)