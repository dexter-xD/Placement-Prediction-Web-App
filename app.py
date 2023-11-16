import numpy as np
import model
from flask import Flask, request, render_template
import pickle
import requests

app = Flask(__name__,template_folder="templates")
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['GET'])
def predict():
    
    gender = request.args.get('gender')
    stream = request.args.get('stream')
    internship = request.args.get('internship')
    cgpa = request.args.get('cgpa')
    backlogs = request.args.get('backlogs')
    arr = np.array([gender,stream,internship,cgpa,backlogs])
    brr = np.asarray(arr, dtype=float)
    output = model.predict([brr])
    if(output==1):
        out = 'You have high chances of getting placed 🎉🎉🎉'
    else:
        out = 'You have low chances of getting placed. 😟😟😟'

    advice_response = requests.get('https://api.adviceslip.com/advice')
    advice = advice_response.json().get('slip', {}).get('advice', 'No advice available')
    return render_template('out.html', output=out, advice=advice)

if __name__ == "__main__":
    app.run(debug=True)
