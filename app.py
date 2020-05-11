from flask import Flask,render_template, url_for, request
import math
import pandas as pd
import numpy as np
import string
import re
from sklearn.externals import joblib


app = Flask(__name__)

@app.route('/',methods=['POST','GET'])
#Text Cleaning

def index():
    if request.method=='POST':
        review = request.form['content']
        review = [review]
        new = pd.DataFrame(review)
        new=new.rename(columns={0: "comment"})
        #lowercase and remove punctuation
        new["comment"] = new["comment"].str.lower().apply(lambda x:''.join([i for i in x if i not in string.punctuation]))
        mystop = joblib.load("MyStopwords.txt")
        #remove stopwords
        new["comment"] = new["comment"].apply(lambda x: ' '.join([word for word in x.split() if word not in (mystop)]))
        input_review=[]
        for i in new["comment"]:
            input_review.append(i)

        final_model = joblib.load("MyRidge.sav")
        #Prediction using live model
        y_pred = final_model.predict(input_review)

        return "The Prediction of the Rating is : " + str(y_pred)
    else:
        return render_template('index.html')



if __name__ == "__main__":
    app.run(debug=True)