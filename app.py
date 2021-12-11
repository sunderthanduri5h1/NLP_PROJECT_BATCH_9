from flask import Flask, render_template
import pandas as pd
import numpy as np
from flask import request
from ldunlp import predict

import pandas as pd 
import numpy as np

data=pd.read_csv('Language Detection.csv')

data.head()

data["Language"].value_counts()

df = data["Text"]
df1 = data["Language"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df1 = le.fit_transform(df1)

df1

df

df.tail()

import re
import sys
li = []
# iterating through all the text
for x in df:
       
        x = re.sub(r'[!@#$(),n"%^*?:;~`0-9]', ' ', x)
        #x = re.sub(r'[]','',x)
        x = x.lower()
        li.append(x)

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
df= cv.fit_transform(li).toarray()
df.shape

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, df1, train_size = 0.80)

from sklearn.naive_bayes import MultinomialNB
model = MultinomialNB()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)



def predict(text):
     x = cv.transform([text]).toarray() 
     lang = model.predict(x)
     lang = le.inverse_transform(lang) 
     print("The langauge is in",lang[0])




app = Flask(__name__)

@app.route('/')
def home():
   return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
   if request.method=='POST':
      data=request.form['s1']
      print(data)
      a=predict(data)

   return render_template('index.html', prediction_text='The people who are effected by the above person are : \n\n {}'.format(a))

if __name__ == '__main__':
    app.debug = True
    app.run()






#s=input()
#predict(s)



