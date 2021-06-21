# Modules
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from flask import Flask, render_template, request
# Reading Dataset
dataset=pd.read_csv('Dataset_spine.csv')
dataset=dataset.iloc[:,:-1]

# Seperating the Input and Target Variables
X=dataset.iloc[:,:-1].values
Y=dataset.iloc[:,-1].values

# Data Pre-Processing 

# Categorical Data into Binary Data
le=LabelEncoder()
Y=le.fit_transform(Y)

# Feature Scaling - Standard Deviation
sc=StandardScaler()
X=sc.fit_transform(X)

# Splitting the Dataset into Train and Test
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.25)

# SVC Classifier
classifier=SVC(kernel='linear')
classifier.fit(X_train,Y_train)

app=Flask(__name__)

@app.route('/')
def gets_connected():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def read_data():
    c1=float(request.form['col1'])
    c2=float(request.form['col2'])
    c3=float(request.form['col3'])
    c4=float(request.form['col4'])
    c5=float(request.form['col5'])
    c6=float(request.form['col6'])
    c7=float(request.form['col7'])
    c8=float(request.form['col8'])
    c9=float(request.form['col9'])
    c10=float(request.form['col10'])
    c11=float(request.form['col11'])
    c12=float(request.form['col12'])
    X=dataset.iloc[:,:-1].values
    x_in=[c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12]
    X = np.append(X, np.array([x_in]), axis=0)
    X=np.array(sc.fit_transform(X))
    x_in=X[-1]
    y_out=classifier.predict([x_in])
    print(y_out)    
    if(y_out[0]==0):
        text='Abnormalities Detected'
    else:
        text='Normal'
    return render_template('index.html', pred=text)
    

if __name__=="__main__":
    app.run(debug=True)