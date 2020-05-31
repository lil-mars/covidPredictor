# Artificial Neural Network

# Installing Theano
# pip install --upgrade --no-deps git+git://github.com/Theano/Theano.git

# Installing Tensorflow
# pip install tensorflow

# Installing Keras
# pip install --upgrade keras

# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Covid19_data.csv')
X = dataset.iloc[:, 9:23].values
y = dataset.iloc[:, 23].values

# onehotencoder = OneHotEncoder(categorical_features = [1])
# X = onehotencoder.fit_transform(X).toarray()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Part 2 - Now let's make the ANN!

# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(units=7, kernel_initializer='uniform', activation='relu', input_dim=14))

# Adding the second hidden layer
classifier.add(Dense(units=7, kernel_initializer='uniform', activation='relu'))
# Adding the output layer
classifier.add(Dense(units=1, kernel_initializer='uniform', activation='sigmoid'))
# Compiling the ANN
classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# Fitting the ANN to the Training set
# classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
classifier.fit(X_train, y_train, epochs=120)
# Part 3 - Making predictions and
# evaluating the model

# Predicting the Test set results
y_pred = classifier.predict(X_test)
# y_percentage = y_pred * 100
y_pred = (y_pred > 0.5)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

# Predicting a single new observation
"""
[1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0, 1]
Predict if the person has covid
1 fiebre: 1
2 tos: 1
3 cansansio: 1
4 difRespiratoria: 1
5 dolorMuscular: 1
6 dolorCabeza: 1,
7 olfato: 0,
8 gusto: 1,
9 diarre: 1,
10 erupcionCutanea: 0,
11 cambioColorManos: 0,
12 saleDeCasa: 1,
13 lavaManos: 1,
14 contactoEnfermo: 0,
15 covid: 1
"""
sc_1 = sc.transform(np.array([[1, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 0]]))
new_prediction = classifier.predict(sc_1)

new_prediction_3 = classifier.predict(sc.transform(np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]))) * 100
new_prediction = (new_prediction > 0.5)

from flask import Flask, request
from flask_restful import Resource, Api

app = Flask(__name__)
api = Api(app)


class Receiver(Resource):
    data = 'No user data'

    def post(self):
        print(request.json)
        request_json = request.json
        print(request_json['fiebre'], request_json['tos'])
        data = [
            request_json['fiebre'], request_json['tos'], request_json['cansansio'], request_json['difRespiratoria'],
            request_json['dolorMuscular'], request_json['dolorCabeza'], request_json['olfato'],
            request_json['gusto'],
            request_json['diarrea'], request_json['erupcion'], request_json['cambioColor'],
            request_json['saleCasa'],
            request_json['lavaManos'], request_json['contactoEnfermo']
        ]
        print(data)
        sc_2 = sc.transform(np.array([data]))
        prediction = classifier.predict(sc_2)
        probability = float(prediction * 100)
        print(probability)
        return {'probabilidad': probability, 'consejo': 'Cuidese'}

    def get(self):
        new_prediction = classifier.predict(sc.transform(np.array([[0.0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])))
        print(new_prediction)
        return {'probability': float(new_prediction) * 100}


api.add_resource(Receiver, '/receiver')
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=4444)
