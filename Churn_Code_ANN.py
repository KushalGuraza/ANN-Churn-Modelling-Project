import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout , Activation
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV

data = pd.read_csv("Z:/courses/GITHUB/2_Churn Modelling_ANN//Churn_Modelling.csv")

print(data.isnull().sum())

print(data.head())

#intializing dependent and independent variables
x = data.iloc[:,3:13]
y = data.iloc[:,-1]

# dummy_coding
x1 = x.select_dtypes(exclude = "number")
columns_to_drop = x1.columns
x_dummycoding = data[columns_to_drop]
x_dummycoding = pd.get_dummies(x_dummycoding,  drop_first= True)
x = x.drop(columns_to_drop, axis = 1)

x = pd.concat( [x , x_dummycoding], axis = 1)


#splitting the data
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=0)

#scaling the data
scaling = StandardScaler()
x_train = scaling.fit_transform(x_train)
x_test = scaling.transform(x_test)

#performing hyper-parameter tuning using gridsearch cv
def create_model(layers, activation):
    model = Sequential()
    for i, nodes in enumerate(layers):
        if i==0:
            model.add(Dense(nodes,input_dim=x_train.shape[1]))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
        else:
            model.add(Dense(nodes))
            model.add(Activation(activation))
            model.add(Dropout(0.3))
            
    model.add(Dense(units = 1, kernel_initializer= 'glorot_uniform', activation = 'sigmoid')) 
    
    model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
    return model
    
model = KerasClassifier(build_fn=create_model, verbose=0)

layers = [(20,), (40, 20), (45, 30, 15)]
activations = ['sigmoid', 'relu']
param_grid = dict(layers=layers, activation=activations, batch_size = [128, 256], epochs=[30])
grid = GridSearchCV(estimator=model, param_grid=param_grid,cv=5)

grid_result = grid.fit(x_train, y_train)

print(grid_result.best_params_)
print(grid_result.best_score_)


#modelling ANN
classifier = Sequential()

# adding the input and the first hidden layer
classifier.add(Dense( units = 40, kernel_initializer = 'glorot_uniform', activation="relu", input_dim =  11 ))
#dropping out few neurons to avoid overfitting the data
classifier.add(Dropout(0.3))

# adding second hidden layer
classifier.add(Dense(units = 20 , kernel_initializer = "glorot_uniform", activation = "relu"))
classifier.add(Dropout(0.3))

# adding the output layer
classifier.add(Dense(units = 1, activation = "sigmoid", kernel_initializer= "glorot_uniform"))

#compiling ANN
classifier.compile( optimizer = "Adam", loss = "binary_crossentropy", metrics=["accuracy"] )

#fitting the model
model = classifier.fit(x_train, y_train, validation_split=0.33, batch_size=128, epochs=30)

#predicting for test
y_pred = classifier.predict_classes(x_test)

#confusion matrix
cm = confusion_matrix(y_pred,y_test)

#accuracy_score
score = accuracy_score(y_pred,y_test)
print( "                                                 ")
print( "*************************************************")
print( "                                                 ") 
print( "Accuracy of the model is" + str(score*100) + "%")
print( "                                                 ")
print( "*************************************************")
print( "                                                 ")

#ANN MODEL
classifier.summary()











