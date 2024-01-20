# importing all the libraries

import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model
import pickle


sns.set_style("darkgrid")
pd.set_option("display.max_columns", None)
pd.options.plotting.backend = "plotly"




# Reading Data

df = pd.read_csv("C:/Users/arjun/Desktop/Final Year Project/Data/autism_screening.csv")
print(df.head())




# Data Cleaning 

df.drop(index = 52, inplace = True)
df.reset_index(inplace = True)


df['age'] = df['age'].fillna(np.round(df['age'].mean(), 0))

# replacing
df['ethnicity'] = df['ethnicity'].replace('?', 'Others')
df['ethnicity'] = df['ethnicity'].replace('others', 'Others')
df['relation'] = df['relation'].replace('?', df['relation'].mode()[0])

# droping row
df.drop(['index','age_desc'], axis=1, inplace=True)

# Extract features and target variable
X = df.drop("Class/ASD", axis=1)
Y = df['Class/ASD']

# Save the one-hot encoding transformation
one_hot_encoder = pd.get_dummies(X)
pickle.dump(one_hot_encoder, open('one_hot_encoder.pkl', 'wb'))




# Splitting data in train and test

# Splitting data in train and test
X_train, X_test, Y_train, Y_test = train_test_split(one_hot_encoder, pd.get_dummies(df['Class/ASD']), test_size=0.25)
print(f"Shape of X_train is: {X_train.shape}")
print(f"Shape of Y_train is: {Y_train.shape}\n")
print(f"Shape of X_test is: {X_test.shape}")
print(f"Shape of Y_test is: {Y_test.shape}")




# Creating ANN model

input_dim = X_train.shape[1]  # Assuming X_train has been one-hot encoded
model = Sequential()
model.add(Dense(8, input_dim=input_dim, kernel_initializer='normal', activation='relu'))
model.add(Dense(5, activation="relu", kernel_initializer='normal'))
model.add(Dense(2, activation='sigmoid'))
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])





# Checking summary of Model

model.summary()




# Training Model

model.fit(X_train, Y_train, epochs=20, batch_size=10)



# neural_network = model.fit(X_train, Y_train, epochs = 20, batch_size = 10)

# acc = neural_network.history['accuracy']
# loss = neural_network.history['loss']

# epoch = [i + 1 for i in range(len(acc))]




# Evaluating Model

# loss, acc = model.evaluate(X_test, Y_test)

# print(f"Accuracy on unseen data is: { np.round(acc, 2) }")
# print(f'Loss on unseen data is: { np.round(loss, 2) }')




# Classification Report

# prediction = model.predict(X_test)
# prediction = np.argmax(prediction, axis = 1)

# print(accuracy_score(Y_test[['YES']], prediction))

# print(classification_report(Y_test[['YES']], prediction))   



# Pickling (storing in a type of file format so that we can send it to any device for processing)

pickle.dump(model, open('model.pkl','wb'))
