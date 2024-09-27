import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.regularizers import l2
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import pandas as pd

st.title("Car Evaluation Predictor")

data = pd.read_csv('car_cleaned.csv') 

X = data.drop('class', axis=1)  # Replace 'label_column' with the actual name of your label column
y = data['class']

def create_model():
    model = Sequential([
        Dense(64, activation='relu', input_shape = (6,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1, activation='sigmoid')  
    ])
    
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = create_model()

model.fit(X, y, epochs = 50, batch_size = 64)

# Define input features
buying_price = st.selectbox("Buying Price:", ["low", "med", "high", "vhigh"])
maint_cost = st.selectbox("Maintenance Cost:", ["low", "med", "high", "vhigh"])
doors = st.selectbox("Number of Doors:", ["2", "3", "4", "5 or more"])
persons = st.selectbox("Seating Capacity:", ["2", "3", "4", "5 or more"])
lug_boot = st.selectbox("Luggage Boot Size:", ["small", "med", "big"])
safety = st.selectbox("Safety Rating:", ["low", "med", "high"])

# Create a function to encode user input
def encode_input(buying, maint, doors, persons, lug_boot, safety):
    buying_mapping = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
    maint_mapping = {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3}
    doors_mapping = {'2': 0, '3': 1, '4': 2, '5 or more': 3}
    persons_mapping = {'2': 0, '3': 1, '4': 2, '5 or more': 3}
    lug_boot_mapping = {'small': 0, 'med': 1, 'big': 2}
    safety_mapping = {'low': 0, 'med': 1, 'high': 2}

    input_data = np.array([[buying_mapping[buying],
                             maint_mapping[maint],
                             doors_mapping[doors],
                             persons_mapping[persons],
                             lug_boot_mapping[lug_boot],
                             safety_mapping[safety]]])
    
    return input_data

if st.button('Predict'):
    input_data = encode_input(buying_price, maint_cost, doors, persons, lug_boot, safety)
    prediction = model.predict(input_data)
    
    # Convert prediction to acceptable/unacceptable
    output = "acceptable" if prediction[0][0] >= 0.5 else "unacceptable"
    
    st.write(f"The prediction is: {prediction[0][0]}")
    st.write(f"The prediction is: {output}")