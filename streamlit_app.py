import streamlit as st
from tensorflow.keras.models import load_model

model = load_model('my_ann_model.h5')

st.title("Car Evaluation Predictor")

buying_price = st.selectbox("Buying Price:", ["low", "med", "high", "vhigh"])
maint_cost = st.selectbox("Maintenance Cost:", ["low", "med", "high", "vhigh"])
doors = st.selectbox("Number of Doors:", ["2", "3", "4", "5 or more"])
persons = st.selectbox("Seating Capacity:", ["2", "3", "4", "5 or more"])
lug_boot = st.selectbox("Luggage Boot Size:", ["small", "med", "big"])
safety = st.selectbox("Safety Rating:", ["low", "med", "high"])

ordinal_mapping = {
    'buying': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'maint': {'low': 0, 'med': 1, 'high': 2, 'vhigh': 3},
    'doors': {'2': 0, '3': 1, '4': 2, '5 or more': 3},
    'persons': {'2': 0, '3': 1, '4': 2, '5 or more': 3},
    'lug_boot': {'small': 0, 'med': 1, 'big': 2},
    'safety': {'low': 0, 'med': 1, 'high': 2},
}

input_data = np.array([
    ordinal_mapping['buying'][buying_price],
    ordinal_mapping['maint'][maint_cost],
    ordinal_mapping['doors'][doors],
    ordinal_mapping['persons'][persons],
    ordinal_mapping['lug_boot'][lug_boot],
    ordinal_mapping['safety'][safety]
]).reshape(1, -1)

if st.button("Predict"):
    prediction = model.predict(input_data)
    st.write(f"The predicted class is: {prediction[0][0]}")
