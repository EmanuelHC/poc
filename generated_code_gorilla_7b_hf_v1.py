

import joblib
import pandas as pd

def load_model():
    model = joblib.load('sklearn_model.joblib')
    return model

def process_data(date):
    data = pd.DataFrame({'date': [date]})
    model = load_model()
    response = model.predict(data)
    return response[0]

date = '2019-12-09'
# Process the data
response = process_data(date)
print(response)