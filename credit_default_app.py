import streamlit as st 
import numpy as np 
import pandas as pd
import joblib
import sklearn 
import os
import matplotlib


def load_data(data_path):
    df = pd.read_csv(data_path)
    return df

education_label = {
    'Graduate School':1,
    'University':2,
    'High School':3, 
    'Others':4
    }

marriage_label = {
    'Married':1,
    'Single':2, 
    'Others':3
    }

last_month_pay_status_label = {
    'The Account Started That Month With A Zero Balance And Never Used Any Credit':-2,
    'The Account Had A Balance That Was Paid In Full':-1,
    'At Least The Minimum Payment Was Made, But The Entire Balance Was Not Paid':0,
    'Payment Delayed by 1 Month':1,
    'Payment Delayed By 2 Months':2,
    'Payment Delayed By 3 Months':3,
    'Payment Delayed By 4 Months':4,
    'Payment Delayed By 5 Months':5,
    'Payment Delayed By 6 Months':6,
    'Payment Delayed By 7 Months':7,
    'Payment Delayed By 8 Months':8,
    'Payment Delayed By 9 Months':9
    }

result_label = {
    'THE CUSTOMER IS LIKEY TO BE CLASSIFIED AS A DEFAULTER':1,
    'THE CUSTOMER IS LIKELY TO BE CLASSIFIED AS NOT A DEFAULTER':0
    }

def get_keys_vals(val, my_dict):
    for key, value in my_dict.items():
        if val == value:
            return key
        elif val == key:
            return value 

def load_prediction_model(model_file):
    loaded_model = joblib.load(open(os.path.join(model_file), 'rb'))
    return loaded_model

def app():
    st.title('CREDIT CARD DEFAULT PREDICTION APP')
    st.subheader('Developed By: Vedansh Kumar Kawadkar')

    limit_bal = st.number_input('ENTER THE BALANCE LIMIT')
    education = st.selectbox('ENTER THE FINAL EDUCATIONAL QUALIFICATION', tuple(education_label.keys()))
    marriage = st.selectbox('ENTER THE MARITAL STATUS', tuple(marriage_label.keys()))
    age = st.number_input('ENTER THE AGE')
    pay_1 = st.selectbox('PAYMENT DUES STATUS OF LAST MONTH', tuple(last_month_pay_status_label.keys()))
    
    st.text('ENTER THE CREDIT CARD BILL AMOUNT FOR THE LAST 6 MONTHS')

    bill_amt1 = st.number_input('ENTER THE BILL AMOUNT FOR LAST MONTH')
    bill_amt2 = st.number_input('ENTER THE BILL AMOUNT FOR SECOND LAST MONTH')
    bill_amt3 = st.number_input('ENTER THE BILL AMOUNT FOR THIRD LAST MONTH')
    bill_amt4 = st.number_input('ENTER THE BILL AMOUNT FOR FOURTH LAST MONTH')
    bill_amt5 = st.number_input('ENTER THE BILL AMOUNT FOR FIFTH LAST MONTH')
    bill_amt6 = st.number_input('ENTER THE BILL AMOUNT FOR SIXTH LAST MONTH')

    st.text('ENTER THE CREDIT CARD PAYED AMOUNT FOR THE LAST 6 MONTHS')

    pay_amt1 = st.number_input('ENTER THE PAYED AMOUNT FOR LAST MONTH')
    pay_amt2 = st.number_input('ENTER THE PAYED AMOUNT FOR SECOND LAST MONTH')
    pay_amt3 = st.number_input('ENTER THE PAYED AMOUNT FOR THIRD LAST MONTH')
    pay_amt4 = st.number_input('ENTER THE PAYED AMOUNT FOR FOURTH LAST MONTH')
    pay_amt5 = st.number_input('ENTER THE PAYED AMOUNT FOR FIFTH LAST MONTH')
    pay_amt6 = st.number_input('ENTER THE PAYED AMOUNT FOR SIXTH LAST MONTH')

    val_education = get_keys_vals(education, education_label)
    val_marriage = get_keys_vals(marriage, marriage_label)
    val_pay_1 = get_keys_vals(pay_1, last_month_pay_status_label)
    
    sample_data = [
        limit_bal, val_education, val_marriage, age, val_pay_1,
        bill_amt1, bill_amt2, bill_amt3, bill_amt4, bill_amt5, bill_amt6,
        pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6
    ]

    prep_data = np.array(sample_data).reshape(1, -1)

    predictor = load_prediction_model('H:\TechnoColabs Projects\model_joblib')
    prediction = predictor.predict(prep_data)
    result = get_keys_vals(prediction, result_label)


    if st.button('PREDICT'):
        if prediction == 1:
            st.error('THE CLIENT IS LIKELY TO BE A DEFAULTER')
        
        elif prediction == 0:
            st.success('THE CLIENT IS LIKELY TO BE A NON-DEFAULTER')

if __name__ == '__main__':
    app()

