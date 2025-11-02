import streamlit as st
import pickle
import joblib
import numpy as np
import pandas as pd


pipe = joblib.load(open('pipe.pkl', 'rb'))
df = joblib.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# brand
company=st.selectbox('Brand',df['Company'].unique())

# type of laptop
type=st.selectbox('Type',df['TypeName'].unique())

# Ram
ram=st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight=st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen=st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips=st.selectbox('IPS',['No','Yes'])

# Retina
retina=st.selectbox('Retina',['No','Yes'])

# Screen Size
screen_size=st.number_input('Screen Size')

# Resolution
resolution=st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# CPU
cpu=st.selectbox('CPU',df['cpu_brand'].unique())

# Clock Speed
clock_speed=st.number_input('Clock Speed')

# HDD

hdd=st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])
ssd=st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])
gpu=st.selectbox('GPU',df['Gpu_brand'].unique())
os=st.selectbox('OS',df['OS'].unique())

if st.button('Predict Price'):
    ppi=None
    if touchscreen=='Yes':
        touchscreen=1
    else:
        touchscreen=0
        
    if ips=='Yes':
        ips=1
    else:
        ips=0
        
    if retina=='Yes':
        retina=1
    else:
        retina=0
    
    X_res=int(resolution.split('x')[0])
    Y_res=int(resolution.split('x')[1])
    ppi=((X_res**2)+(Y_res**2))**0.5/screen_size

    query_data = [[company, type, ram, weight, touchscreen, ips, retina, ppi, cpu, clock_speed, hdd, ssd, gpu, os]]
    columns = ['Company', 'TypeName', 'Ram', 'Weight', 'Touchscreen', 'Ips', 'Retina', 'PPI', 'cpu_brand', 'Clock_speed', 'HDD', 'SSD', 'Gpu_brand', 'OS']
    query = pd.DataFrame(query_data, columns=columns)

    # Predict price
    pred = np.exp(pipe.predict(query)[0])
    st.title("The Predicted Price of this configuration is ₹" + str(int(pred)))

    
