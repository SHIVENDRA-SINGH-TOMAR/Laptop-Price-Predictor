import streamlit as st
import pickle
import numpy as np

#import the model
pipe = pickle.load(open('E:/Kaggle DataSets/Laptop Price/pipe.pkl', 'rb'))
lap_data = pickle.load(open('E:/Kaggle DataSets/Laptop Price/lap_data.pkl', 'rb'))

st.title("Laptop Price Predictor")

#brand
company = st.selectbox("Brand", lap_data['Company'].unique())

#type of laptop
type = st.selectbox("Type", lap_data["TypeName"].unique())

#Ram
ram = st.selectbox("RAM(in GB)", lap_data["Ram"].unique())

#Weight
weight = st.number_input("Weight of Laptop")

#TouchScreen
touchscreen = st.selectbox("TouchScreen", ["No", "Yes"])

#IPS
ips = st.selectbox("IPS", ["No", "Yes"])

#screen size
screen_size = st.number_input('Screen Size')

#resolution
res = st.selectbox("Screen Resolution", ['1920x1080', '1366x786', '1600x900', '3840x2160', '3200x1800', '2880x1600', '2560x1440', '1304x1440'])

#cpu
cpu = st.selectbox("CPU", lap_data["cpu brand"].unique())

#HDD
hdd = st.selectbox("HDD(in GB)", [0,128,256,512,1024,2048])

#SSD
ssd = st.selectbox("SSD(in GB)", [0,8,128,256,512,1024])

#GPU
gpu = st.selectbox("GPU", lap_data["Gpu brand"].unique())

#OS
os = st.selectbox("Operating System", lap_data["os"].unique())

if st.button("Predict Price"):
    #query 
    if touchscreen=="Yes":
        touchscreen=1
    else :
        touchscreen=0

    if ips=="Yes":
        ips=1
    else :
        ips =0

    xres = int(res.split('x')[0])
    yres = int(res.split('x')[1])
    ppi = ((xres**2) + (yres**2))**0.5/screen_size
    query = np.array([company, type, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os])
    
    new = query.reshape(1,12)
   
    st.title(round(np.exp(pipe.predict(new))[0], 2))