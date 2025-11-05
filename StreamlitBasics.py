import streamlit as st
st.title("AI Powered Loan Eligibility Prediction System")

st.header("welcome to our website")
st.text("How are you")
st.markdown("bold text")
option=st.radio('select a option:',['M','E',"N"])
st.write("you selected",option)
city=st.selectbox('select your city:',['Hyderabad','Bangalore','Pune'])
st.write(f'selected {city}')
name =st.text_input("enter your name:")
st.write("helo:",name)