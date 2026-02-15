import streamlit as st 
import pandas as pd 

st.title("Upload Dataset")

file = st.file_uploader("Upload your dataset", type=["csv"])

if file:
    df = pd.read_csv(file)
    st.session_state["df"] = df

# if "df" in st.session_state: