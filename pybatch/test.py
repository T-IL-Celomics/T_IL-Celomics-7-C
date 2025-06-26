import streamlit as st

st.write("YEP")
a = st.radio("Choose", ["abc", "bcd", "def"], index=1)
st.success(a)