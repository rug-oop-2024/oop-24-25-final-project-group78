import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)
st.sidebar.success("Select a page above.")
# Commented because the decoder could not decode the emoji's in the readme
# st.markdown(open("README.md").read())
