import streamlit as st

st.set_page_config(
    page_title="Sushruta",
    page_icon="🧪",
)

st.write("# Welcome to Sushruta! 👋")

st.sidebar.success("Visit SushrutaGPT first!")

st.markdown(
    """
    **Sushruta** documents our journey understanding, analyzing, and finally training a model on the UCI AIDS dataset.
    
    We have three goals: 
    
    * to make information on **healthcare privacy** more widely known and accessible 📢⚖️
    * to **improve AIDS diagnosis outcomes** with machine learning 🧠🤖
    * to provide **interpretability** behind the diagnosis process ❤️🖼️

    Take a look at the sidebar to get started!
"""
)