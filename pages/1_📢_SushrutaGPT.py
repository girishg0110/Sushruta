from langchain.chains import RetrievalQA
from langchain.embeddings.google_palm import GooglePalmEmbeddings
from langchain.llms import GooglePalm
from langchain.vectorstores import Chroma
import streamlit as st

GOOGLE_PALM_API_KEY = st.secrets["GOOGLE_PALM_API_KEY"]
embeddings = GooglePalmEmbeddings(google_api_key = GOOGLE_PALM_API_KEY)

docsearch = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
qa = RetrievalQA.from_chain_type(llm=GooglePalm(google_api_key=GOOGLE_PALM_API_KEY), chain_type="stuff", retriever=docsearch.as_retriever())

if "qa" not in st.session_state:
    st.session_state["qa"] = qa

st.title("SushrutaGPT 📢")
st.write("SushrutaGPT is a medical & legal chatbot that answers questions about the federal HIPAA healthcare protection act in the United States.")

if prompt := st.chat_input("e.g. What does HIPAA stand for?"):
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = st.session_state["qa"].run(prompt)
        message_placeholder.markdown(full_response)