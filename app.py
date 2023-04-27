from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import streamlit as st
import os
from PIL import Image

os.environ["OPENAI_API_KEY"] = "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"

# Define Streamlit app
def app():
    st.title("Welcome to the OceanML Demo App")
    image = Image.open("A3.png")
    st.sidebar.image(image, use_column_width=True )
    # Add a file uploader to the app
    uploaded_file = st.file_uploader("Upload a research paper (PDF)", type="pdf")

    # Read the text from the uploaded PDF file
    if uploaded_file is not None:
        pdf_reader = PdfReader(uploaded_file)
        raw_text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            raw_text += page.extract_text()

        # Split the text into smaller chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
        )
        texts = text_splitter.split_text(raw_text)

        # Download embeddings from OpenAI
        embeddings = OpenAIEmbeddings()

        # Create a vector store for the text chunks
        vector_store = FAISS.from_texts(texts, embeddings)

        # Load the question-answering chain
        chain = load_qa_chain(OpenAI(), chain_type="stuff")

        # Get a question from the user
        question = st.text_input("Enter a question")

        # Search for similar text chunks and get an answer to the question
        docs = vector_store.similarity_search(question)
        answer = chain.run(input_documents=docs, question=question)

        # Display the answer to the user
        st.subheader("Answer:")
        st.write(answer)

# Run the app
if __name__ == "__main__":
    app()
