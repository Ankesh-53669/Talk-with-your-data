# Imports
import streamlit as st
from PyPDF2 import PdfReader
from docx import Document

from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()
# Retrieve the Google API key from the environment variables
os.getenv("GOOGLE_API_KEY")
# Configure the Google Generative AI with the API key
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

def get_text_from_pdf(pdf_docs):
    """
    Extract text from PDF documents and concatenate them.
    """
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_from_doc(doc_docs):
    """
    Extract text from DOC documents and concatenate them.
    """
    text = ""
    for doc_file in doc_docs:
        doc = Document(doc_file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    return text


def get_text_chunks(text):
    """
    Split text into chunks using RecursiveCharacterTextSplitter.
    """
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    """
    Generate embeddings and create a FAISS vector store from text chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

def get_conversational_chain():
    """
    Create a conversational chain for question answering.
    """
    prompt_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details, if the answer is not in
    provided context just say, "answer is not available in the context", don't provide the wrong answer\n\n
    Context:\n {context}?\n
    Question: \n{question}\n

    Answer:
    """

    # Initialize ChatGoogleGenerativeAI model
    model = ChatGoogleGenerativeAI(model="gemini-1.0-pro", temperature=0)

    # Define conversation buffer memory
    buffer_memory = ConversationBufferMemory(memory_key="gemini_conversation")

    # Define a prompt template for user interactions
    prompt = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"],
        memory=buffer_memory  # Include the buffer memory
    )

    # Load the question-answering chain
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(user_question):
    """
    Process user input, perform similarity search, and generate a response using the conversational chain.
    """
    # Load embeddings for Google Generative AI
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    
    # Load the FAISS vector store from the saved index
    new_db = FAISS.load_local("faiss_index", embeddings)
    # Perform similarity search based on user's question
    docs = new_db.similarity_search(user_question)

    # Get the conversational chain for question answering
    chain = get_conversational_chain()
    
    # Generate a response using the chain
    response = chain(
        {"input_documents": docs, "question": user_question},
        return_only_outputs=True
    )

    return response["output_text"]

def main():
    """
    Streamlit app to upload PDFs/DOCs, process them, and interact with the chatbot.
    """
    # Configure Streamlit page settings
    st.set_page_config(
        page_title="Chatbot",
        
    )

    # Sidebar for uploading files
    with st.sidebar:
        st.title("Menu:")
        # File uploader for PDFs and DOCs
        uploaded_files = st.file_uploader(
            "Upload PDF or DOC Files", accept_multiple_files=True, type=["pdf", "docx"])
        # Button to initiate processing
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                # Extract text from uploaded files, split into chunks, and create a vector store
                raw_text = ""
                for uploaded_file in uploaded_files:
                    if uploaded_file.type == "application/pdf":
                        raw_text += get_text_from_pdf([uploaded_file]) + "\n"
                    elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        raw_text += get_text_from_doc([uploaded_file]) + "\n"
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")

    # Main content area for displaying chat messages
    st.title("Chatbot")
    
    # Initialize session state for chat messages
    if "messages" not in st.session_state:
      st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

    # Display existing chat messages
    for msg in st.session_state.messages:
      st.chat_message(msg["role"]).write(msg["content"])

    # Get user input from the chat input box
    if prompt := st.chat_input():
        # Add user input to session state messages
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        # Generate a response based on user input
        response = user_input(prompt)
    
        # Add assistant's response to session state messages
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()
