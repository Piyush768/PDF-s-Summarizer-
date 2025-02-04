import os
import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import time

# Load environment variables
load_dotenv()

# Ensure OpenAI API key is set
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    st.error("⚠️ OpenAI API key is missing! Please set it in the .env file.")
    st.stop()

# Set the OpenAI API key environment variable
os.environ['OPENAI_API_KEY'] = api_key

# Streamlit app title
st.title("AI-driven Insights from Documents 📈")

# Initialize ChatOpenAI for GPT-3.5 Turbo (Chat Model)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.9, max_tokens=500)

# Define the prompt template
prompt = """
Answer the questions based on the provided context only.
Please provide the most accurate response based on the question.
<context>
{context}
</context>
Question: {input}
"""

# Function to load and process documents
def vector_embedding():
    if "vectors" not in st.session_state:
        with st.spinner("Embedding documents..."):
            # Initialize OpenAI embeddings and document loader
            st.session_state.embeddings = OpenAIEmbeddings()

            # Load the PDF directly (instead of directory loader)
            st.session_state.loader = PyPDFLoader("/Users/piyushpatil/Desktop/Piyush/Projects/Projects/Sumarization with Langchain/content/NIPS-2017-attention-is-all-you-need-Paper.pdf")
            st.session_state.docs = st.session_state.loader.load()

            # Debug: print the number of documents loaded
            print(f"Loaded {len(st.session_state.docs)} documents.")

            # Split documents into chunks
            st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:20])

            # Debug: Check how many chunks were created
            print(f"Created {len(st.session_state.final_documents)} chunks.")

            # Check if documents were split correctly
            if not st.session_state.final_documents:
                st.error("⚠️ No documents were created after splitting! Check your input data.")
                st.stop()

            # Extract only the text content from the Document objects for FAISS
            texts = [doc.page_content for doc in st.session_state.final_documents]

            # Add metadata to each document chunk (e.g., adding the 'source' key)
            for i, doc in enumerate(st.session_state.final_documents):
                doc.metadata['source'] = f"Document-{i+1}"

            # Generate embeddings for the documents
            embeddings = st.session_state.embeddings.embed_documents(texts)

            # Check if embeddings were generated
            if not embeddings:
                st.error("⚠️ No embeddings generated! Check your embedding process.")
                st.stop()

            # Create FAISS index from the document chunks with metadata
            st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        st.success("Vector Store DB is Ready")

# User input for question
prompt1 = st.text_input("Enter your question from the documents:")

# Button to trigger document embedding
if st.button("Embed Documents"):
    vector_embedding()

# Process the question and generate response
if prompt1:
    try:
        # Prepare the document chain for answering questions
        retriever = st.session_state.vectors.as_retriever()
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=retriever)
        
        # Start time for response generation
        start = time.process_time()
        
        # Generate the response using the retrieval chain
        response = chain({"question": prompt1}, return_only_outputs=True)
        
        # Show the response and time taken
        st.write(f"Response time: {time.time() - start:.2f} seconds")
        st.write(response.get("answer", "⚠️ No answer found."))

        # Show the relevant documents (sources) only if they are found
        if 'sources' in response and response["sources"]:
            sources = response["sources"]
            if isinstance(sources, list):  # Ensure it's a list
                with st.expander("Document Similarity Search"):
                    for i, doc in enumerate(sources):
                        st.write(doc['page_content'])  # Accessing page_content of the source
                        st.write("--------------------------------")
        else:
            st.write("No sources to display.")
    except Exception as e:
        st.error(f"An error occurred: {e}")
