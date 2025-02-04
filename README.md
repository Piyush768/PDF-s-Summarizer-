# PDF-s-Summarizer-

**PDF-s-Summarizer-** is a web application built using **Streamlit** and **LangChain** that allows users to extract key insights and answers from PDF documents using **OpenAI’s GPT-3.5** model. The app processes the document, splits it into manageable chunks, creates embeddings, and uses **FAISS** for fast retrieval to answer questions based on the document’s content.

---

### **Features**
- **Upload PDF Documents**: The app processes PDF documents and extracts content.
- **Question Answering**: Ask questions based on the document’s content, and get AI-powered answers.
- **Fast Retrieval**: Utilizes **FAISS** for efficient document retrieval based on context.
- **Text Splitting**: The document is split into smaller chunks to improve content processing.

---

### **How It Works**

1. **PDF Document Loading**: The app loads the document (NIPS 2017 paper in the example).
2. **Text Processing**: The document is split into smaller chunks for better handling.
3. **Embeddings Creation**: **OpenAI Embeddings** are used to convert document chunks into vector representations.
4. **FAISS Indexing**: The chunks are indexed using **FAISS** for fast similarity search.
5. **Question Answering**: Ask questions, and the AI answers based on the context of the document using **OpenAI’s GPT-3.5** model.

---

### **Technologies Used**
- **Streamlit**: For building the interactive UI.
- **LangChain**: For processing documents, embeddings, and retrieval.
- **OpenAI GPT-3.5**: For generating answers based on the content.
- **FAISS**: For fast document similarity search.
- **PyPDFLoader**: For loading PDF documents.

---

### **Setup & Usage**

1. **Install Dependencies**:  
   Create a virtual environment and install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. **Set OpenAI API Key**:  
   Create a `.env` file in the root directory with the following:
   ```env
   OPENAI_API_KEY=your-api-key-here
   ```

3. **Run the Application**:  
   Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. **Interact with the App**:  
   - Enter the **PDF document path** and click **Embed Documents** to process the content.
   - Ask any **question** related to the document, and the AI will provide an answer.

---
