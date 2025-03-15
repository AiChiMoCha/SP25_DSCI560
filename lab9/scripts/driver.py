import os
import sqlite3
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # Make sure your OPENAI_API_KEY is set
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

def build_knowledgebase(data_folder="../data", db_path="pdf_texts.db", faiss_index_path="faiss_index"):
    """
    1. Read text from PDF files and store it in a local database.
    2. Split the text into smaller chunks and generate embeddings which are stored in a vector database.
    3. Create a QA chain based on the knowledge base for answering questions.
    """
    # Create or connect to the database and create a table to store PDF texts
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS pdf_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            content TEXT
        )
    ''')
    conn.commit()
    
    # Loop through the PDFs in the data folder, extract text and store it in the database
    for file_name in os.listdir(data_folder):
        if file_name.lower().endswith(".pdf"):
            pdf_path = os.path.join(data_folder, file_name)
            text = ""
            with open(pdf_path, "rb") as f:
                reader = PdfReader(f)
                for page in reader.pages:
                    # If text cannot be extracted from a page, use an empty string
                    text += page.extract_text() or ""
            cursor.execute('INSERT INTO pdf_texts (filename, content) VALUES (?, ?)', (file_name, text))
            conn.commit()
            print(f"Stored {file_name} in the database")
    conn.close()
    
    # Read all PDF texts from the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, content FROM pdf_texts")
    rows = cursor.fetchall()
    conn.close()
    
    all_chunks = []
    # Split each PDF's text into chunks
    for filename, content in rows:
        text_splitter = CharacterTextSplitter(
            separator="\n",      # Split by newline if possible
            chunk_size=500,
            chunk_overlap=50,
            length_function=len
        )
        chunks = text_splitter.split_text(content)
        print(f"{filename} divided into {len(chunks)} chunks")
        # Optionally, add filename information to each chunk for tracking
        all_chunks.extend(chunks)
    print(f"A total of {len(all_chunks)} chunks were generated")
    
    # Create the vector database
    embeddings = OpenAIEmbeddings()  # Make sure your OPENAI_API_KEY is set
    vector_store = FAISS.from_texts(all_chunks, embeddings)
    vector_store.save_local(faiss_index_path)
    print("Vector database has been created and saved")
    
    # Create the QA chain. This follows a Retrieval Augmented Generation approach:
    # retrieve the closest text chunks based on the question and then let the LLM answer.
    llm = ChatOpenAI(temperature=0.2)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever()
    )
    
    return qa_chain

def conversation_driver():
    """
    Driver function:
    - Calls the above function to build the knowledge base QA chain.
    - Continuously gets user questions, passes them to the QA chain to get an answer, and prints the answer.
    - The program exits when the user types 'exit'.
    """
    qa_chain = build_knowledgebase()
    print("Knowledge base QA chain is ready. Enter your question (type 'exit' to quit):")
    while True:
        user_question = input("Your question: ")
        if user_question.strip().lower() == "exit":
            print("Exiting the program.")
            break
        answer = qa_chain.invoke({"query": user_question})
        print("Answer:", answer)

if __name__ == "__main__":
    conversation_driver()
