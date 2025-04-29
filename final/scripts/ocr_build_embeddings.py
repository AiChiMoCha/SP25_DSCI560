"""
Enhanced TXT text extraction script for document embeddings.
Modified from PDF extraction script to work with TXT files directly.
"""

import os
import sqlite3
import argparse
from pathlib import Path
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def extract_text_from_txt(txt_path):
    """
    Extract text from a TXT file
    
    Args:
        txt_path: Path to the TXT file
        
    Returns:
        Extracted text as string
    """
    try:
        with open(txt_path, "r", encoding="utf-8") as f:
            text = f.read()
        return text
    except UnicodeDecodeError:
        # Try with different encodings if utf-8 fails
        try:
            with open(txt_path, "r", encoding="latin-1") as f:
                text = f.read()
            print(f"  Note: Opened {txt_path} with latin-1 encoding")
            return text
        except Exception as e:
            print(f"  Error reading file with latin-1 encoding: {str(e)}")
            return ""
    except Exception as e:
        print(f"  Error reading file: {str(e)}")
        return ""

def extract_text_from_txts(docs_folder, db_path):
    """
    Extract text from TXT files and store in a SQLite database
    
    Args:
        docs_folder: Path to folder containing TXT documents
        db_path: Path to SQLite database to store extracted text
    """
    print(f"\n1. Extracting text from TXT files in {docs_folder}...")
    
    # Create or connect to the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Create table if it doesn't exist
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS txt_texts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            content TEXT
        )
    ''')
    
    # Clear existing data
    cursor.execute("DELETE FROM txt_texts")
    conn.commit()
    
    # Process each TXT in the docs folder
    txt_count = 0
    for file_name in os.listdir(docs_folder):
        if file_name.lower().endswith(".txt"):
            txt_path = os.path.join(docs_folder, file_name)
            
            try:
                print(f"  Processing: {file_name}")
                
                # Extract text from TXT file
                text = extract_text_from_txt(txt_path)
                
                # Check if we got any text
                if not text or len(text.strip()) == 0:
                    print(f"  ⚠ Warning: No text extracted from {file_name}")
                
                # Store the extracted text in the database
                cursor.execute('INSERT INTO txt_texts (filename, content) VALUES (?, ?)', 
                              (file_name, text))
                conn.commit()
                
                print(f"  ✓ Completed: {file_name} ({len(text)} characters)")
                txt_count += 1
                
            except Exception as e:
                print(f"  ✗ Error processing {file_name}: {str(e)}")
    
    conn.close()
    print(f"\nExtracted text from {txt_count} TXT files and stored in {db_path}")

def create_chunks_and_embeddings(db_path, faiss_index_path, chunk_size=1000, chunk_overlap=100):
    """
    Create text chunks and embeddings from the stored TXT texts
    
    Args:
        db_path: Path to SQLite database with stored text
        faiss_index_path: Path to store the FAISS vector index
        chunk_size: Size of text chunks
        chunk_overlap: Overlap between chunks
    """
    print(f"\n2. Creating text chunks and embeddings...")
    
    # Check if OpenAI API key is set
    if not os.environ.get("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable is not set.")
        print("Please set it in your .env file or export it to your environment.")
        return False
    
    # Read all TXT texts from the database
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, content FROM txt_texts")
    rows = cursor.fetchall()
    conn.close()
    
    if not rows:
        print("No TXT texts found in the database. Run extraction first.")
        return False
    
    all_docs = []
    file_chunks = {}  # Track chunks per file
    
    # Split each TXT's text into chunks
    print(f"  Using chunk size: {chunk_size}, overlap: {chunk_overlap}")
    for filename, content in rows:
        # Skip if content is empty
        if not content or len(content.strip()) == 0:
            print(f"  • {filename}: skipped (no content)")
            continue
            
        # Special handling for very large content
        very_large = len(content) > 5000000  # 5MB threshold
        if very_large:
            print(f"  • {filename}: very large file ({len(content)/1000000:.1f}MB), using larger chunks")
            
        # Try to create text splitter with appropriate settings
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=chunk_size * (5 if very_large else 1),  # Larger chunks for large files
            chunk_overlap=chunk_overlap * (2 if very_large else 1),
            length_function=len
        )
        
        try:
            chunks = text_splitter.split_text(content)
            
            # Create document objects with proper metadata for each chunk
            for chunk in chunks:
                # Skip very small chunks
                if len(chunk.strip()) < 50:
                    continue
                    
                all_docs.append({
                    "content": chunk,
                    "metadata": {"source": filename}
                })
            
            file_chunks[filename] = len(chunks)
            print(f"  • {filename}: created {len(chunks)} chunks")
            
        except Exception as e:
            print(f"  ✗ Error chunking {filename}: {str(e)}")
    
    total_chunks = len(all_docs)
    print(f"\n  Total chunks across all documents: {total_chunks}")
    
    if total_chunks == 0:
        print("Error: No valid chunks created. Cannot proceed with embeddings.")
        return False
    
    # Create the vector database
    print(f"\n3. Generating embeddings with OpenAI API...")
    try:
        embeddings = OpenAIEmbeddings()
        
        # Create FAISS index from documents with metadata
        texts = [doc["content"] for doc in all_docs]
        metadatas = [doc["metadata"] for doc in all_docs]
        
        # Process in batches
        batch_size = 100
        vector_store = None
        
        for i in range(0, len(texts), batch_size):
            end_idx = min(i + batch_size, len(texts))
            print(f"  Processing batch {i//batch_size + 1}/{(len(texts) + batch_size - 1)//batch_size}: chunks {i} to {end_idx-1}")
            
            batch_texts = texts[i:end_idx]
            batch_meta = metadatas[i:end_idx]
            
            batch_vs = FAISS.from_texts(batch_texts, embeddings, metadatas=batch_meta)
            
            if vector_store is None:
                vector_store = batch_vs
            else:
                vector_store.merge_from(batch_vs)
        
        # Save the vector store locally
        vector_store.save_local(faiss_index_path)
        print(f"\n✓ Vector database successfully created and saved to {faiss_index_path}")
        print(f"  Contains {total_chunks} chunks from {len(file_chunks)} TXT documents")
        return True
        
    except Exception as e:
        print(f"\n✗ Error creating embeddings: {str(e)}")
        return False

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Build embeddings for financial TXT documents')
    parser.add_argument('--docs', type=str, default='../docs',
                      help='Path to the folder containing TXT files')
    parser.add_argument('--db', type=str, default='finance_texts.db',
                      help='Path to the SQLite database file')
    parser.add_argument('--index', type=str, default='finance_faiss_index',
                      help='Path to store the FAISS index')
    parser.add_argument('--chunk-size', type=int, default=1000,
                      help='Size of text chunks')
    parser.add_argument('--chunk-overlap', type=int, default=100,
                      help='Overlap between chunks')
    parser.add_argument('--skip-extraction', action='store_true',
                      help='Skip TXT text extraction and use existing database')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Financial TXT Embedding Builder")
    print("=" * 80)
    
    # Step 1: Extract text from TXT files
    if not args.skip_extraction:
        extract_text_from_txts(args.docs, args.db)
    else:
        print("\n1. Skipping TXT text extraction (using existing database)")
    
    # Step 2: Create chunks and embeddings
    success = create_chunks_and_embeddings(
        args.db, 
        args.index,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap
    )
    
    if success:
        print("\n✓ Embedding process complete! You can now run the finance_chatbot.py script.")
    else:
        print("\n✗ Embedding process failed. Please check the errors above.")

if __name__ == "__main__":
    main()