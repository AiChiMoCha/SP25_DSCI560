"""
This script builds a GraphRAG knowledge graph from financial PDF documents.
It extracts entities, relationships, and creates a structured knowledge graph
that can be used for more accurate retrieval and answering of questions.

Requirements:
- pip install graphrag pdfplumber
- Set GRAPHRAG_API_KEY in your environment or .env file
"""

import os
import argparse
import pdfplumber
import glob
from dotenv import load_dotenv
from graphrag.api import GraphRAGIndexer
from graphrag.config import IndexerConfig

# Load environment variables from .env file
load_dotenv()

def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file using pdfplumber"""
    try:
        text = ""
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() + "\n\n"
        return text
    except Exception as e:
        print(f"Error extracting text from {pdf_path}: {str(e)}")
        return ""

def prepare_documents(input_dir, output_dir):
    """
    Prepare documents for GraphRAG indexing
    
    Args:
        input_dir: Directory containing PDF documents
        output_dir: Directory where text files will be written
    
    Returns:
        List of paths to prepared text files
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all PDF files in the input directory
    pdf_files = glob.glob(os.path.join(input_dir, "*.pdf"))
    
    if not pdf_files:
        raise ValueError(f"No PDF files found in {input_dir}")
    
    # Process each PDF file
    text_files = []
    for pdf_path in pdf_files:
        print(f"Processing {pdf_path}...")
        
        # Extract text from PDF
        text = extract_text_from_pdf(pdf_path)
        
        if not text:
            print(f"Warning: No text extracted from {pdf_path}")
            continue
        
        # Create output file path
        file_name = os.path.basename(pdf_path).replace(".pdf", ".txt")
        output_path = os.path.join(output_dir, file_name)
        
        # Write text to file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        text_files.append(output_path)
        print(f"  → Created {output_path}")
    
    return text_files

def build_graphrag_index(data_root, input_dir, prepared_dir):
    """
    Build a GraphRAG index from prepared documents
    
    Args:
        data_root: Directory where GraphRAG will store its data
        input_dir: Directory containing PDF documents
        prepared_dir: Directory where text files are prepared
    """
    # Prepare documents
    print(f"Preparing documents from {input_dir}...")
    text_files = prepare_documents(input_dir, prepared_dir)
    
    if not text_files:
        raise ValueError("No documents were prepared")
    
    # Check if API key is set
    if not os.environ.get("GRAPHRAG_API_KEY"):
        raise ValueError(
            "GRAPHRAG_API_KEY environment variable is not set. "
            "Please set it in your .env file or export it to your environment."
        )
    
    # Create GraphRAG data directory
    os.makedirs(data_root, exist_ok=True)
    
    # Initialize GraphRAG
    print(f"Initializing GraphRAG in {data_root}...")
    try:
        # Create a configuration for the indexer
        config = IndexerConfig(
            # Customize chunking settings if needed
            chunker_config={
                "chunk_size": 1000,  # Characters per chunk
                "chunk_overlap": 200,  # Overlap between chunks
            },
            # Customize embedding settings if needed
            embedder_config={
                "batch_size": 10,  # Number of texts to embed in a batch
            }
        )
        
        # Create an indexer instance
        indexer = GraphRAGIndexer(
            root=data_root,
            config=config
        )
        
        # Start the indexing process
        print("Starting GraphRAG indexing process...")
        print("This may take some time depending on the size of your documents.")
        
        # Run the indexing pipeline
        indexer.run(input_files=text_files)
        
        print(f"Successfully built GraphRAG index in {data_root}/output")
        return True
    except Exception as e:
        print(f"Error building GraphRAG index: {str(e)}")
        return False

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Build GraphRAG Knowledge Graph from Financial PDFs')
    parser.add_argument('--pdfs', type=str, default='financial_pdfs',
                      help='Directory containing financial PDF documents')
    parser.add_argument('--output', type=str, default='finance_graphrag_data',
                      help='Directory where GraphRAG will store its data')
    parser.add_argument('--prepared', type=str, default='prepared_texts',
                      help='Directory for intermediate prepared text files')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Financial GraphRAG Index Builder")
    print("=" * 80)
    
    try:
        success = build_graphrag_index(
            data_root=args.output,
            input_dir=args.pdfs,
            prepared_dir=args.prepared
        )
        
        if success:
            print("\nGraphRAG knowledge graph successfully built!")
            print(f"You can now run the chatbot script to query this knowledge graph.")
        else:
            print("\nFailed to build GraphRAG knowledge graph.")
    except Exception as e:
        print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()