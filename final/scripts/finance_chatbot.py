"""
This script implements a chatbot that answers questions based on financial PDF documents.
It uses pre-built embeddings to retrieve relevant information and generate answers.

First run build_embeddings.py to create the required embeddings before using this script.
"""

import os
import argparse
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

class FinancialChatbot:
    def __init__(self, index_path="finance_faiss_index", model_name=None):
        """
        Initialize the financial chatbot
        
        Args:
            index_path: Path to the pre-built FAISS index
            model_name: Name of the OpenAI model to use (defaults to environment variable or gpt-4o-mini)
        """
        self.index_path = index_path
        self.model_name = model_name or os.environ.get("MODEL_NAME", "gpt-4o-mini")
        
        # Check if FAISS index exists
        if not os.path.exists(index_path):
            raise FileNotFoundError(
                f"FAISS index not found at {index_path}. " 
                f"Please run build_embeddings.py first."
            )
        
        # Check if OpenAI API key is set
        if not os.environ.get("OPENAI_API_KEY"):
            raise ValueError(
                "OPENAI_API_KEY environment variable is not set. "
                "Please set it in your .env file or export it to your environment."
            )
            
        # Load the vector store
        self._load_vector_store()
        
        # Create the retrieval chain
        self._create_retrieval_chain()
        
    def _load_vector_store(self):
        """Load the pre-built FAISS vector store"""
        try:
            embeddings = OpenAIEmbeddings()
            self.vector_store = FAISS.load_local(
                self.index_path, 
                embeddings,
                allow_dangerous_deserialization=True
            )
            return True
        except Exception as e:
            raise RuntimeError(f"Error loading vector store: {str(e)}")
        
    def _create_retrieval_chain(self):
        """Create the retrieval chain for answering questions"""
        # Initialize the language model
        llm = ChatOpenAI(
            temperature=0.1,  # Low temperature for more factual responses
            model=self.model_name,
        )
        
        # Use a simpler RetrievalQA chain without memory
        self.qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": 5}  # Retrieve top 5 most relevant chunks
            ),
            return_source_documents=True
        )
    
    def answer_question(self, question):
        """
        Answer a question based on the financial texts
        
        Args:
            question: The user's question
            
        Returns:
            A tuple containing (answer, source_documents)
        """
        try:
            # Get the answer from the retrieval chain
            result = self.qa_chain.invoke({"query": question})
            
            # Extract the answer and source documents
            answer = result.get('result', "I couldn't find an answer to that question.")
            source_docs = result.get('source_documents', [])
            
            # Extract source information from document metadata
            sources = []
            for doc in source_docs:
                if hasattr(doc, 'metadata') and 'source' in doc.metadata:
                    source_file = doc.metadata['source']
                    sources.append(source_file)
            
            # Remove duplicates while preserving order
            unique_sources = []
            for source in sources:
                if source not in unique_sources:
                    unique_sources.append(source)
            
            return answer, unique_sources
        except Exception as e:
            print(f"Error in answer_question: {str(e)}")
            raise
    
    def run_test(self):
        """Run a quick test to verify source attribution"""
        print("\nRunning source attribution test...")
        # List of test questions that should target different books
        test_questions = [
            "What are candlestick patterns?",     # Should find in Steve Nison's book
            "Explain the Black-Scholes model",    # Should find in Hull's book
            "What is technical analysis?",        # Should find in Murphy's book
            "How do you calculate option delta?", # Should find in Natenberg's book
            "What is volatility skew?"            # Could be in multiple books
        ]
        
        for question in test_questions:
            print(f"\nTest Question: {question}")
            try:
                answer, sources = self.answer_question(question)
                print("Sources found:")
                for source in sources:
                    print(f"• {source}")
            except Exception as e:
                print(f"Error: {str(e)}")
            print("-" * 40)
        
        print("\nTest complete. If you see different sources for different questions, attribution is working correctly.\n")

def run_interactive_chatbot(chatbot):
    """Run an interactive chatbot session"""
    print("\n📚 Financial Chatbot ready! Ask me questions about:")
    print("• Technical Analysis (John Murphy)")
    print("• Option Pricing and Volatility (Sheldon Natenberg)")
    print("• Options, Futures, and Other Derivatives (John Hull)")
    print("• Japanese Candlestick Charting (Steve Nison)")
    print("\nType 'exit' or 'quit' to end the session.\n")
    
    # Interactive loop
    while True:
        user_input = input("Question: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        # Get the answer
        try:
            answer, sources = chatbot.answer_question(user_input)
            
            print("\n" + "=" * 80)
            print(answer)
            print("-" * 80)
            
            if sources:
                print("Sources:")
                for source in sources:
                    print(f"• {source}")
            else:
                print("No specific sources found for this answer.")
            
            print("=" * 80 + "\n")
            
        except Exception as e:
            print(f"\nError generating answer: {str(e)}")

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Financial Chatbot using PDF documents')
    parser.add_argument('--index', type=str, default='finance_faiss_index',
                      help='Path to the FAISS index')
    parser.add_argument('--model', type=str, default=None,
                      help='OpenAI model to use (default: uses MODEL_NAME from .env or gpt-4o-mini)')
    parser.add_argument('--test', action='store_true',
                      help='Run a quick test to verify source attribution')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Financial Chatbot")
    print("=" * 80)
    
    try:
        # Initialize the chatbot
        chatbot = FinancialChatbot(
            index_path=args.index,
            model_name=args.model
        )
        
        # If in test mode, run a verification to check source attribution
        if args.test:
            chatbot.run_test()
        else:
            run_interactive_chatbot(chatbot)
            
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("\nPlease run build_embeddings.py first to create the necessary files.")
    except ValueError as e:
        print(f"\nError: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

if __name__ == "__main__":
    main()