"""
This script implements a chatbot that answers questions based on financial PDF documents.
It uses Microsoft's GraphRAG framework for improved retrieval over traditional RAG methods.
GraphRAG extracts entities, relationships, and structures knowledge as a graph for more precise answers.

First run build_graphrag_index.py to create the required knowledge graph before using this script.
"""

import os
import argparse
from dotenv import load_dotenv
import pandas as pd
from graphrag.api import GraphRAGQueryEngine
from graphrag.config import QueryEngineConfig
from graphrag.schema import QueryMethod

# Load environment variables from .env file
load_dotenv()

class FinancialGraphRAGChatbot:
    def __init__(self, data_root="finance_graphrag_data", model_name=None):
        """
        Initialize the financial chatbot powered by GraphRAG
        
        Args:
            data_root: Path to the GraphRAG data directory with indexed documents
            model_name: Name of the LLM model to use (defaults to environment variable or Azure OpenAI)
        """
        self.data_root = data_root
        self.model_name = model_name or os.environ.get("MODEL_NAME")
        
        # Check if GraphRAG index exists
        if not os.path.exists(os.path.join(data_root, "output")):
            raise FileNotFoundError(
                f"GraphRAG index not found at {data_root}/output. " 
                f"Please run build_graphrag_index.py first."
            )
        
        # Check if API key is set
        if not os.environ.get("GRAPHRAG_API_KEY"):
            raise ValueError(
                "GRAPHRAG_API_KEY environment variable is not set. "
                "Please set it in your .env file or export it to your environment."
            )
            
        # Initialize the query engine
        self._initialize_query_engine()
        
    def _initialize_query_engine(self):
        """Initialize the GraphRAG query engine"""
        try:
            # Create a configuration for the query engine
            config = QueryEngineConfig(
                # Using Local for specific questions about documents or entities
                # Use Global for more general questions that need holistic understanding
                default_method=QueryMethod.LOCAL
            )
            
            # Create a query engine instance
            self.query_engine = GraphRAGQueryEngine(
                root=self.data_root,
                config=config
            )
            
            print("GraphRAG query engine initialized successfully.")
            return True
        except Exception as e:
            raise RuntimeError(f"Error initializing GraphRAG query engine: {str(e)}")
    
    def answer_question(self, question, method=None):
        """
        Answer a question based on the financial knowledge graph
        
        Args:
            question: The user's question
            method: QueryMethod.LOCAL or QueryMethod.GLOBAL (if None, uses default from config)
            
        Returns:
            A tuple containing (answer, sources)
        """
        try:
            # Determine which method to use
            query_method = method or QueryMethod.LOCAL
            
            # Get the answer from the query engine
            query_result = self.query_engine.query(
                query=question,
                method=query_method
            )
            
            # Extract the answer and source documents
            answer = query_result.answer if hasattr(query_result, 'answer') else query_result.response
            
            # Extract source information
            sources = []
            if hasattr(query_result, 'sources'):
                for source in query_result.sources:
                    if source not in sources:
                        sources.append(source)
            
            return answer, sources
        except Exception as e:
            print(f"Error in answer_question: {str(e)}")
            raise
    
    def run_test(self):
        """Run a quick test to verify source attribution and different query modes"""
        print("\nRunning source attribution test...")
        # List of test questions that should target different aspects
        test_questions = [
            ("What are candlestick patterns?", QueryMethod.LOCAL),     # Specific question
            ("What are the main themes in financial derivatives?", QueryMethod.GLOBAL),  # Broad question
            ("Explain relationships between options and volatility", QueryMethod.GLOBAL),
            ("How do you calculate option delta?", QueryMethod.LOCAL), 
            ("What is volatility skew?", QueryMethod.LOCAL)            
        ]
        
        for question, method in test_questions:
            print(f"\nTest Question: {question}")
            print(f"Using method: {method}")
            try:
                answer, sources = self.answer_question(question, method)
                print(f"Answer: {answer[:100]}...")  # Show beginning of answer
                print("Sources found:")
                for source in sources:
                    print(f"• {source}")
            except Exception as e:
                print(f"Error: {str(e)}")
            print("-" * 40)
        
        print("\nTest complete. If you see different answers for different methods, GraphRAG is working correctly.\n")

def run_interactive_chatbot(chatbot):
    """Run an interactive chatbot session"""
    print("\n📊 Financial GraphRAG Chatbot ready! Ask me questions about:")
    print("• Technical Analysis (John Murphy)")
    print("• Option Pricing and Volatility (Sheldon Natenberg)")
    print("• Options, Futures, and Other Derivatives (John Hull)")
    print("• Japanese Candlestick Charting (Steve Nison)")
    print("\nType 'exit' or 'quit' to end the session.")
    print("Type 'global:' before your question for broad, theme-based answers.")
    print("Type 'local:' before your question for specific, factual answers (default).\n")
    
    # Interactive loop
    while True:
        user_input = input("Question: ")
        
        if user_input.lower() in ['exit', 'quit', 'bye']:
            print("Goodbye!")
            break
        
        # Determine which method to use based on prefix
        method = QueryMethod.LOCAL  # Default
        if user_input.lower().startswith("global:"):
            method = QueryMethod.GLOBAL
            user_input = user_input[7:].strip()  # Remove prefix
        elif user_input.lower().startswith("local:"):
            user_input = user_input[6:].strip()  # Remove prefix
        
        # Get the answer
        try:
            answer, sources = chatbot.answer_question(user_input, method)
            
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
    parser = argparse.ArgumentParser(description='Financial Chatbot using GraphRAG')
    parser.add_argument('--root', type=str, default='finance_graphrag_data',
                      help='Path to the GraphRAG data directory')
    parser.add_argument('--model', type=str, default=None,
                      help='LLM model to use (default: uses MODEL_NAME from .env)')
    parser.add_argument('--test', action='store_true',
                      help='Run a quick test to verify functionality')
    args = parser.parse_args()
    
    print("=" * 80)
    print("Financial GraphRAG Chatbot")
    print("=" * 80)
    
    try:
        # Initialize the chatbot
        chatbot = FinancialGraphRAGChatbot(
            data_root=args.root,
            model_name=args.model
        )
        
        # If in test mode, run a verification to check functionality
        if args.test:
            chatbot.run_test()
        else:
            run_interactive_chatbot(chatbot)
            
    except FileNotFoundError as e:
        print(f"\nError: {str(e)}")
        print("\nPlease run build_graphrag_index.py first to create the necessary knowledge graph.")
    except ValueError as e:
        print(f"\nError: {str(e)}")
    except Exception as e:
        print(f"\nUnexpected error: {str(e)}")

if __name__ == "__main__":
    main()