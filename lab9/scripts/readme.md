# Knowledge Base QA Chatbot

This project reads text from PDF files, stores the text in a local SQLite database, splits the text into smaller chunks, creates embeddings using OpenAI, and builds a question answering chain based on a retrieval approach.

## Setup Instructions

### Prerequisites

- **Python:** Version 3.8 or higher.
- **OpenAI API Key:** Obtain your API key from [OpenAI](https://openai.com/) and set it as an environment variable.

### Installation Steps

1. **Clone the repository or copy the project files.**

2. **Create and activate a virtual environment (optional but recommended):**

   ```bash
   python -m venv venv
   source venv/bin/activate   # On Linux or macOS
   venv\Scripts\activate      # On Windows
   ```

3. **Install the required libraries.** You can run these commands:

   ```bash
   pip install PyPDF2
   pip install langchain
   pip install langchain-openai
   pip install langchain-community
   pip install openai
   ```

   Alternatively, create a `requirements.txt` file with the following content:

   ```
   PyPDF2
   langchain
   langchain-openai
   langchain-community
   openai
   ```

   Then install with:

   ```bash
   pip install -r requirements.txt
   ```

4. **Set your OpenAI API key.**

   On Linux or macOS:

   ```bash
   export OPENAI_API_KEY=your_openai_api_key_here
   ```

   On Windows:

   ```cmd
   set OPENAI_API_KEY=your_openai_api_key_here
   ```

### Data Setup

- Create a folder named `data` one level above the script directory (i.e., `../data`).
- Place your PDF files in this folder.

### Running the Program

Run the driver script with:

```bash
python driver.py
```

The program will:
- Build the knowledge base from your PDF files.
- Create a vector database for the text chunks.
- Start an interactive session where you can ask questions. Type `exit` to quit.

## Libraries and Versions

The following libraries were used in this project:

- **PyPDF2:** Latest version (tested with PyPDF2 3.x)
- **langchain:** Core components installed as `langchain-core` version 0.3.45
- **langchain-openai:** Version 0.3.8
- **langchain-community:** Latest version (required for updated FAISS and chat model imports)
- **openai:** Version 1.66.3
- **sqlite3:** Built into Python's standard library

## Project Structure

- **driver.py:** Main script that builds the knowledge base and runs the interactive QA session.
- **data/**: Folder that should contain the PDF files.
- **pdf_texts.db:** SQLite database file storing extracted PDF text.
- **faiss_index/**: Folder where the FAISS vector database is stored.

## Notes

- This project uses a retrieval approach to find the most relevant text chunks for a given question, and then an LLM generates the answer.
- Ensure that your OpenAI API key is valid and set correctly.

---

This README provides the necessary instructions and lists the key libraries with their versions for setting up and running the project.