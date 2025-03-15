import sqlite3
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings  # 使用最新的包，请先运行: pip install -U langchain-openai
from langchain.vectorstores import FAISS

def load_texts_from_db(db_path="pdf_texts.db"):
    """
    从数据库中读取所有 PDF 文件对应的文本数据
    返回一个列表，元素为 (filename, content)
    """
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT filename, content FROM pdf_texts")
    rows = cursor.fetchall()
    conn.close()
    return rows

def get_chunks_from_text(text, chunk_size=500, chunk_overlap=50):
    """
    使用 CharacterTextSplitter 将文本分成多个块
    每块大约 chunk_size 个字符，块与块之间有 chunk_overlap 个字符重叠
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",      # 尽可能在换行处分割
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store_from_chunks(chunks):
    """
    使用 OpenAI 的嵌入接口将文本块转换成向量，
    并将它们存入 FAISS 向量数据库中
    """
    embeddings = OpenAIEmbeddings()  # 确保设置好 OPENAI_API_KEY 环境变量
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def main():
    # 从数据库加载数据
    data = load_texts_from_db()
    all_chunks = []
    
    # 对每个 PDF 文本进行分块
    for filename, content in data:
        chunks = get_chunks_from_text(content)
        print(f"{filename} divide to {len(chunks)} embedding Chunks")
        # 这里可以在每个块前加上文件名信息以便后续追踪
        all_chunks.extend(chunks)
    
    print(f"In total {len(all_chunks)} embedding Chunks")
    
    # 生成向量数据库
    vector_store = create_vector_store_from_chunks(all_chunks)
    vector_store.save_local("faiss_index")
    print("Vector Database has been generated")
        
if __name__ == "__main__":
    main()
