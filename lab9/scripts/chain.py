import autogen as ag

# 定义一个继承自 AssistantAgent 的检索型对话助手
class RetrievalAssistantAgent(ag.AssistantAgent):
    def __init__(self, name, llm, vector_store, **kwargs):
        super().__init__(name=name, llm=llm, **kwargs)
        self.vector_store = vector_store
        self.chat_history = []  # 用于保存对话历史
    
    def generate_response(self, message):
        # 提取用户问题
        question = message.content
        # 从向量数据库中检索相关文档（返回的对象需含有 page_content 属性）
        retrieved_docs = self.vector_store.similarity_search(question, k=3)
        context = "\n".join(doc.page_content for doc in retrieved_docs)
        
        # 将对话历史拼接到提示中
        history_text = "\n".join(self.chat_history)
        prompt = (
            f"对话历史：\n{history_text}\n\n"
            f"相关上下文：\n{context}\n\n"
            f"用户问题：{question}\n请回答："
        )
        
        # 通过 LLM 生成回答
        answer = self.llm.generate(prompt)
        # 保存当前问答到对话历史
        self.chat_history.append(f"用户: {question}")
        self.chat_history.append(f"机器人: {answer}")
        return answer

def create_conversation_chain(llm, vector_store):
    """
    传入 LLM 模型和向量数据库，返回一个检索型对话助手。
    """
    assistant = RetrievalAssistantAgent(name="Assistant", llm=llm, vector_store=vector_store)
    return assistant

if __name__ == "__main__":
    # 初始化 LLM 模型（确保设置了 OPENAI_API_KEY 环境变量）
    from autogen.llm import OpenAILLM
    llm = OpenAILLM(api_key="YOUR_OPENAI_API_KEY", model="gpt-3.5-turbo", temperature=0)
    
    # 从本地加载向量数据库（假设先前已将向量库保存到 "faiss_index" 文件夹中）
    from langchain_openai import OpenAIEmbeddings
    from langchain.vectorstores import FAISS
    vector_store = FAISS.load_local("faiss_index", OpenAIEmbeddings())
    
    # 创建对话链
    conversation_agent = create_conversation_chain(llm, vector_store)
    
    print("对话链已建立，输入 'exit' 可退出。")
    while True:
        user_input = input("用户: ")
        if user_input.lower() == "exit":
            break
        
        # 构造一个简单的消息对象，要求具有 content 属性
        class Message:
            def __init__(self, content):
                self.content = content
        message = Message(user_input)
        
        response = conversation_agent.generate_response(message)
        print("机器人:", response)
