from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

# 使用字典來存儲爬取的內容
memory_store = {}

@tool
def crawl_and_embed(url: str) -> str:
    """爬取指定URL並進行嵌入"""
    loader = WebBaseLoader(url)
    docs = loader.load()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(docs, embeddings)
    memory_store[url] = vectorstore
    return "網頁內容已成功爬取並嵌入"

@tool
def rag_query(query: str) -> str:
    """使用RAG查詢並生成回應"""
    if not memory_store:
        return "沒有可用的內容，請先爬取一些網頁。"
    
    embeddings = OpenAIEmbeddings()
    combined_vectorstore = FAISS.from_documents(
        [Document(page_content="")],  # 創建一個空的向量存儲
        embeddings
    )
    
    # 合併所有存儲的向量存儲
    for vectorstore in memory_store.values():
        combined_vectorstore.merge_from(vectorstore)
    
    retriever = combined_vectorstore.as_retriever()
    
    template = """
                根據以下上下文用中文回答問題:
                {context}
                問題: {question}
                回答:
                """
    prompt = ChatPromptTemplate.from_template(template)
    
    model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | model
        | StrOutputParser()
    )
    
    return chain.invoke(query)

tools = [crawl_and_embed, rag_query]

tool_node = ToolNode(tools)

model = ChatOpenAI(model="gpt-3.5-turbo", temperature=0).bind_tools(tools)

# 定義是否繼續的函數
def should_continue(state: MessagesState) -> Literal["tools", END]:
    messages = state['messages']
    last_message = messages[-1]
    if last_message.tool_calls:
        return "tools"
    return END

# 定義調用模型的函數
def call_model(state: MessagesState):
    messages = state['messages']
    response = model.invoke(messages)
    return {"messages": [response]}

# 定義圖
workflow = StateGraph(MessagesState)

workflow.add_node("agent", call_model)
workflow.add_node("tools", tool_node)

workflow.add_edge(START, "agent")
workflow.add_conditional_edges("agent", should_continue)
workflow.add_edge("tools", 'agent')

# 初始化記憶體以持久化狀態
checkpointer = MemorySaver()

# 編譯圖
app = workflow.compile(checkpointer=checkpointer)

# # 使用Runnable
# def chat(message: str, thread_id: int = 1):
#     final_state = app.invoke(
#         {"messages": [HumanMessage(content=message)]},
#         config={"configurable": {"thread_id": thread_id}}
#     )
#     return final_state["messages"][-1].content

# 修改主循環
config = {"configurable": {"thread_id": "abc123"}}
while True:
    input_text = input('>>> ')
    if input_text.lower() == 'bye':
        break
    
    if input_text.startswith("scrape:"):
        _, url = input_text.split("scrape:", 1)
        retrieval_chain = crawl_and_embed.invoke(url.strip())
        print("網頁內容已爬取並處理完成，請輸入您的問題：")
    else:
        # 使用 langgraph 處理對話
        result = app.invoke({"messages": [HumanMessage(content=input_text)]}, config)
        result['messages'][-1].pretty_print()

# 示例使用
# print(chat("請爬取並嵌入 https://www.explainthis.io/zh-hant/swe/what-is-closure"))
# print(chat("這個網站的主要內容是什麼?"))
