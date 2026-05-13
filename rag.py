from langchain_huggingface import HuggingFaceEmbeddings
import db_init as db
from pathlib import Path
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType, Collection, connections, utility
from langchain_deepseek import ChatDeepSeek
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from pydantic import ConfigDict
import subprocess
import re
import sys
import json
import os

BASE_PATH = Path("E:/ML/RAG/")#基础文件夹位置
PDF_Folder = BASE_PATH / "PDFs"
MD_Folder = BASE_PATH / "data"
BGE_MODEL_PATH = BASE_PATH  / "models" / "bge-m3"
top_k = 5

PDF_files = list(PDF_Folder.glob("*.pdf"))
IMG_Folder = list(MD_Folder.glob("*/images"))
MD_files = list(MD_Folder.glob("*/*.md"))
json_files = list(MD_Folder.glob("*/*content_list.json"))
pdf_names = [pdf_name.stem.split(" - ")[2].strip() for pdf_name in PDF_files]
DATA_path = [d for d in MD_Folder.glob("*") if d.is_dir()]


load_dotenv()
MILVUS_URI = "http://localhost:19530"
DATABASE_NAME = "Planet"
COLLECTION_NAME = "Lunar"

client = MilvusClient(
    uri=MILVUS_URI,
    token=os.getenv("MILVUS_TOKEN"),
    timeout=1000,
)
if utility.has_database(DATABASE_NAME):#如果数据库存在，先删除
    utility.drop_database(DATABASE_NAME)
utility.create_database(DATABASE_NAME)
client.use_database(DATABASE_NAME)
all_docs = db.init_db() 
# 第一次运行时使用后续使用可改为下述代码，防止重复多次运行
#text_docs = db.chunk_markdown(MD_files, PDF_files, pdf_names, DATA_path, chunk_size=500, chunk_overlap=20)
#img_docs = db.load_img_from_json("all_papers_data_vision.json", DATA_path, PDF_files)
#all_docs = text_docs + img_docs
print("载入bge-m3嵌入模型...")
text_embedding = HuggingFaceEmbeddings(
    model_name=BGE_MODEL_PATH,       
    model_kwargs={"device": "cuda"},     
    encode_kwargs={"normalize_embeddings": True}
)
print("bge-m3嵌入模型加载完成")
def create_milvus_collection():
    
    schema = CollectionSchema(
        fields=[
            FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1024),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        ],
    )

    texts = [doc.page_content for doc in all_docs]
    embeddings = text_embedding.embed_documents(texts)
    print("所有文档嵌入计算完成")

    connections.connect(host="localhost", port="19530", db_name=DATABASE_NAME)

    if utility.has_collection(COLLECTION_NAME):#如果集合存在，先删除
        Collection(COLLECTION_NAME).drop()

    col = Collection(COLLECTION_NAME, schema=schema)

    texts = [doc.page_content for doc in all_docs]
    embeddings = text_embedding.embed_documents(texts)
    metas = [json.dumps(doc.metadata, ensure_ascii=False) for doc in all_docs]
    col.insert([texts, embeddings, metas])
    col.flush()

    dense_index = {
        "index_type": "HNSW",
        "metric_type": "COSINE",
        "params": {"M": 16, "efConstruction": 200}
    }
    col.create_index("embedding", dense_index)

    # 索引创建成功后，才能加载集合
    print("索引创建完成")

    col.load()

class MilvusRetriever(BaseRetriever):
    client: MilvusClient
    collection: str
    embed_model: HuggingFaceEmbeddings
    top_k: int = 3

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _get_relevant_documents(self, query: str) -> list[Document]:
        query_embedding = self.embed_model.embed_query(query)
        results = self.client.search( # 批量查询，list[list[dict]]
            collection_name=self.collection,
            data=[query_embedding],
            output_fields=["text", "metadata"],
            limit=self.top_k,
        )
        docs = []
        for result in results:
            for res in result:
                entity = res["entity"]
                text = entity["text"]
                meta = json.loads(entity["metadata"])
                docs.append(Document(page_content = text, metadata = meta))
        return docs

paper_retriever = MilvusRetriever(
    client = client,
    collection = COLLECTION_NAME,
    embed_model = text_embedding,
    top_k = top_k,
)

bm25_retriever = BM25Retriever.from_documents(all_docs, k=top_k)
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, paper_retriever],
    weights=[0.5, 0.5],
) # 组合检索器，向量检索和关键字检索的权重均为0.5


client.load_collection(COLLECTION_NAME)

llm = ChatDeepSeek(
    model="deepseek-v4-flash",
    temperature=0.5,
    timeout=1000,
    max_retries=2,
)

class RAGState(TypedDict):
    messages: list          # 对话历史
    question: str           # 当前用户问题
    context_docs: list      # 检索到的文档
    answer: str             # 最终答案
    sources: list           # 新增

def build_agent(retriever, llm):
    system_prompt = """
        你是一个专业的学术论文助手。根据提供的上下文回答用户问题。
        要求：
        1. 如果上下文中有答案，请准确引用，并注明来源（如 [文件链接] 或 [图片链接]）。
        2. 如果上下文不足以回答问题，请如实说明，不要编造信息。
        3. 使用贴合问题的上下文信息，可以有多个引用文件和图片，如果上下文信息不相关，不要强行引用。
        4. 请在最后输出引用的文件、图片的绝对路径，输出示例为
        “
        **引用文件**
        - E:/ML/RAG/data/1.pdf
        - E:/ML/RAG/data/1.png
        ”。

        上下文：
        {context}
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "{question}"),
    ])

    def retrieve(state: RAGState) -> dict:
        docs = retriever.invoke(state["question"])
        return {"context_docs": docs}

    def generate(state: RAGState) -> dict:
        docs = state["context_docs"]
        # 组织上下文，区分文本和图片
        parts = []
        sources = set()
        for i, doc in enumerate(docs):
            meta = doc.metadata
            if meta.get("type") == "image":
                src = meta["img_paths"]   # 图片路径
                label = "图片"
            else:
                src = meta.get("pdf_path")
                label = "文本"
            parts.append(f"[{i+1}·{label}] 来源: {src}\n{doc.page_content}")
            sources.add(src)

        context_str = "\n\n".join(parts)

        chain = prompt | llm | StrOutputParser()
        answer = chain.invoke({
            "messages": state["messages"],
            "context": context_str,
            "question": state["question"],
        })
        return {
            "answer": answer,
            "sources": list(sources)
        }

    graph = StateGraph(RAGState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    # 使用内存保存对话历史（多轮对话）
    memory = MemorySaver()
    return graph.compile(checkpointer=memory)

agent = build_agent(ensemble_retriever, llm)

def open_file(file_path: str):
    """用操作系统默认程序打开文件（支持 PDF、图片等）"""

    # Windows 系统下路径中的 / 改为 \（实际上 / 也能用，但有些老软件要求 \）
    if sys.platform == "win32":
        file_path = file_path.replace("/", "\\")
        os.startfile(file_path)          # Windows 专用，非常可靠

def extract_ref_paths(text: str) -> list[str]:
    """
    从大模型回答中提取“引用文件”列表中的本地路径。
    适用格式：
        **引用文件**
        - C:/path/to/file.pdf
        - E:/path/to/image.jpg
    """
    # 定位“引用文件”部分（可能中文也可英文）
    # 先找到 **引用文件** 或 **引用源文件** 等标题
    pattern_section = r'\*\*引用(?:源)?文件\*\*\s*\n(.*?)(?=\n\*\*|\Z)'
    match = re.search(pattern_section, text, re.DOTALL)
    if not match:
        return []
    
    lines = match.group(1).strip().splitlines()
    paths = []
    for line in lines:
        # 去掉行首的 "- " 及空格
        stripped = line.strip().lstrip('- ').strip()
        if stripped and not stripped.startswith('**'):  # 忽略可能的新标题
            paths.append(stripped)
    return paths
# ============================================================
# 8. 交互循环
# ============================================================
print("=== 问答就绪，输入 'quit' 退出 ===")
thread_id = "user_1"
history = []
while True:
    query = input("\n用户: ").strip()
    if query.lower() in ["quit", "exit", "q"]:
        break

    # 初始状态要包含 sources 的初始值
    state_input = {
        "question": query,
        "messages": history,          # 对话历史
        "context_docs": [],
        "answer": "",
        "sources": []
    }
    config = {"configurable": {"thread_id": thread_id}}
    result = agent.invoke(state_input, config)

    answer = result["answer"]
    sources = result.get("sources", [])

    print(f"\n助手: {answer}")
    ref_paths = extract_ref_paths(answer)
    if ref_paths:
        choice = input("是否打开所有引用文件？(y/n): ").strip().lower()
        if choice == "y":
            for path in ref_paths:
                try:
                    open_file(path)  
                except Exception as e:
                    print(f"无法打开 {path}: {e}")
        print("解析到的路径:", ref_paths)