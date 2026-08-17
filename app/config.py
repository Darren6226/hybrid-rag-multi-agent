import os
import sys
import logging
from typing import List
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.embeddings import Embeddings

# 环境配置
load_dotenv()

# ============================================================
# 统一日志配置
# ============================================================
LOG_FORMAT = "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

logging.basicConfig(
    level=logging.INFO,
    format=LOG_FORMAT,
    datefmt=DATE_FORMAT,
    stream=sys.stderr,
)

# 降低第三方库日志级别，避免刷屏
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("langchain").setLevel(logging.WARNING)
logging.getLogger("langgraph").setLevel(logging.WARNING)

# Milvus 警告过滤
class MilvusAsyncWarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        return not ('AsyncMilvusClient' in msg and 'no running event loop' in msg)

for handler in logging.root.handlers:
    handler.addFilter(MilvusAsyncWarningFilter())

milvus_logger = logging.getLogger('pymilvus')
milvus_logger.addFilter(MilvusAsyncWarningFilter())
milvus_logger.setLevel(logging.ERROR)

logger = logging.getLogger(__name__)

# DashScope 配置
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")
DASHSCOPE_API_BASE = os.getenv("DASHSCOPE_API_BASE")

# SenseNova 配置
SENSENOVA_API_KEY = os.getenv("SENSENOVA_API_KEY")

modelscope_base_url = os.getenv("modelscope_base_url")
modelscope_api_key = os.getenv("modelscope_api_key")

class DashScopeEmbeddings(Embeddings):
    def __init__(self, model: str, api_key: str, base_url: str = None):
        self.model = model
        self.api_key = api_key
        # 只有在明确提供base_url时才使用，否则使用DashScope原生API
        self.base_url = base_url

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        # 如果有base_url，使用ModelScope API方式
        if self.base_url:
            import requests
            import json
            
            all_embeddings = []
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            batch_size = 10
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                
                payload = {
                    "model": self.model,
                    "input": batch,
                    "encoding_format": "float"
                }
                
                try:
                    response = requests.post(
                        f"{self.base_url}/embeddings",
                        headers=headers,
                        json=payload,
                        timeout=30
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if "data" in result:
                            embeddings = [item["embedding"] for item in result["data"]]
                            all_embeddings.extend(embeddings)
                        else:
                            raise Exception(f"ModelScope API返回格式错误: {result}")
                    else:
                        raise Exception(f"ModelScope Embedding Error: {response.status_code} - {response.text}")
                        
                except requests.exceptions.RequestException as e:
                    raise Exception(f"ModelScope API请求失败: {str(e)}")
                    
            return all_embeddings
        else:
            # DashScope 原生 API：多模态 embedding 模型用 MultiModalEmbedding，否则用 TextEmbedding
            all_embeddings = []
            batch_size = 10
            is_multimodal = 'vision' in self.model.lower() or 'multimodal' in self.model.lower()

            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                if is_multimodal:
                    import dashscope
                    input_data = [{'text': t} for t in batch]
                    resp = dashscope.MultiModalEmbedding.call(
                        model=self.model, input=input_data, api_key=self.api_key
                    )
                else:
                    from dashscope import TextEmbedding
                    resp = TextEmbedding.call(model=self.model, input=batch, api_key=self.api_key)

                if resp.status_code == 200:
                    all_embeddings.extend([item['embedding'] for item in resp.output['embeddings']])
                else:
                    raise Exception(f"DashScope Embedding Error: {resp.message}")
            return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        return self.embed_documents([text])[0]

# LLM 与 Embeddings 初始化
llm = ChatOpenAI(
    model="qwen3.7-max",
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_API_BASE,
    temperature=0,
    request_timeout=120
)

# GraphRAG 使用的 LLM（实体关系抽取用 qwen3.7-plus 足够，比 max 快数倍）
graph_llm = ChatOpenAI(
    temperature=0,
    model="qwen3.7-plus",
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_API_BASE,
    request_timeout=200
)

# Supervisor 使用的 LLM（使用 qwen3.7-max，需支持 JSON / 结构化输出模式）
supervisor_llm = ChatOpenAI(
    temperature=0,
    model="qwen3.7-max",
    api_key=DASHSCOPE_API_KEY,
    base_url=DASHSCOPE_API_BASE,
    request_timeout=120
)

# 使用 DashScope 的 Embedding 服务（不传 base_url，避免使用 ModelScope）
embeddings = DashScopeEmbeddings(
    model="tongyi-embedding-vision-flash-2026-03-06",
    api_key=DASHSCOPE_API_KEY,
    base_url=None  # 使用DashScope原生API
)

# ModelScope LLM（DashScope 免费额度用完时的 fallback）
modelscope_llm = None
if modelscope_base_url and modelscope_api_key:
    modelscope_llm = ChatOpenAI(
        model="qwen-plus",
        api_key=modelscope_api_key,
        base_url=modelscope_base_url,
        temperature=0
    )

# SenseNova DeepSeek LLM（OpenAI 兼容接口）
# 使用场景：可作为 llm/graph_llm/supervisor_llm 的替代选项
# 需在 .env 中配置 SENSENOVA_API_KEY
deepseek_llm = None
if SENSENOVA_API_KEY:
    deepseek_llm = ChatOpenAI(
        model="deepseek-v4-flash",
        api_key=SENSENOVA_API_KEY,
        base_url="https://token.sensenova.cn/v1",
        temperature=0
    )

# 数据库配置（支持环境变量覆盖，默认值与 docker-compose-rag.yml 端口映射一致）
DATABASE_URI = os.getenv(
    "DATABASE_URI", 'mysql+pymysql://root:root@localhost:5306/langgraph?charset=utf8mb4')
NEO4J_URL = os.getenv("NEO4J_URL", 'bolt://localhost:17687')
NEO4J_USERNAME = os.getenv("NEO4J_USERNAME", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")
MILVUS_URI = os.getenv("MILVUS_URI", "http://localhost:19530")
MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")