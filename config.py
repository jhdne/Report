import os
from dotenv import load_dotenv
from pinecone import PodSpec, ServerlessSpec

# 加载 .env 文件中的环境变量
load_dotenv()

# -------------------------- 基础配置 --------------------------
BATCH_SIZE = 50  # 批量拉取大小（CMC 免费版建议 50-100）
REQUEST_DELAY = 2  # 请求间隔（秒），避免 API 限流

# -------------------------- CoinMarketCap API 配置 --------------------------
CMC_CONFIG = {
    "api_key": os.getenv("CMC_API_KEY"),
    "base_url": "https://pro-api.coinmarketcap.com",
    "endpoints": {
        "map": "/v1/cryptocurrency/map",          # 获取 UCID 列表
        "info": "/v2/cryptocurrency/info",        # 获取代币详情
        "quotes": "/v1/cryptocurrency/quotes/latest"  # 获取市场数据
    },
    "headers": {
        "X-CMC_PRO_API_KEY": os.getenv("CMC_API_KEY"),
        "Accept": "application/json"
    },
    "quotes_params": {
        "convert": "USD"
    }
}

# -------------------------- Pinecone 配置 (已更新) --------------------------
# Llama-2 text embedding v2 支持的维度: 1024, 2048, 768, 512, 384
EMBEDDING_MODEL_DIMENSION = 1024

PINECONE_CONFIG = {
    "api_key": os.getenv("PINECONE_API_KEY"),
    "index_name": "coindata",
    "metric": "cosine",
    # PodSpec 用于指定云和区域。请根据您的 Pinecone 项目环境修改
    # ServerlessSpec 仅在特定 AWS 区域可用
    "spec": PodSpec(environment="gcp-starter") # 或 ServerlessSpec(cloud="aws", region="us-west-2")
}

# -------------------------- 数据字段配置 --------------------------
METADATA_FIELDS = [
    "cmc_id", "logo", "name", "symbol", "contracts",
    "circulating_supply", "total_supply", "max_supply",
    "category", "telegram_members", "twitter_followers",
    "urls", "tags", "description", "fdv"
]