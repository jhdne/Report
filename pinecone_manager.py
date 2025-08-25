import time
from pinecone import Pinecone
from config import PINECONE_CONFIG, EMBEDDING_MODEL_DIMENSION

def init_pinecone_client():
    """初始化 Pinecone 客户端"""
    try:
        pc = Pinecone(api_key=PINECONE_CONFIG["api_key"])
        print("✅ Pinecone 客户端初始化成功")
        return pc
    except Exception as e:
        print(f"❌ Pinecone 客户端初始化失败：{e}")
        return None

def get_or_create_index(pc_client):
    """检查索引是否存在，不存在则创建"""
    index_name = PINECONE_CONFIG["index_name"]

    if index_name not in pc_client.list_indexes().names():
        print(f"🔧 索引 {index_name} 不存在，开始创建...")
        try:
            pc_client.create_index(
                name=index_name,
                dimension=EMBEDDING_MODEL_DIMENSION,
                metric=PINECONE_CONFIG["metric"],
                spec=PINECONE_CONFIG["spec"]
            )
            while not pc_client.describe_index(index_name).status['ready']:
                print("⏳ 正在等待索引创建完成...")
                time.sleep(5)
            print(f"✅ 索引 {index_name} 创建成功")
        except Exception as e:
            print(f"❌ 索引创建失败：{e}")
            return None

    try:
        index = pc_client.Index(index_name)
        print(f"✅ 成功连接到索引：{index_name}")
        return index
    except Exception as e:
        print(f"❌ 连接索引 {index_name} 失败：{e}")
        return None

def upsert_data_to_pinecone(index, pinecone_data):
    """将处理后的数据批量存入 Pinecone"""
    if not pinecone_data:
        print("⚠️ 无待存储的数据，跳过 Pinecone 存储步骤")
        return

    batch_size = 100
    try:
        print(f"🚀 开始分批次上传数据，每批 {batch_size} 条...")
        for i in range(0, len(pinecone_data), batch_size):
            batch = pinecone_data[i:i + batch_size]
            response = index.upsert(vectors=batch)
            print(f"✅ 成功上传批次 {i // batch_size + 1}，共 {response.get('upserted_count', 0)} 条向量")

        index_stats = index.describe_index_stats()
        print(f"📊 数据上传完成！索引当前统计：总向量数 = {index_stats.get('total_vector_count', 0)}")
    except Exception as e:
        print(f"❌ 数据存入 Pinecone 失败：{e}")