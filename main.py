import numpy as np
from typing import List
from cmc_fetcher import fetch_ucids, fetch_coin_details, fetch_market_data
from data_processor import process_data
from pinecone_manager import init_pinecone_client, get_or_create_index, upsert_data_to_pinecone
from config import EMBEDDING_MODEL_DIMENSION
from utils import save_ucids_snapshot

def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    这是一个虚拟的文本向量化函数。
    【【【重要】】】您需要在这里替换为对 `llama-text-embed-v2` 模型的真实 API 调用。
    """
    print(f"⚠️ 警告：正在使用【虚拟】向量化函数生成随机向量。请替换为真实模型 API 调用！")
    return np.random.rand(len(texts), EMBEDDING_MODEL_DIMENSION).tolist()

def run_sync_process(ucids: List[int]):
    """执行同步的核心流程"""
    if not ucids:
        print("无 UCID 需要处理。")
        return

    # 拉取数据
    coin_details = fetch_coin_details(ucids)
    market_data = fetch_market_data(ucids)
    if not coin_details or not market_data:
        print("❌ 获取详情或市场数据失败，流程终止")
        return

    # 处理数据
    processed_list = process_data(ucids, coin_details, market_data)
    if not processed_list: return

    # 向量化
    print("\n向量化...")
    texts_to_embed = [item["token_info"] for item in processed_list]
    vectors = embed_texts(texts_to_embed)

    # 准备最终数据
    pinecone_data = [
        {"id": item["id"], "values": vectors[i], "metadata": item["metadata"]}
        for i, item in enumerate(processed_list)
    ]
    print("✅ 数据已转换为 Pinecone 格式")

    # 存储到 Pinecone
    print("\n存储到 Pinecone...")
    pc_client = init_pinecone_client()
    if not pc_client: return
    index = get_or_create_index(pc_client)
    if not index: return
    upsert_data_to_pinecone(index, pinecone_data)

def main():
    print("=" * 60)
    print("📌 开始执行【首次全量同步】流程")
    print("=" * 60)

    all_ucids = fetch_ucids()
    if not all_ucids: return
    print(f"🔍 本次处理 {len(all_ucids)} 个代币")

    run_sync_process(all_ucids)

    print("\n保存 UCID 快照...")
    save_ucids_snapshot(all_ucids)

    print("\n🎉 全量同步流程执行完毕！")

if __name__ == "__main__":
    main()