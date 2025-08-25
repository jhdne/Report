import requests
import time
from typing import List, Dict, Any
from config import CMC_CONFIG, BATCH_SIZE, REQUEST_DELAY

def _fetch_in_batches(ucids: List[int], endpoint_key: str, params_extra: Dict = None) -> Dict[str, Any]:
    """通用批量获取函数"""
    data_map: Dict[str, Any] = {}
    total_batches = (len(ucids) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_idx in range(total_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE
        batch_ucids_str = ",".join(map(str, ucids[start_idx:end_idx]))

        params = {"id": batch_ucids_str}
        if params_extra:
            params.update(params_extra)

        try:
            response = requests.get(
                url=f"{CMC_CONFIG['base_url']}{CMC_CONFIG['endpoints'][endpoint_key]}",
                headers=CMC_CONFIG["headers"],
                params=params,
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            if data["status"]["error_code"] == 0:
                data_map.update(data["data"])
                print(f"✅ {endpoint_key} 拉取：第 {batch_idx+1}/{total_batches} 批成功")
            else:
                print(f"❌ {endpoint_key} API 错误 (批次 {batch_idx+1}): {data['status']['error_message']}")

        except requests.exceptions.RequestException as e:
            print(f"❌ {endpoint_key} 请求失败 (批次 {batch_idx+1}): {e}")

        time.sleep(REQUEST_DELAY)

    print(f"✅ {endpoint_key} 数据拉取完成，共获取 {len(data_map)} 个代币的数据")
    return data_map


def fetch_ucids() -> List[int]:
    """获取所有代币的 UCID 列表"""
    ucids: List[int] = []
    start = 1
    while True:
        try:
            response = requests.get(
                url=f"{CMC_CONFIG['base_url']}{CMC_CONFIG['endpoints']['map']}",
                headers=CMC_CONFIG["headers"],
                params={"start": start, "limit": BATCH_SIZE},
                timeout=15
            )
            response.raise_for_status()
            data = response.json()

            if data["status"]["error_code"] != 0:
                print(f"❌ 获取 UCID 错误：{data['status']['error_message']}")
                break

            batch_data = data["data"]
            if not batch_data:
                print(f"✅ UCID 拉取完成，共 {len(ucids)} 个代币")
                break

            ucids.extend(coin["id"] for coin in batch_data)
            print(f"✅ 已获取 {len(ucids)} 个 UCID...")
            start += BATCH_SIZE
            time.sleep(REQUEST_DELAY)

        except requests.exceptions.RequestException as e:
            print(f"❌ UCID 请求失败：{e}")
            break
    return ucids

def fetch_coin_details(ucids: List[int]) -> Dict[str, Any]:
    """批量获取代币详情"""
    return _fetch_in_batches(ucids, "info")

def fetch_market_data(ucids: List[int]) -> Dict[str, Any]:
    """批量获取市场数据"""
    return _fetch_in_batches(ucids, "quotes", CMC_CONFIG["quotes_params"])

def extract_social_data(links: Dict[str, Any]) -> Dict[str, Any]:
    """提取社交数据"""
    return {
        "twitter_followers": links.get("twitter", [{}])[0].get("followers"),
        "telegram_members": links.get("telegram", [{}])[0].get("members")
    }

def extract_urls(urls: Dict[str, Any]) -> Dict[str, Any]:
    """提取 URL 信息"""
    return {
        "website": urls.get("website", [""])[0],
        "whitepaper": urls.get("whitepaper", [""])[0],
        "twitter": urls.get("twitter", [""])[0]
    }