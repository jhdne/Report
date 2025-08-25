from typing import List, Dict, Any
from config import METADATA_FIELDS
from cmc_fetcher import extract_social_data, extract_urls

def process_data(ucids: List[int], coin_details: Dict[str, Any], market_data: Dict[str, Any]) -> List[Dict[str, Any]]:
    """整合数据，生成待向量化文本和元数据"""
    processed_list: List[Dict[str, Any]] = []

    for ucid in ucids:
        ucid_str = str(ucid)
        detail = coin_details.get(ucid_str, {})
        market = market_data.get(ucid_str, {})
        usd_quote = market.get("quote", {}).get("USD", {})

        social_data = extract_social_data(detail.get("urls", {}))
        url_data = extract_urls(detail.get("urls", {}))

        token_info = (
            f"代币基础信息：名称：{detail.get('name', '未知')} ({detail.get('symbol', '未知')}), "
            f"分类：{detail.get('category', '未知')}, 标签：{', '.join(detail.get('tags', []))}. "
            f"简介：{detail.get('description', '无简介')[:200]}. "
            f"合约地址：{detail.get('platform', {}).get('token_address', '未知') if detail.get('platform') else '未知'}. "
            f"供应量：流通 {market.get('circulating_supply', '未知')}, 总 {market.get('total_supply', '未知')}. "
            f"市场数据：FDV {usd_quote.get('fully_diluted_valuation', '未知')} 美元. "
            f"官方链接：官网 {url_data['website']}, 白皮书 {url_data['whitepaper']}."
        )

        metadata: Dict[str, Any] = {
            "cmc_id": ucid,
            "logo": detail.get("logo"),
            "name": detail.get("name"),
            "symbol": detail.get("symbol"),
            "contracts": detail.get("platform", {}).get("token_address") if detail.get("platform") else None,
            "circulating_supply": market.get("circulating_supply"),
            "total_supply": market.get("total_supply"),
            "max_supply": market.get("max_supply"),
            "category": detail.get("category"),
            "telegram_members": social_data.get("telegram_members"),
            "twitter_followers": social_data.get("twitter_followers"),
            "urls": url_data,
            "tags": ", ".join(detail.get("tags", [])),
            "description": detail.get("description"),
            "fdv": usd_quote.get("fully_diluted_valuation"),
        }
        metadata_cleaned = {k: v for k, v in metadata.items() if v is not None and v != ''}

        processed_list.append({
            "id": f"cmc-{ucid}",
            "token_info": token_info,
            "metadata": metadata_cleaned
        })

    print(f"✅ 数据处理完成，共生成 {len(processed_list)} 条待处理数据")
    return processed_list