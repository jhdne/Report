import json
from typing import List, Set

SNAPSHOT_FILE = "ucids_snapshot.json"

def save_ucids_snapshot(ucids: List[int]):
    """将 UCID 列表保存到快照文件中"""
    try:
        with open(SNAPSHOT_FILE, 'w') as f:
            json.dump(ucids, f)
        print(f"✅ 成功将 {len(ucids)} 个 UCID 保存到快照文件: {SNAPSHOT_FILE}")
    except IOError as e:
        print(f"❌ 保存 UCID 快照文件失败: {e}")

def load_ucids_snapshot() -> Set[int]:
    """从快照文件加载 UCID 列表，并返回一个集合以便快速查找"""
    try:
        with open(SNAPSHOT_FILE, 'r') as f:
            ucids = json.load(f)
        print(f"✅ 成功从快照文件加载 {len(ucids)} 个 UCID")
        return set(ucids)
    except FileNotFoundError:
        print(f"⚠️ 未找到 UCID 快照文件。")
        return set()
    except (IOError, json.JSONDecodeError) as e:
        print(f"❌ 加载 UCID 快照文件失败: {e}")
        return set()