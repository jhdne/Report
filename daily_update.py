import numpy as np
from typing import List
from cmc_fetcher import fetch_ucids
from utils import load_ucids_snapshot, save_ucids_snapshot
# 导入与 main.py 相同的核心处理函数
from main import run_sync_process

def daily_update():
    print("=" * 60)
    print("🔄 开始执行【每日增量更新】流程")
    print("=" * 60)

    # 步骤 1：获取最新和旧的 UCID
    print("\n获取最新 UCID 列表并加载快照...")
    current_ucids_list = fetch_ucids()
    if not current_ucids_list:
        print("❌ 未能获取到当前 UCID，流程终止")
        return

    old_ucids_set = load_ucids_snapshot()
    if not old_ucids_set:
        print("❌ 未找到旧的 UCID 快照，请先运行 main.py 进行首次全量同步。")
        return

    # 步骤 2：识别新增的 UCID
    new_ucids = [ucid for ucid in current_ucids_list if ucid not in old_ucids_set]

    if not new_ucids:
        print("\n✅ 未发现新增代币，无需更新。")
        print("=" * 60)
        return

    print(f"\n🔍 发现 {len(new_ucids)} 个新增代币，开始处理...")
    print(f"新增代币ID: {new_ucids}")

    # 步骤 3：对新增代币执行同步流程
    run_sync_process(new_ucids)

    # 步骤 4：用最新的全量 UCID 更新快照
    print("\n更新 UCID 快照文件...")
    save_ucids_snapshot(current_ucids_list)

    print("\n🎉 每日增量更新流程执行完毕！")

if __name__ == "__main__":
    daily_update()