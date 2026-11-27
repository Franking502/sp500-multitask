import os
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import numpy as np
import pandas as pd
from LTB_train_PPO import train_ppo, rollout, sharpe
from LTB_q_learning import run_tabular_q
from LTB_plots import plot_curves

OUT_DIR = "report"
COLS = ["log_ret", "ma_ratio_20", "rsi_14", "macd_hist", "vol_21_ann"]


def get_metrics(rets):
    '''累计净值'''
    nw = np.cumprod(1 + rets)
    '''累计收益率'''
    cum = nw[-1] - 1
    '''当前净值相对历史峰值的回撤比例'''
    '''np.maximum.accumulate(nw)历史最高净值'''
    '''np.min 历史上最低的一次回撤'''
    mdd = np.min((nw - np.maximum.accumulate(nw)) / np.maximum.accumulate(nw))
    return nw, cum, sharpe(rets), mdd


def main():
    '''1. 读数据（sp500 特征表），按 segment 切成 train / val / test
    2. 跑 Q-Learning → 得到 q_rets（测试集每日收益）
    3. 对特征做标准化（用训练集的 mean/std）
    4. 训练 PPO → 得到 ppo_rets
    5. 造一个 Buy&Hold 基准（买了不动）
    6. 三条曲线对齐长度 → 算指标（累计收益/Sharpe/最大回撤）→ 存 CSV
    7. 画净值曲线'''
    # ------- 准备工作：读数据 -------
    ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_path = os.path.join(ROOT_DIR, "data/processed/sp500_features.csv")
    df = pd.read_csv(data_path, index_col=0, parse_dates=True)
    print(f"[数据] 已读取: {data_path}  共 {len(df)} 行")

    # ------- 准备工作： 切分数据集 -------
    train_df = df[df["segment"] == "train"].copy()
    val_df = df[df["segment"] == "val"].copy()
    test_df = df[df["segment"] == "test"].copy()

    # ------- 1. Q-Learning（用原始特征）-------
    q_rets = run_tabular_q(train_df, val_df, test_df)[10:]

    # ------- 2. PPO 特征标准化 -------
    mean, std = train_df[COLS].mean(), train_df[COLS].std().replace(0, 1e-8)
    for s in (train_df, val_df, test_df):
        s[COLS] = ((s[COLS] - mean) / std).clip(-5, 5)

    # ------- 3. 训练 PPO + 测试集回测 -------
    os.makedirs(OUT_DIR, exist_ok=True)

    # 修改调用：使用 train_ppo，并将保存路径加上 .zip
    model, steps, val_curve = train_ppo(train_df, val_df, COLS, os.path.join(OUT_DIR, "best_ppo.zip"))

    ppo_rets = rollout(model, test_df, COLS)

    # ------- 4. 三条策略对齐长度 -------
    n = len(ppo_rets)
    q_rets = q_rets[:n]
    bh_rets = test_df["ret"].values[10: 10 + n].copy()
    bh_rets[0] -= 0.001  # 买入扣一次手续费

    #字典键名
    strategies = {
        "PPO": ppo_rets,
        "Q-Learning": q_rets,
        "Buy & Hold": bh_rets,}

    # ------- 5. 计算指标、打印、存 CSV -------
    rows, nw_dict = [], {}
    dates = test_df.index[10: 10 + n]
    for name, r in strategies.items():
        nw, cum, shp, mdd = get_metrics(r)
        rows.append({
            "Strategy": name,
            "Cumulative Return": f"{cum:.2%}",
            "Sharpe": f"{shp:.3f}",
            "Max Drawdown": f"{mdd:.2%}", })
        nw_dict[f"{name} (Sharpe={shp:.2f})"] = nw

    result = pd.DataFrame(rows)
    result.to_csv(os.path.join(OUT_DIR, "test_metrics.csv"), index=False)
    print("\n=== 测试集长期收益对比 ===")
    print(result.to_string(index=False))

    # ------- 6. 画图 -------
    plot_curves(steps, val_curve, dates, nw_dict, OUT_DIR)
    print(f"\n[成功] 所有产出已完整保存至: {OUT_DIR}/")


if __name__ == "__main__":
    main()