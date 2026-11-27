import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_curves(steps, val_curve, dates, strategies_nw, out_dir):

    # 1. 训练曲线
    plt.figure(figsize=(7, 3.5))
    plt.plot(steps, val_curve, marker="o", color="blue")
    plt.title("PPO Validation Sharpe Curve")
    plt.xlabel("Timesteps")
    plt.ylabel("Sharpe")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "train_curve.png"), dpi=200)
    plt.close()

    # 2. 收益对比
    df = pd.DataFrame(strategies_nw, index=dates)
    ax = df.plot(figsize=(10, 4.5), title="Test Set Long-Term Net Worth Comparison", ylabel="Net Worth")
    ax.grid(alpha=0.3)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/long_term_comparison.png", dpi=200)
    plt.close(fig)


    #3.回撤曲线
    dd_df = (df / df.cummax() - 1) * 100
    ax = dd_df.plot(figsize=(10, 4.5), title="Drawdown Comparison (%)", ylabel="Drawdown (%)", alpha=0.8)
    ax.grid(alpha=0.3)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/drawdown.png", dpi=200)
    plt.close(fig)

    #4.单步收益分布直方图
    rets_df = df.pct_change().dropna() * 100
    ax = rets_df.plot.hist(bins=40, alpha=0.5, density=True, figsize=(8, 4), title="Step Return Distribution (%)")
    ax.set_xlabel("Step Return (%)")
    ax.grid(alpha=0.3)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(f"{out_dir}/return_hist.png", dpi=200)
    plt.close(fig)



