"""Plot overall errors, improvements and per-step model comparisons."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


# Recorded per-step MAEs from the validation and test runs in TimeSeriesAnalysis.ipynb.
RESULTS = {5: {'Validation (226 windows)': {'Naive': [25.979250225663748,
                                            41.131410305309736,
                                            46.585746292035424,
                                            54.407332433628326,
                                            57.15433138495574],
                                  'XGBoost': [25.86270048314084,
                                              41.017283417559156,
                                              46.37312535965283,
                                              53.95604802260099,
                                              56.78134249265945],
                                  'LSTM': [25.737023249688352,
                                           40.4135745737586,
                                           45.5643882050529,
                                           53.69289677153933,
                                           56.899578759243944]},
     'Test (219 windows)': {'Naive': [41.45376266210047,
                                      56.32753196347029,
                                      74.4351979589041,
                                      80.98502497260272,
                                      88.35314886301366],
                            'XGBoost': [41.37200215600047,
                                        56.25697968057814,
                                        74.10962558742125,
                                        80.51319048390211,
                                        87.77445419934654],
                            'LSTM': [41.609919668835715,
                                     56.00393750522356,
                                     73.43621812543743,
                                     80.41631744427201,
                                     87.2139625445327]}},
 20: {'Validation (56 windows)': {'Naive': [28.278390107142894,
                                            34.64428712499999,
                                            39.44570057142857,
                                            47.69678173214292,
                                            48.16123742857145,
                                            53.28179923214285,
                                            66.91552292857146,
                                            62.98447955357146,
                                            70.99984739285715,
                                            68.14574319642861,
                                            76.03985160714285,
                                            83.48878264285716,
                                            87.96840116071427,
                                            89.64447023214287,
                                            94.54180032142858,
                                            95.85089553571429,
                                            104.74003380357149,
                                            109.43662805357143,
                                            112.4223372321428,
                                            121.47000560714282],
                                  'XGBoost': [28.11594615131486,
                                              34.56291872183749,
                                              38.82592611485249,
                                              46.309326544952306,
                                              46.89109099744237,
                                              52.01693309221324,
                                              65.35601244177506,
                                              61.523785148883015,
                                              69.11126148980713,
                                              66.93368406403575,
                                              74.87109591032872,
                                              80.96819434889495,
                                              85.27940097978045,
                                              87.41514211304202,
                                              91.77362484857885,
                                              93.20040780553056,
                                              102.39367068485483,
                                              105.88187620653316,
                                              109.83266447098318,
                                              117.65403216335402],
                                  'LSTM': [27.474531194563344,
                                           33.96342851574343,
                                           37.24267283011408,
                                           45.62674737262868,
                                           46.204204381850275,
                                           50.671733786700784,
                                           64.33694437801748,
                                           59.888274767571836,
                                           67.91098219624357,
                                           65.40375588339191,
                                           74.90569490284265,
                                           80.40523645012135,
                                           84.525690753636,
                                           86.56457687272719,
                                           91.11825672095831,
                                           93.72565560513264,
                                           101.79974314069092,
                                           105.76569018580001,
                                           109.95371004573585,
                                           117.7476088160171]},
      'Test (54 windows)': {'Naive': [42.178326611111146,
                                      61.108330648148055,
                                      84.55371994444447,
                                      92.60519296296297,
                                      100.24816896296298,
                                      99.72261548148151,
                                      116.31702105555557,
                                      124.12220149999996,
                                      134.61150896296294,
                                      133.31964788888885,
                                      116.22428383333333,
                                      127.10318559259268,
                                      134.1363345000001,
                                      138.85443338888885,
                                      139.6794568518519,
                                      143.49646901851852,
                                      149.4458866851852,
                                      155.09721318518515,
                                      149.62112088888887,
                                      148.75981535185184],
                            'XGBoost': [42.306429376557055,
                                        61.52439605958329,
                                        84.13298936686962,
                                        91.98935711288512,
                                        100.10462341484184,
                                        99.3791593129835,
                                        115.19314098535835,
                                        123.11493270915444,
                                        133.5605789927515,
                                        131.9898668246081,
                                        114.62091006738791,
                                        124.21696819069764,
                                        131.96423985699133,
                                        135.15094924220554,
                                        135.81503303342922,
                                        138.95071411710498,
                                        145.30662569943803,
                                        152.85883881558334,
                                        146.58263143308739,
                                        143.71969614830186],
                            'LSTM': [42.53452073817907,
                                     61.70338513884676,
                                     83.07796410016421,
                                     91.38298498406546,
                                     99.39692949143912,
                                     99.18147644934098,
                                     114.17538426276681,
                                     122.39822730300081,
                                     132.26842445090003,
                                     130.30281670722138,
                                     113.46399852647345,
                                     123.53205198610046,
                                     131.44012096552257,
                                     134.13792185677644,
                                     135.24345045116738,
                                     138.63135902171666,
                                     144.77169488650713,
                                     152.38553048193918,
                                     146.5103017794534,
                                     142.81679762475926]}}}

STYLES = {
    "Naive": dict(color="#687484", linestyle="--", marker="x"),
    "XGBoost": dict(color="#a15116", linestyle="-.", marker="^"),
    "LSTM": dict(color="#00695f", linestyle="-", marker="o"),
}
MODEL_LABELS = {"Naive": "朴素预测（Naive）", "XGBoost": "XGBoost 模型", "LSTM": "LSTM 模型"}


def save_figure(fig, output):
    if output is None:
        return fig
    fig.savefig(output, dpi=190, facecolor="white", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def plot_overall_improvement(output=None, results=RESULTS):
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.5), sharey=True, layout="constrained")
    fig.suptitle("相对朴素预测的整体误差改善比例", fontsize=17, weight="bold")
    all_improvements = []
    for col, ax in enumerate(axes):
        for offset, name in zip((-0.18, 0.18), ("XGBoost", "LSTM")):
            improvements = []
            for stages in results.values():
                models = list(stages.values())[col]
                baseline = np.mean(models["Naive"])
                improvements.append(100 * (baseline - np.mean(models[name])) / baseline)
            bars = ax.bar(np.arange(2) + offset, improvements, width=0.32,
                          color=STYLES[name]["color"], label=MODEL_LABELS[name])
            all_improvements.extend(improvements)
            ax.bar_label(bars, labels=[f"{value:.2f}%" for value in improvements], padding=4)
        ax.set(title=("验证集", "测试集")[col], xticks=range(2),
               xticklabels=["未来 5 个交易日", "未来 20 个交易日"])
        ax.set_axisbelow(True)
        ax.grid(axis="y", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("整体 MAE 下降比例（%）")
    low, high = min(0, min(all_improvements)), max(0, max(all_improvements))
    pad = max((high - low) * 0.15, 0.1)
    axes[0].set_ylim(low - pad if low < 0 else 0, high + pad)
    axes[0].legend(frameon=False)
    return save_figure(fig, output)


def plot_step_improvement(output=None, results=RESULTS):
    fig, axes = plt.subplots(2, 1, figsize=(13, 6.5), layout="constrained")
    fig.suptitle("各预测步长相对朴素预测的误差改善", fontsize=17, weight="bold")
    for ax, (horizon, stages) in zip(axes, results.items()):
        labels, rows = [], []
        for stage, models in stages.items():
            for name in ("XGBoost", "LSTM"):
                label = "验证集" if stage.startswith("Validation") else "测试集"
                labels.append(f"{label} / {MODEL_LABELS[name]}")
                rows.append(np.array(models["Naive"]) - np.array(models[name]))
        values = np.stack(rows)
        limit = np.max(np.abs(values))
        heatmap = ax.imshow(values, cmap="RdYlGn", vmin=-limit, vmax=limit, aspect="auto")
        for (row, col), value in np.ndenumerate(values):
            color = "white" if abs(value) > 0.65 * limit else "#202b33"
            ax.text(col, row, f"{value:.2f}", ha="center", va="center", color=color, fontsize=8)
        ax.set(title=f"未来 {horizon} 个交易日", xlabel="预测步长（第几个交易日）",
               xticks=range(horizon), xticklabels=range(1, horizon + 1),
               yticks=range(len(labels)), yticklabels=labels)
        fig.colorbar(heatmap, ax=ax, label="朴素预测 MAE − 模型 MAE（指数点）")
    return save_figure(fig, output)


def main():
    folder = Path(__file__).resolve().parents[2] / "reports/1_TimeSerialAnalyze"
    folder.mkdir(parents=True, exist_ok=True)
    plot_overall_improvement(folder / "overall_mae_improvement.png")
    plot_step_improvement(folder / "step_mae_improvement_heatmap.png")


if __name__ == "__main__":
    main()
