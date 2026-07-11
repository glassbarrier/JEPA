"""训练日志分析脚本

读取 hybrid-training.log（CSVLogger 输出），量化训练趋势，
帮助判断：是否坍塌 / 是否停滞 / SIGReg 是否生效 / 是否需要调参。

用法:
    python scripts/analyze_training_log.py --log checkpoints/industrial_detection/hybrid-training.log
    python scripts/analyze_training_log.py --log <path> --plot curves.png
"""

import argparse
import csv
import re
from collections import defaultdict


def parse_log(path):
    """解析日志，跳过重复的表头行，返回按 epoch 聚合的数据。"""
    rows = []
    header_re = re.compile(r"^epoch,itr,total_loss,pred_loss,sigreg_loss,time")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if header_re.match(line):
                continue
            parts = line.split(",")
            if len(parts) < 6:
                continue
            try:
                epoch = int(float(parts[0]))
                itr = int(float(parts[1]))
                total = float(parts[2])
                pred = float(parts[3])
                sig = float(parts[4])
                t = float(parts[5])
            except ValueError:
                continue
            rows.append({
                "epoch": epoch, "itr": itr,
                "total": total, "pred": pred, "sig": sig, "time": t,
            })
    return rows


def summarize(rows):
    by_epoch = defaultdict(list)
    for r in rows:
        by_epoch[r["epoch"]].append(r)

    print("=" * 64)
    print("训练日志分析")
    print("=" * 64)
    print(f"总记录数: {len(rows)}")
    if not rows:
        print("无有效数据")
        return

    epochs = sorted(by_epoch.keys())
    print(f"Epoch 范围: {epochs[0]} ~ {epochs[-1]}")

    # 每 epoch 平均
    print("\n%-6s %10s %10s %10s %10s" % ("epoch", "pred_avg", "pred_min", "sig_avg", "ms_avg"))
    first_pred = None
    last_pred = None
    for e in epochs:
        rs = by_epoch[e]
        pavg = sum(x["pred"] for x in rs) / len(rs)
        pmin = min(x["pred"] for x in rs)
        savg = sum(x["sig"] for x in rs) / len(rs)
        tavg = sum(x["time"] for x in rs) / len(rs)
        print("%-6d %10.3f %10.3f %10.4f %10.1f" % (e, pavg, pmin, savg, tavg))
        if first_pred is None:
            first_pred = pavg
        last_pred = pavg

    # 关键诊断
    print("\n--- 诊断 ---")
    # 1. 坍塌: pred 接近 0
    global_min = min(x["pred"] for x in rows)
    if global_min < 0.01:
        print("[警告] 预测损失接近 0 -> 可能发生表示坍塌！")
    else:
        print("[OK] 未检测到表示坍塌 (pred 最小值 %.3f)" % global_min)

    # 2. 爆炸 / NaN
    bad = [x for x in rows if not (x["pred"] == x["pred"]) or abs(x["pred"]) > 1e4]
    if bad:
        print("[警告] 检测到 %d 条异常( NaN / 爆炸 )!" % len(bad))
    else:
        print("[OK] 无 NaN / 爆炸")

    # 3. 停滞: 后期变化很小
    if len(epochs) >= 10:
        early = sum(x["pred"] for x in by_epoch[epochs[0]]) / len(by_epoch[epochs[0]])
        mid = by_epoch[epochs[len(epochs)//2]]
        mid_avg = sum(x["pred"] for x in mid) / len(mid)
        late = by_epoch[epochs[-1]]
        late_avg = sum(x["pred"] for x in late) / len(late)
        drop_early = early - mid_avg
        drop_late = mid_avg - late_avg
        print("[趋势] epoch1 平均 pred=%.2f, 中期=%.2f, 末期=%.2f" % (early, mid_avg, late_avg))
        if drop_late < 0.05 * max(1.0, drop_early):
            print("  -> 后期基本停滞 (下降幅度 << 前期)，学习率可能过早衰减到 1e-6")
        else:
            print("  -> 后期仍有下降")

    # 4. SIGReg 是否生效
    sig_first = by_epoch[epochs[0]][0]["sig"]
    sig_last = by_epoch[epochs[-1]][-1]["sig"]
    print("[SIGReg] 首=%.4f, 末=%.4f, 变化=%.4f" % (sig_first, sig_last, sig_last - sig_first))
    if abs(sig_last - sig_first) < 0.02:
        print("  -> SIGReg 几乎没动，正则项影响很弱 (对照论文应早期快速下降后平台)")

    # 5. 总损失构成
    last = by_epoch[epochs[-1]][-1]
    print("\n[损失构成(末期)] pred=%.3f, sigreg=%.3f, 总=%.3f" % (
        last["pred"], last["sig"], last["total"]))
    print("  -> 总损失几乎全来自预测损失; sigreg 用 weight=0.09 仅占约 %.3f" % (0.09 * last["sig"]))

    print("\n--- 建议 ---")
    if last_pred and first_pred and (first_pred - last_pred) / max(1.0, first_pred) < 0.3:
        print("1. 损失下降有限(<30%)，考虑: 增大 lr 或缩短 warmup、改 cosine 末值 final_lr 不要过小")
        print("2. 对照严格审阅: 当前是 I-JEPA 掩码预测 + stop-gradient 目标，并非 LeWM 时序下一帧预测")
        print("3. 若目标是 LeWM, 应改为 teacher-forcing 在线目标 + 帧序列输入 + BatchNorm 投影")
    print("4. 先做下游probe (线性/LP) 验证表征是否有用，再决定是否加训")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="checkpoints/industrial_detection/hybrid-training.log")
    ap.add_argument("--plot", default=None, help="可选: 输出 PNG 曲线")
    args = ap.parse_args()

    rows = parse_log(args.log)
    summarize(rows)

    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import numpy as np
        except Exception as e:
            print("绘图失败 (缺 matplotlib):", e)
            return
        by_epoch = defaultdict(list)
        for r in rows:
            by_epoch[r["epoch"]].append(r)
        epochs = sorted(by_epoch.keys())
        pred_avg = [sum(x["pred"] for x in by_epoch[e]) / len(by_epoch[e]) for e in epochs]
        pred_min = [min(x["pred"] for x in by_epoch[e]) for e in epochs]
        sig_avg = [sum(x["sig"] for x in by_epoch[e]) / len(by_epoch[e]) for e in epochs]

        fig, axes = plt.subplots(1, 3, figsize=(21, 5))

        # 面板1: pred_loss vs epoch (均值 + 最优包络)
        ax = axes[0]
        ax.plot(epochs, pred_avg, "b-", label="pred_mean")
        ax.fill_between(epochs, pred_min, pred_avg, color="b", alpha=0.15)
        ax.plot(epochs, pred_min, "b--", alpha=0.6, label="pred_min")
        ax.set_title("Prediction Loss vs Epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel("pred_loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 面板2: sigreg_loss vs epoch
        ax = axes[1]
        ax.plot(epochs, sig_avg, "r-", label="sigreg_mean")
        ax.set_title("SIGReg Loss vs Epoch")
        ax.set_xlabel("epoch")
        ax.set_ylabel("sigreg_loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 面板3: 前期逐迭代平滑曲线 (看早期动态/是否震荡)
        ax = axes[2]
        niters = min(3000, len(rows))
        it = [r["itr"] + (r["epoch"] - epochs[0]) * 100000 for r in rows[:niters]]
        # 用全局迭代序号(每 epoch 近似 600 it)做 x
        x = list(range(niters))
        y = [r["pred"] for r in rows[:niters]]
        y = np.array(y, dtype=float)
        if len(y) > 50:
            k = 50
            kernel = np.ones(k) / k
            y_smooth = np.convolve(y, kernel, mode="same")
            ax.plot(x, y, "gray", alpha=0.25, label="raw")
            ax.plot(x, y_smooth, "b-", label="smoothed(k=50)")
        else:
            ax.plot(x, y, "b-", label="pred")
        ax.set_title("Early Training Dynamics (per-iter pred_loss)")
        ax.set_xlabel("iteration (first %d)" % niters)
        ax.set_ylabel("pred_loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        fig.suptitle("Hybrid IJEPA-LeWM Training Analysis", fontsize=13)
        fig.tight_layout()
        fig.savefig(args.plot, dpi=120)
        print("曲线已保存:", args.plot)


if __name__ == "__main__":
    main()
