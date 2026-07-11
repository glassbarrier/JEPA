"""下游评估脚本（线性探针 / kNN / 异常检测 AUROC）

加载预训练权重，抽取 [CLS] 表征，在下游任务上衡量表征质量。
适用于你的 12 类工业缺陷数据集，也适用于 MVTec AD（train 仅 good）。

用法（服务器上）:
    python scripts/eval_downstream.py \
        --checkpoint ./checkpoints/industrial_detection/weights/hybrid-epoch-10.pth \
        --data_root /home/jovyan/UCAD/mvtec2d \
        --config configs/train/industrial_detection.yaml \
        --out ./checkpoints/industrial_detection/eval_epoch10.json

说明:
    - 多分类探针: 仅当某 split 含 >1 类时计算（你的 12 类数据 train 含 good+缺陷）。
    - 异常检测 AUROC: 始终计算。good=0, 缺陷=1；用训练 good 特征拟合高斯，
      以马氏距离打分，对测试集算 image-level AUROC（MVTec 标准做法）。
"""

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.hybrid_jepa import HybridJEPA
from src.models.vision_transformer import vit_tiny, vit_predictor
from src.datasets.industrial_detection import IndustrialDetectionDataset

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.model_selection import train_test_split
    _HAS_SK = True
except ImportError:
    _HAS_SK = False


def build_model(device):
    """重建与训练一致的 HybridJEPA（绕过 helper.py）。"""
    encoder = vit_tiny(patch_size=16, embed_dim=192)
    predictor = vit_predictor(
        num_patches=encoder.patch_embed.num_patches,
        embed_dim=192,
        predictor_embed_dim=192,
        depth=6,
        num_heads=12,
    )
    model = HybridJEPA(
        encoder=encoder,
        predictor=predictor,
        use_sigreg=True,
        sigreg_weight=0.09,
    )
    return model.to(device)


def eval_transform(target_size=224):
    import torchvision.transforms as T
    return T.Compose([
        T.Resize(target_size),
        T.CenterCrop(target_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                   std=[0.229, 0.224, 0.225]),
    ])


def extract_features(model, dataset, device, batch_size=32, use_cls_token=True):
    """抽取 [CLS] 表征，返回 (X, y_idx, y_binary)。"""
    model.eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=4, pin_memory=True)
    feats, idxs, binaries = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            info = model.encode({"pixels": imgs}, use_cls_token=use_cls_token)
            emb = info["emb"]  # (B, 1, D) 或 (B, D)
            if emb.dim() == 3:
                emb = emb[:, 0, :]
            feats.append(emb.cpu().numpy())
            idxs.append(labels.numpy())
            # 二元标签: good=0, 其余(缺陷)=1
            bin_vec = np.array([
                0 if dataset.idx_to_label[int(l)] == "good" else 1 for l in labels
            ])
            binaries.append(bin_vec)
    X = np.concatenate(feats, axis=0)
    y_idx = np.concatenate(idxs, axis=0)
    y_bin = np.concatenate(binaries, axis=0)
    return X, y_idx, y_bin


def _gaussian_anomaly_auc(X_train_good, X_test, y_test_bin):
    """在训练 good 特征上拟合高斯，用马氏距离对测试打分，返回 AUROC。"""
    mean = X_train_good.mean(axis=0)
    # 加 ridge 防止奇异
    cov = np.cov(X_train_good, rowvar=False) + 1e-6 * np.eye(X_train_good.shape[1])
    inv_cov = np.linalg.pinv(cov)
    d = X_test - mean
    mahal = np.sum((d @ inv_cov) * d, axis=1)  # 越大越异常
    # roc_auc 期望缺陷(y=1)分数更高 -> 用 -mahal (缺陷马氏距离大, -mahal 小? 取负)
    # 直接用 mahal 作为异常分数: 缺陷 -> 高; roc_auc_score(y_true, score)
    if len(np.unique(y_test_bin)) < 2:
        return None
    return float(roc_auc_score(y_test_bin, mahal))


def evaluate_category(model, category, data_root, device, batch_size):
    """对单个类别计算异常检测 AUROC（及多分类探针若可用）。"""
    tf = eval_transform()
    train_ds = IndustrialDetectionDataset(data_root, categories=[category],
                                          split="train", transform=tf)
    test_ds = IndustrialDetectionDataset(data_root, categories=[category],
                                         split="test", transform=tf)
    if len(train_ds) == 0 or len(test_ds) == 0:
        return None

    Xtr, ytr_idx, ytr_bin = extract_features(model, train_ds, device, batch_size)
    Xte, yte_idx, yte_bin = extract_features(model, test_ds, device, batch_size)

    res = {"category": category, "n_train": len(train_ds), "n_test": len(test_ds)}

    # 异常检测 AUROC（始终计算）
    train_good = Xtr[ytr_bin == 0]
    if len(train_good) == 0:
        train_good = Xtr  # 无 good 时退化为全体
    auc = _gaussian_anomaly_auc(train_good, Xte, yte_bin)
    res["anomaly_auc"] = auc

    # 多分类探针（仅当训练含 >1 类时）
    res["linear_acc"] = None
    res["knn_acc"] = None
    if _HAS_SK and len(np.unique(ytr_idx)) > 1:
        Xtr_s = StandardScaler().fit_transform(Xtr)
        Xte_s = StandardScaler().fit_transform(Xte)
        # 用训练集自己切一分做早检，避免对 test 过拟合报告
        Xa, Xb, ya, yb = train_test_split(Xtr_s, ytr_idx, test_size=0.2,
                                          random_state=0, stratify=ytr_idx)
        clf = LogisticRegression(max_iter=2000)
        clf.fit(Xa, ya)
        res["linear_acc"] = float(accuracy_score(yb, clf.predict(Xb)))
        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(Xa, ya)
        res["knn_acc"] = float(accuracy_score(yb, knn.predict(Xb)))
    return res


def run_downstream_eval(checkpoint_path, data_root, config_path=None,
                        categories=None, batch_size=32, device=None,
                        out_path=None):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if not _HAS_SK:
        raise RuntimeError("需要 scikit-learn: pip install scikit-learn")

    print(f"[eval] device={device}")
    print(f"[eval] loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = build_model(device)
    sd = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(sd)
    print(f"[eval] model loaded (epoch={ckpt.get('epoch', '?')})")

    # 解析数据根: 支持 config.data.root_path 或 image_folder
    if data_root is None and config_path and os.path.exists(config_path):
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        data_root = cfg.get("data", {}).get("root_path") or cfg.get("data", {}).get("image_folder")

    # 收集类别
    if categories is None:
        root = __import__("pathlib").Path(data_root)
        categories = [d.name for d in root.iterdir()
                      if d.is_dir() and not d.name.startswith(".")]

    results = []
    for cat in categories:
        r = evaluate_category(model, cat, data_root, device, batch_size)
        if r is not None:
            results.append(r)
            a = r["anomaly_auc"]
            la = r["linear_acc"]
            print(f"  {cat:24s} AUC={a if a is None else round(a,4)} "
                  f"linear={la if la is None else round(la,4)}")

    if not results:
        print("[eval] 无可用类别")
        return {}

    aucs = [r["anomaly_auc"] for r in results if r["anomaly_auc"] is not None]
    summary = {
        "checkpoint": checkpoint_path,
        "n_categories": len(results),
        "mean_anomaly_auc": float(np.mean(aucs)) if aucs else None,
        "per_category": results,
    }
    print(f"\n[eval] 平均异常检测 AUROC = {summary['mean_anomaly_auc']}")

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"[eval] 结果已保存: {out_path}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data_root", default=None)
    ap.add_argument("--config", default="configs/train/industrial_detection.yaml")
    ap.add_argument("--categories", nargs="*", default=None)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    run_downstream_eval(args.checkpoint, args.data_root, args.config,
                        args.categories, args.batch_size, out_path=args.out)


if __name__ == "__main__":
    main()
