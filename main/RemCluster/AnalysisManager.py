# AnalysisManager.py
# CPU streaming clustering for REM features (epoch-level & patch-level)
# - Handles shapes:
#   (61,1536): row0 is CLS (time+freq), rows 1..60 are patches
#   (60,1536): patches flattened; original order = (channel, patch, embedding)
#   (15,4,2,768): 15 patches × 4 channels × 2 domains × 768
# - Restores to (4,15,1536) correctly (channel, patch, embedding).
# - Provides epoch-level and patch-level pipelines, diagnostics and plots.

import os, re, glob, h5py, json, time, logging, argparse, random
import numpy as np
from typing import List, Tuple, Dict, Optional

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import joblib

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


class AnalysisManager:
    def __init__(self,
                 features_dir: str,
                 pattern: str = "*.h5",
                 order: str = "patch_channel",   # 保留参数以兼容旧调用
                 seed: int = 42):
        self.features_dir = features_dir
        self.pattern = pattern
        self.order = order
        self.seed = seed
        self.rng = random.Random(seed)

        self.paths: List[str] = sorted(glob.glob(os.path.join(features_dir, pattern)))
        if not self.paths:
            raise FileNotFoundError(f"No files matched: {features_dir}/{pattern}")

        self._epoch_pat = re.compile(r"^epoch_(\d+)$")
        self._total_epochs: Optional[int] = None
        self._rs_count = 0
        self.logger: Optional[logging.Logger] = None
        self.save_dir = ""

    # ---------------- logging ----------------
    def _init_logger(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        logger = logging.getLogger("AnalysisManager")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))

        fh = logging.FileHandler(os.path.join(save_dir, "run.log"), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"))

        logger.addHandler(ch)
        logger.addHandler(fh)
        self.logger = logger

    def log(self, msg: str):
        print(msg)
        if self.logger:
            self.logger.info(msg)

    # ---------------- I/O ----------------
    def iter_epochs_raw(self):
        """Deterministic iteration: yield (subject_id, epoch_id, raw_array)."""
        for p in self.paths:
            sid = os.path.splitext(os.path.basename(p))[0]
            with h5py.File(p, "r") as f:
                keys = []
                for k in f.keys():
                    m = self._epoch_pat.match(k)
                    if m:
                        keys.append((int(m.group(1)), k))
                keys.sort(key=lambda t: t[0])
                for eid, kk in keys:
                    yield sid, eid, f[kk][()]

    def count_epochs(self) -> int:
        if self._total_epochs is not None:
            return self._total_epochs
        total = 0
        for p in self.paths:
            with h5py.File(p, "r") as f:
                total += sum(1 for k in f.keys() if self._epoch_pat.match(k))
        self._total_epochs = total
        return total

    # ---------------- shape restore (CRITICAL) ----------------
    def to_channels_patches1536(self, raw: np.ndarray) -> np.ndarray:
        """
        Restore to (4,15,1536) = (channel, patch, embedding).
        Assumptions:
          - During saving, 4×15×1536 was flattened to (60,1536) WITHOUT transpose.
          - Row 0 of (61,1536) is CLS, rows 1..60 are patches in the same flattened order.
        Supported inputs:
          * (61,1536) -> drop CLS -> reshape(4,15,1536)
          * (60,1536) -> reshape(4,15,1536)
          * (15,4,2,768) -> concat domains -> (15,4,1536) -> transpose to (4,15,1536)
        """
        if raw.shape == (61, 1536):
            raw = raw[1:, :]  # remove CLS -> (60,1536)

        if raw.shape == (60, 1536):
            return raw.reshape(4, 15, 1536)

        if raw.shape == (15, 4, 2, 768):
            v = np.concatenate([raw[..., 0, :], raw[..., 1, :]], axis=-1)  # (15,4,1536)
            return v.transpose(1, 0, 2)  # (4,15,1536)

        raise ValueError(f"Unsupported raw shape: {raw.shape}")

    # ---------------- feature builders ----------------
    def vec_concat1536(self, raw: np.ndarray) -> np.ndarray:
        """
        Epoch → 1536 vector by averaging across channel & patch, while keeping domains concatenated.
        """
        v = self.to_channels_patches1536(raw)  # (4,15,1536)
        return v.mean(axis=(0, 1))            # (1536,)

    def vec_mean768(self, raw: np.ndarray) -> np.ndarray:
        """
        Epoch → 768 vector by averaging time/freq halves after averaging channel & patch.
        """
        v = self.to_channels_patches1536(raw)  # (4,15,1536)
        time = v[..., :768].mean(axis=(0, 1))  # (768,)
        freq = v[..., 768:].mean(axis=(0, 1))  # (768,)
        return 0.5 * (time + freq)             # (768,)

    def patches_matrix(self, raw: np.ndarray, patch_feat: str = "concat1536") -> np.ndarray:
        """
        Return patch matrix of shape (4,15,D).
        - patch_feat == "concat1536": D=1536 (time||freq)
        - patch_feat == "mean768"   : D=768  (avg over time/freq)
        """
        v = self.to_channels_patches1536(raw)  # (4,15,1536)
        if patch_feat == "concat1536":
            return v  # (4,15,1536)
        elif patch_feat == "mean768":
            time = v[..., :768]
            freq = v[..., 768:]
            return 0.5 * (time + freq)  # (4,15,768)
        else:
            raise ValueError(f"Unknown patch_feat: {patch_feat}")

    def patches_15xD(self, raw: np.ndarray, patch_feat: str = "mean768",
                     channel_agg: str = "mean") -> np.ndarray:
        """
        Patch-level vector per epoch: (15, D).
        Build from (4,15,D) by aggregating channel.
        channel_agg: "mean" | "none"
          - "mean": average across channels -> (15,D)
          - "none": return (4,15,D) (caller handles reshape/stack)
        """
        v = self.patches_matrix(raw, patch_feat=patch_feat)  # (4,15,D)
        if channel_agg == "mean":
            return v.mean(axis=0)    # (15,D)
        elif channel_agg == "none":
            return v                 # (4,15,D)
        else:
            raise ValueError(f"Unknown channel_agg: {channel_agg}")

    # ---------------- reservoir sampling for plots ----------------
    def _reservoir_append(self, feats: list, idxs: list, vec, idx: int, cap: int):
        self._rs_count += 1
        if len(feats) < cap:
            feats.append(vec); idxs.append(idx)
        else:
            j = self.rng.randrange(0, self._rs_count)
            if j < cap:
                feats[j] = vec; idxs[j] = idx

    # ---------------- save helpers ----------------
    def _save_epoch_labels(self, save_dir: str, labels: np.ndarray,
                           order_index: List[Tuple[str, int]]):
        if len(labels) != len(order_index):
            raise ValueError("labels length != order_index length")
        np.save(os.path.join(save_dir, "labels.npy"), labels)
        with open(os.path.join(save_dir, "index.csv"), "w", encoding="utf-8") as f:
            f.write("subject,epoch\n")
            for sid, eid in order_index:
                f.write(f"{sid},{eid}\n")
        np.save(os.path.join(save_dir, "subject.npy"),
                np.array([s for s, _ in order_index], dtype=object))
        np.save(os.path.join(save_dir, "epoch.npy"),
                np.array([e for _, e in order_index], dtype=np.int32))

    def _save_patch_labels(self, save_dir: str, labels: np.ndarray,
                           order_index: List[Tuple[str, int, int]]):
        """
        Patch-level: labels aligned with [(sid, eid, pid), ...], pid in [0..14]
        """
        if len(labels) != len(order_index):
            raise ValueError("labels length != order_index length (patch)")
        np.save(os.path.join(save_dir, "patch_labels.npy"), labels)
        with open(os.path.join(save_dir, "patch_index.csv"), "w", encoding="utf-8") as f:
            f.write("subject,epoch,patch\n")
            for sid, eid, pid in order_index:
                f.write(f"{sid},{eid},{pid}\n")

    # ---------------- generic epoch pipeline ----------------
    def _pipeline_generic_epoch(self, vec_fn, out_dim: int,
                                k=2, pca_dim=128, batch_rows=200_000, normalize=True):
        total = self.count_epochs()
        scaler = StandardScaler() if normalize else None

        # pass1: scaler
        if scaler is not None:
            buf = []
            self.log(f"\n[Pass1] Fit StandardScaler (dim={out_dim})")
            for _sid, _eid, raw in tqdm(self.iter_epochs_raw(), total=total, desc="Scaler", ncols=100):
                buf.append(vec_fn(raw))
                if len(buf) >= batch_rows:
                    scaler.partial_fit(np.stack(buf, 0)); buf = []
            if buf:
                scaler.partial_fit(np.stack(buf, 0)); buf = []

        # pass2: ipca
        ipca = IncrementalPCA(n_components=min(pca_dim, out_dim))
        buf = []
        self.log("[Pass2] Fit IncrementalPCA")
        for _sid, _eid, raw in tqdm(self.iter_epochs_raw(), total=total, desc="PCA", ncols=100):
            buf.append(vec_fn(raw))
            if len(buf) >= batch_rows:
                X = np.stack(buf, 0)
                if scaler is not None: X = scaler.transform(X)
                ipca.partial_fit(X); buf = []
        if buf:
            X = np.stack(buf, 0)
            if scaler is not None: X = scaler.transform(X)
            ipca.partial_fit(X); buf = []

        # pass3: kmeans
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=self.seed)
        buf = []
        self.log("[Pass3] Fit MiniBatchKMeans")
        for _sid, _eid, raw in tqdm(self.iter_epochs_raw(), total=total, desc="KMeans", ncols=100):
            buf.append(vec_fn(raw))
            if len(buf) >= batch_rows:
                X = np.stack(buf, 0)
                if scaler is not None: X = scaler.transform(X)
                Xp = ipca.transform(X)
                km.partial_fit(Xp); buf = []
        if buf:
            X = np.stack(buf, 0)
            if scaler is not None: X = scaler.transform(X)
            Xp = ipca.transform(X)
            km.partial_fit(Xp)

        # pass4: predict + order
        labels = np.empty(total, np.int8)
        order_index: List[Tuple[str, int]] = []
        buf = []; i = 0
        self.log("[Pass4] Predict")
        for sid, eid, raw in tqdm(self.iter_epochs_raw(), total=total, desc="Predict", ncols=100):
            buf.append(vec_fn(raw))
            order_index.append((sid, eid))
            if len(buf) >= batch_rows:
                X = np.stack(buf, 0)
                if scaler is not None: X = scaler.transform(X)
                Xp = ipca.transform(X)
                labels[i:i+len(buf)] = km.predict(Xp); i += len(buf); buf = []
        if buf:
            X = np.stack(buf, 0)
            if scaler is not None: X = scaler.transform(X)
            Xp = ipca.transform(X)
            labels[i:i+len(buf)] = km.predict(Xp)

        return scaler, ipca, km, labels, order_index

    def _pipeline_mean768(self, **kw):
        return self._pipeline_generic_epoch(self.vec_mean768, out_dim=768, **kw)

    def _pipeline_concat1536(self, **kw):
        return self._pipeline_generic_epoch(self.vec_concat1536, out_dim=1536, **kw)

    # ---------------- patch-level pipeline ----------------
    def _pipeline_patch_level(
            self,
            patch_feat: str = "mean768",  # "mean768" | "concat1536"
            channel_agg: str = "mean",  # "mean" | "none"
            k: int = 2,
            pca_dim: int = 64,
            batch_rows: int = 200_000,  # 累计到这么多行再做一次 partial_fit；会与 pca_dim 取 max
            normalize: bool = True,
            output_mode: str = "per_patch",  # "per_patch" | "per_channel_patch" | "both"
    ):
        """
        跨所有 subject/epoch 的 patch 级聚类。

        - patch_feat:
            * "mean768"     : 取 (time,freq) 平均后的 768 维
            * "concat1536"  : 保留拼接后的 1536 维
        - channel_agg:
            * "mean"  : 每 epoch 得到 [15, D]
            * "none"  : 每 epoch 得到 [4,15,D]，训练时按 (60,D) 压平；预测可输出：
                - per_channel_patch：60 个 label（索引含通道）
                - per_patch：对同 patch 的 4 通道做投票得到 15 个 label
        - output_mode:
            * "per_patch"（默认）只保存 15 个/epoch；
            * "per_channel_patch" 保存 60 个/epoch；
            * "both" 两者都保存（返回 per_patch 为主，同时把 per_channel_patch 也落盘为 .npz）
        """

        assert output_mode in ("per_patch", "per_channel_patch", "both")
        total_epochs = self.count_epochs()

        # --------- 工具：构建每个 epoch 的 patch 矩阵 ---------
        def build(raw: np.ndarray) -> np.ndarray:
            """
            返回：
              channel_agg == 'mean'  →  (15, D)
              channel_agg == 'none'  →  (60, D) （训练/拟合阶段直接压平）
            并记录 (对 'none' 的预测阶段) 的 reshape 形状以便还原 4×15。
            """
            M = self.patches_15xD(raw, patch_feat=patch_feat, channel_agg=channel_agg)
            if channel_agg == "mean":  # (15, D)
                return M
            else:  # (4,15,D) -> (60,D)
                D = M.shape[-1]
                return M.reshape(-1, D)

        # 维度 D
        # 用第一条样本探测
        first_raw = next(self.iter_epochs_raw())[2]
        P0 = build(first_raw)
        D = P0.shape[-1]
        if pca_dim > D:
            self.log(f"[warn] pca_dim={pca_dim} > feature_dim={D}, 改为 pca_dim={D}")
            pca_dim = D

        # --------- 模型 ---------
        scaler = StandardScaler() if normalize else None
        ipca = IncrementalPCA(n_components=pca_dim)
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=self.seed)

        # 每次 partial_fit 的最小样本行数（防止 < n_components）
        min_chunk = max(pca_dim, 8192)  # 8k 行起跳，按内存自行调整
        th_scaler = max(min_chunk, batch_rows)
        th_ipca = max(min_chunk, batch_rows)
        th_km = max(min_chunk, batch_rows)

        # =========================================================
        # Pass1: Scaler（按 patch 行累积）
        # =========================================================
        if scaler is not None:
            buf, rows = [], 0
            self.log(f"\n[Pass1] Fit StandardScaler (patch-level, feat={patch_feat}, agg={channel_agg})")
            for _sid, _eid, raw in tqdm(self.iter_epochs_raw(), total=total_epochs, desc="Scaler(patch)", ncols=100):
                P = build(raw)  # (15,D) or (60,D)
                buf.append(P);
                rows += P.shape[0]
                if rows >= th_scaler:
                    X = np.vstack(buf)  # (N, D)
                    scaler.partial_fit(X)
                    buf, rows = [], 0
            if rows > 0:
                X = np.vstack(buf)
                scaler.partial_fit(X)

        # =========================================================
        # Pass2: IPCA（按 patch 行累积；批大小 ≥ pca_dim）
        # =========================================================
        self.log("[Pass2] Fit IncrementalPCA (patch-level)")
        buf, rows = [], 0
        for _sid, _eid, raw in tqdm(self.iter_epochs_raw(), total=total_epochs, desc="PCA(patch)", ncols=100):
            P = build(raw)
            if scaler is not None: P = scaler.transform(P)
            buf.append(P);
            rows += P.shape[0]
            if rows >= th_ipca:
                X = np.vstack(buf)
                ipca.partial_fit(X)
                buf, rows = [], 0
        if rows >= pca_dim:
            X = np.vstack(buf)
            ipca.partial_fit(X)
        elif rows > 0:
            self.log(f"[warn] tail rows {rows} < n_components={pca_dim}; skip last IPCA chunk.")

        # =========================================================
        # Pass3: MiniBatchKMeans（按 patch 行累积）
        # =========================================================
        self.log("[Pass3] Fit MiniBatchKMeans (patch-level)")
        buf, rows = [], 0
        for _sid, _eid, raw in tqdm(self.iter_epochs_raw(), total=total_epochs, desc="KMeans(patch)", ncols=100):
            P = build(raw)
            if scaler is not None: P = scaler.transform(P)
            Xp = ipca.transform(P)
            buf.append(Xp);
            rows += Xp.shape[0]
            if rows >= th_km:
                km.partial_fit(np.vstack(buf))
                buf, rows = [], 0
        if rows > 0:
            km.partial_fit(np.vstack(buf))

        # =========================================================
        # Pass4: 预测 + 索引
        # =========================================================
        self.log("[Pass4] Predict (patch-level)")
        per_patch_labels: List[int] = []
        per_patch_index: List[Tuple[str, int, int]] = []  # (sid, eid, pid)

        per_ch_patch_labels: List[int] = []  # 可选
        per_ch_patch_index: List[Tuple[str, int, int, int]] = []  # (sid, eid, ch, pid)

        for sid, eid, raw in tqdm(self.iter_epochs_raw(), total=total_epochs, desc="Predict(patch)", ncols=100):
            if channel_agg == "mean":
                # (15, D) → 15 labels
                P = self.patches_15xD(raw, patch_feat=patch_feat, channel_agg="mean")  # (15,D)
                P2 = scaler.transform(P) if scaler is not None else P
                Xp = ipca.transform(P2)
                labs = km.predict(Xp)  # (15,)
                per_patch_labels.extend(labs.tolist())
                for pid in range(15):
                    per_patch_index.append((sid, eid, pid))

            else:
                # channel_agg == "none": (4,15,D)
                M = self.patches_15xD(raw, patch_feat=patch_feat, channel_agg="none")  # (4,15,D)
                P = M.reshape(-1, M.shape[-1])  # (60,D)
                P2 = scaler.transform(P) if scaler is not None else P
                Xp = ipca.transform(P2)
                labs60 = km.predict(Xp).reshape(4, 15)  # (4,15)

                if output_mode in ("per_channel_patch", "both"):
                    per_ch_patch_labels.extend(labs60.ravel().tolist())
                    for ch in range(4):
                        for pid in range(15):
                            per_ch_patch_index.append((sid, eid, ch, pid))

                if output_mode in ("per_patch", "both"):
                    # 多数投票 → 15
                    maj = np.apply_along_axis(lambda a: np.bincount(a).argmax(), 0, labs60)  # (15,)
                    per_patch_labels.extend(maj.tolist())
                    for pid in range(15):
                        per_patch_index.append((sid, eid, pid))

        # ------- 返回与保存（以 per_patch 为主）-------
        labels_arr = np.array(per_patch_labels, dtype=np.int16)
        # 索引保存：主 labels（per_patch）
        np.save(os.path.join(self.save_dir, "patch_labels.npy"), labels_arr)
        with open(os.path.join(self.save_dir, "patch_index.csv"), "w", encoding="utf-8") as fw:
            fw.write("subject,epoch,patch\n")
            for s, e, p in per_patch_index:
                fw.write(f"{s},{e},{p}\n")

        # 如果需要 per_channel_patch，一并落盘（npz + csv）
        if output_mode in ("per_channel_patch", "both"):
            labels_ch = np.array(per_ch_patch_labels, dtype=np.int16)
            np.save(os.path.join(self.save_dir, "patch_labels_4x15.npy"), labels_ch)
            with open(os.path.join(self.save_dir, "patch_index_4x15.csv"), "w", encoding="utf-8") as fw:
                fw.write("subject,epoch,channel,patch\n")
                for s, e, ch, p in per_ch_patch_index:
                    fw.write(f"{s},{e},{ch},{p}\n")

        return scaler, ipca, km, labels_arr, per_patch_index

    # ---------------- visualization ----------------
    def visualize_kmeans_pca2d_epoch(self, save_dir: str, mode: str, sample_n: int = 200_000):
        import matplotlib.pyplot as plt
        labels = np.load(os.path.join(save_dir, "labels.npy"))
        pca = joblib.load(os.path.join(save_dir, "pca.pkl"))
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        feats, idxs = [], []
        self._rs_count = 0
        total = self.count_epochs()
        self.log("\n[Plot] PCA 2D (epoch-level)")
        for i, (_sid, _eid, raw) in enumerate(tqdm(self.iter_epochs_raw(), total=total, desc="Sample(PCA-epoch)", ncols=100)):
            if mode == "mean768":
                v = self.vec_mean768(raw)
            else:
                v = self.vec_concat1536(raw)
            self._reservoir_append(feats, idxs, v, i, cap=sample_n)

        X = np.stack(feats, 0)
        if scaler is not None:
            X = scaler.transform(X)
        Xp = pca.transform(X)[:, :2]
        y = labels[np.array(idxs, dtype=int)]

        plt.figure(figsize=(8, 8))
        plt.scatter(Xp[:, 0], Xp[:, 1], c=y, s=3, alpha=0.7, cmap="tab10")
        plt.title(f"Epoch KMeans@PCA · {mode} · n={len(y)}")
        plt.tight_layout()
        out = os.path.join(save_dir, f"kmeans_pca2d_epoch_{mode}.png")
        plt.savefig(out, dpi=300); plt.close()
        self.log(f"[plot] Saved: {out}")

    def visualize_umap_epoch(self, save_dir: str, mode: str,
                             sample_n: int = 100_000,
                             use_training_space: bool = True,
                             pca_dim_local: int = 64):
        if not HAS_UMAP:
            self.log("[warn] umap-learn not installed; skip UMAP.")
            return
        import matplotlib.pyplot as plt

        # sample epoch vectors
        feats, idxs = [], []
        self._rs_count = 0
        total = self.count_epochs()

        def _vec(raw):
            if mode == "mean768":
                return self.vec_mean768(raw)
            return self.vec_concat1536(raw)

        for i, (_sid, _eid, raw) in enumerate(tqdm(self.iter_epochs_raw(), total=total, desc="Sample(UMAP-epoch)", ncols=100)):
            v = _vec(raw)
            self._reservoir_append(feats, idxs, v, i, cap=sample_n)

        X = np.stack(feats, 0).astype(np.float32, copy=False)
        idxs = np.array(idxs, dtype=np.int64)

        # project to training space (scaler + pca) if available
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        pca_path = os.path.join(save_dir, "pca.pkl")
        used_space = "raw"
        X_u = X

        if use_training_space and os.path.exists(pca_path):
            try:
                pca = joblib.load(pca_path)
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    X_u = scaler.transform(X_u)
                    used_space = "scaler+pca"
                else:
                    used_space = "pca-only"
                X_u = pca.transform(X_u)
            except Exception as e:
                self.log(f"[warn] load training pca/scaler failed: {e}; fallback to local.")
                use_training_space = False

        if not use_training_space:
            scaler_local = StandardScaler().fit(X_u)
            Xs = scaler_local.transform(X_u)
            if pca_dim_local and pca_dim_local < Xs.shape[1]:
                X_u = PCA(n_components=pca_dim_local, random_state=self.seed).fit_transform(Xs)
                used_space = f"local(scaler+pca{pca_dim_local})"
            else:
                X_u = Xs
                used_space = "local(scaler-only)"

        reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=self.seed)
        emb = reducer.fit_transform(X_u)

        labels_path = os.path.join(save_dir, "labels.npy")
        y = None
        if os.path.exists(labels_path):
            y_all = np.load(labels_path)
            if idxs.max() < len(y_all):
                y = y_all[idxs]

        plt.figure(figsize=(8, 8))
        if y is None:
            plt.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.7)
        else:
            plt.scatter(emb[:, 0], emb[:, 1], c=y, s=3, alpha=0.7, cmap="tab10")
        plt.title(f"UMAP 2D (epoch) · {mode} · n={emb.shape[0]} · space={used_space}")
        plt.tight_layout()
        out = os.path.join(save_dir, f"umap2d_epoch_{mode}.png")
        plt.savefig(out, dpi=300); plt.close()
        self.log(f"[plot] Saved: {out}")

    # ---------------- diagnostics ----------------
    def diagnose_epoch(self, save_dir: str, mode: str,
                       sample_n: int = 100_000,
                       metrics_sample_n: int = 100_000):
        """
        Compute internal metrics (epoch-level) in PCA space used for KMeans.
        """
        pca_path = os.path.join(save_dir, "pca.pkl")
        km_path = os.path.join(save_dir, "kmeans.pkl")
        lab_path = os.path.join(save_dir, "labels.npy")
        if not (os.path.exists(pca_path) and os.path.exists(km_path)):
            self.log("[diagnose] missing models; skip")
            return

        pca = joblib.load(pca_path)
        km = joblib.load(km_path)
        scaler = joblib.load(os.path.join(save_dir, "scaler.pkl")) if os.path.exists(os.path.join(save_dir, "scaler.pkl")) else None
        labels = np.load(lab_path) if os.path.exists(lab_path) else None

        # sample
        feats, idxs = [], []
        self._rs_count = 0
        total = self.count_epochs()

        def _vec(raw):
            if mode == "mean768":
                return self.vec_mean768(raw)
            return self.vec_concat1536(raw)

        for i, (_sid, _eid, raw) in enumerate(tqdm(self.iter_epochs_raw(), total=total, desc="Diag-sample", ncols=100)):
            v = _vec(raw)
            self._reservoir_append(feats, idxs, v, i, cap=max(sample_n, metrics_sample_n))

        X = np.stack(feats, 0).astype(np.float32, copy=False)
        idxs = np.array(idxs, dtype=np.int64)
        if scaler is not None:
            X = scaler.transform(X)
        Xp = pca.transform(X)
        if labels is not None and idxs.max() < len(labels):
            y = labels[idxs]
        else:
            y = km.predict(Xp)

        msel = min(metrics_sample_n, Xp.shape[0])
        Xm, ym = Xp[:msel], y[:msel]

        metrics = {}
        metrics["n_samples_total"] = int(total)
        metrics["n_samples_sampled"] = int(Xp.shape[0])
        metrics["n_metrics_used"] = int(msel)
        metrics["k"] = int(km.n_clusters)

        uniq, cnts = np.unique(y, return_counts=True)
        metrics["cluster_counts"] = {int(k): int(v) for k, v in zip(uniq, cnts)}
        metrics["balance_ratio"] = float(cnts.min() / cnts.max()) if len(cnts) > 1 else 1.0

        metrics["inertia_sample"] = float(
            np.mean(np.min(((Xm[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1), axis=1))
        )

        cd = ((km.cluster_centers_[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
        cd = cd + np.eye(cd.shape[0]) * 1e9
        metrics["min_center_dist_pca"] = float(np.sqrt(cd.min()))

        try:
            metrics["silhouette"] = float(silhouette_score(Xm, ym, metric="euclidean"))
        except Exception as e:
            metrics["silhouette"] = f"err: {e}"
        try:
            metrics["calinski_harabasz"] = float(calinski_harabasz_score(Xm, ym))
        except Exception as e:
            metrics["calinski_harabasz"] = f"err: {e}"
        try:
            metrics["davies_bouldin"] = float(davies_bouldin_score(Xm, ym))
        except Exception as e:
            metrics["davies_bouldin"] = f"err: {e}"

        with open(os.path.join(save_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        self.log(f"[diagnose] metrics.json saved: {metrics}")

        # PCA-2D plot
        try:
            import matplotlib.pyplot as plt
            Xp2 = Xp[:, :2]
            plt.figure(figsize=(8, 8))
            plt.scatter(Xp2[:, 0], Xp2[:, 1], c=y, s=3, alpha=0.8, cmap="tab10")
            plt.title(f"PCA-2D (epoch) · k={km.n_clusters} · n={Xp2.shape[0]}")
            plt.tight_layout()
            out = os.path.join(save_dir, f"pca2d_epoch_{mode}.png")
            plt.savefig(out, dpi=300); plt.close()
            self.log(f"[plot] {out}")
        except Exception as e:
            self.log(f"[warn] PCA-2D plot failed: {e}")

        # UMAP
        if HAS_UMAP:
            try:
                self.visualize_umap_epoch(save_dir=save_dir, mode=mode, sample_n=min(sample_n, 200_000),
                                          use_training_space=True, pca_dim_local=64)
            except Exception as e:
                self.log(f"[warn] UMAP plot failed: {e}")

    # ---------------- summarize ----------------
    def summarize_epoch(self, save_dir: str):
        labels = np.load(os.path.join(save_dir, "labels.npy"))
        subs = np.load(os.path.join(save_dir, "subject.npy"), allow_pickle=True).tolist()
        vals, cnts = np.unique(labels, return_counts=True)
        self.log("== Cluster counts (epoch) ==")
        for v, c in zip(vals, cnts):
            self.log(f"  cluster {v}: {c}")

    def summarize_patch(self, save_dir: str):
        labels = np.load(os.path.join(save_dir, "patch_labels.npy"))
        vals, cnts = np.unique(labels, return_counts=True)
        self.log("== Cluster counts (patch) ==")
        for v, c in zip(vals, cnts):
            self.log(f"  cluster {v}: {c}")

    # ---------------- run ----------------
    def run(self,
            save_dir: str,
            level: str = "epoch",               # "epoch" | "patch"
            mode: str = "concat1536",           # epoch: "concat1536" | "mean768"
            k: int = 2,
            pca_dim: int = 128,
            batch_rows: int = 200_000,
            normalize: bool = True,
            # patch options
            patch_feat: str = "mean768",        # "mean768" | "concat1536"
            channel_agg: str = "mean",
            # viz/diag
            do_diagnose: bool = True,
            do_plot: bool = True):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir
        self._init_logger(save_dir)
        t0 = time.time()
        self.log(f"Start run: level={level}, mode={mode}, k={k}, pca_dim={pca_dim}, normalize={normalize}")
        self.log(f"features_dir={self.features_dir}, files={len(self.paths)}, total_epochs={self.count_epochs()}")

        if level == "epoch":
            if mode == "mean768":
                scaler, ipca, km, labels, order_index = self._pipeline_mean768(
                    k=k, pca_dim=min(64, pca_dim), batch_rows=batch_rows, normalize=normalize
                )
            elif mode == "concat1536":
                scaler, ipca, km, labels, order_index = self._pipeline_concat1536(
                    k=k, pca_dim=pca_dim, batch_rows=batch_rows, normalize=normalize
                )
            else:
                raise ValueError(f"Unknown epoch mode: {mode}")

            if scaler is not None:
                joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
            joblib.dump(ipca, os.path.join(save_dir, "pca.pkl"))
            joblib.dump(km,  os.path.join(save_dir, "kmeans.pkl"))
            self._save_epoch_labels(save_dir, labels, order_index)

            with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(dict(level=level, mode=mode, k=k, pca_dim=pca_dim,
                               batch_rows=batch_rows, normalize=normalize), f, indent=2)

            if do_diagnose:
                self.diagnose_epoch(save_dir, mode=mode, sample_n=100_000, metrics_sample_n=min(100_000, self.count_epochs()))
            if do_plot:
                try:
                    self.visualize_kmeans_pca2d_epoch(save_dir, mode=mode, sample_n=min(200_000, self.count_epochs()))
                except Exception as e:
                    self.log(f"[warn] PCA plot failed: {e}")
                if HAS_UMAP:
                    try:
                        self.visualize_umap_epoch(save_dir, mode=mode, sample_n=100_000, use_training_space=True)
                    except Exception as e:
                        self.log(f"[warn] UMAP plot failed: {e}")

            self.summarize_epoch(save_dir)

        elif level == "patch":
            scaler, ipca, km, labels, order_index = self._pipeline_patch_level(
                patch_feat=patch_feat,
                channel_agg=channel_agg,
                k=k,
                pca_dim=min(64, pca_dim),
                batch_rows=batch_rows,
                normalize=normalize
            )
            if scaler is not None:
                joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
            joblib.dump(ipca, os.path.join(save_dir, "pca.pkl"))
            joblib.dump(km,  os.path.join(save_dir, "kmeans.pkl"))

            self._save_patch_labels(save_dir, labels, order_index)
            with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
                json.dump(dict(level=level, patch_feat=patch_feat, channel_agg=channel_agg,
                               k=k, pca_dim=pca_dim, batch_rows=batch_rows, normalize=normalize), f, indent=2)

            self.summarize_patch(save_dir)
        else:
            raise ValueError(f"Unknown level: {level}")

        self.log(f"[Done] Total time: {time.time()-t0:.1f}s")


# ---------------- CLI ----------------
def build_argparser():
    ap = argparse.ArgumentParser(description="REM clustering (epoch-level & patch-level)")
    ap.add_argument("--features_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="*.h5")
    ap.add_argument("--save_dir", type=str, required=True)

    ap.add_argument("--level", type=str, default="epoch", choices=["epoch", "patch"])

    # epoch-level
    ap.add_argument("--mode", type=str, default="concat1536", choices=["concat1536", "mean768"])

    # patch-level options
    ap.add_argument("--patch_feat", type=str, default="mean768", choices=["mean768", "concat1536"])
    ap.add_argument("--channel_agg", type=str, default="mean", choices=["mean", "none"])

    # shared
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--batch_rows", type=int, default=200_000)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.set_defaults(normalize=True)

    ap.add_argument("--no-diagnose", dest="do_diagnose", action="store_false")
    ap.add_argument("--diagnose", dest="do_diagnose", action="store_true")
    ap.set_defaults(do_diagnose=True)

    ap.add_argument("--no-plot", dest="do_plot", action="store_false")
    ap.add_argument("--plot", dest="do_plot", action="store_true")
    ap.set_defaults(do_plot=True)

    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()

    am = AnalysisManager(features_dir=args.features_dir,
                         pattern=args.pattern,
                         order="patch_channel",
                         seed=42)

    am.run(save_dir=args.save_dir,
           level=args.level,
           mode=args.mode,
           k=args.k,
           pca_dim=args.pca_dim,
           batch_rows=args.batch_rows,
           normalize=args.normalize,
           patch_feat=args.patch_feat,
           channel_agg=args.channel_agg,
           do_diagnose=args.do_diagnose,
           do_plot=args.do_plot)