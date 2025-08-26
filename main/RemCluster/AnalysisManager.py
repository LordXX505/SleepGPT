# analysis_manager_mode.py  (CPU only + CLS mode + argparse)
import os, re, glob, h5py, json, time, logging, random, argparse
import numpy as np
from typing import List, Tuple, Dict, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.cluster import MiniBatchKMeans
import joblib
from tqdm import tqdm

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


class AnalysisManager:
    """
    CPU 流式 REM 聚类：支持三种原始形状
      - (61,1536): 第 0 行为 CLS(time+freq)，其余 60 行为 patch
      - (60,1536): 60 个 patch (time+freq)
      - (15,4,2,768): 15 patch × 4 channel × 2 domain × 768

    模式：
      - mean768      : epoch→768（time/freq 融合）→ IPCA → MiniBatchKMeans
      - concat1536   : epoch→1536（time||freq）→ IPCA → MiniBatchKMeans
      - epoch_cls    : 直接使用 CLS(1536) 向量 → IPCA → MiniBatchKMeans（要求 61×1536）
      - patch_vote   : patch(15×768) → KMeans → epoch 投票
      - multiview    : 仅当可还原 [15,4,2,768] 时；8 视图标准化+方差加权 → 768 → IPCA → KMeans

    所有预测阶段记录 order_index=[(sid,eid), ...]，labels 与其一一对齐保存。
    """

    def __init__(self, features_dir: str, pattern: str = "*.h5",
                 order: str = "patch_channel", seed: int = 42):
        self.features_dir = features_dir
        self.pattern = pattern
        self.order = order
        self.rng = random.Random(seed)
        self.seed = seed

        self.paths: List[str] = sorted(glob.glob(os.path.join(features_dir, pattern)))
        if not self.paths:
            raise FileNotFoundError(f"No files matched: {features_dir}/{pattern}")
        self._epoch_pat = re.compile(r"^epoch_(\d+)$")
        self._total_epochs = None
        self._rs_count = 0
        self.logger: Optional[logging.Logger] = None

    # ---------- logging ----------
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
        logger.addHandler(ch); logger.addHandler(fh)
        self.logger = logger

    def log(self, msg: str):
        print(msg)
        if self.logger: self.logger.info(msg)

    # ---------- iterate raw ----------
    def iter_epochs_raw(self):
        """yield (sid, eid, raw) in deterministic order."""
        for p in self.paths:
            sid = os.path.splitext(os.path.basename(p))[0]
            with h5py.File(p, "r") as f:
                ekeys = []
                for k in f.keys():
                    m = self._epoch_pat.match(k)
                    if m: ekeys.append((int(m.group(1)), k))
                ekeys.sort(key=lambda t: t[0])
                for eid, kk in ekeys:
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

    # ---------- shape helpers ----------
    def get_cls_and_patches(self, raw: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        """
        返回 (cls1536 or None, patches60x1536)
        - (61,1536): (raw[0], raw[1:])
        - (60,1536): (None, raw)
        - (15,4,2,768): (None, to60x1536)
        """
        if raw.shape == (61,1536):
            return raw[0], raw[1:]
        if raw.shape == (60,1536):
            return None, raw
        if raw.shape == (15,4,2,768):
            # [15,4,2,768] → [15,4,1536] → [60,1536]
            v = np.concatenate([raw[...,0,:], raw[...,1,:]], axis=-1)
            return None, v.reshape(15*4, 1536)
        raise ValueError(f"Unknown epoch shape: {raw.shape}")

    def to_1542_or_none(self, raw: np.ndarray) -> Optional[np.ndarray]:
        """若可转换则返回 [15,4,2,768]，否则 None。"""
        if raw.shape == (15,4,2,768):
            return raw
        if raw.shape in [(60,1536),(61,1536)]:
            cls, patches = self.get_cls_and_patches(raw)
            # 只能在已知顺序下把 60 还原到 15×4；这里假定 order=patch_channel
            if patches.shape == (60,1536) and self.order == "patch_channel":
                v = patches.reshape(15, 4, 1536)
                v = np.stack([v[...,:768], v[...,768:]], axis=2)  # [15,4,2,768]
                return v
        return None

    # ---------- feature builders ----------
    def vec_concat1536(self, raw: np.ndarray) -> np.ndarray:
        """epoch → 1536：如果有 60×1536 就对 60 均值；如果 1542 就对 patch×channel 均值后 concat。"""
        if raw.shape in [(60,1536),(61,1536)]:
            _, patches = self.get_cls_and_patches(raw)
            return patches.mean(axis=0)
        if raw.shape == (15,4,2,768):
            dom = raw.mean(axis=(0,1))  # [2,768]
            return dom.reshape(-1)
        raise ValueError(raw.shape)

    def vec_mean768(self, raw: np.ndarray) -> np.ndarray:
        """epoch → 768：把 1536 拆成两半求均，或直接在 1542 上对 (patch,channel,domain) 均值。"""
        if raw.shape in [(60,1536),(61,1536)]:
            _, patches = self.get_cls_and_patches(raw)    # [60,1536]
            m1536 = patches.mean(axis=0)                  # [1536]
            return 0.5*(m1536[:768] + m1536[768:])        # [768]
        if raw.shape == (15,4,2,768):
            return raw.mean(axis=(0,1,2))                 # [768]
        raise ValueError(raw.shape)

    def patches_15x768(self, raw: np.ndarray) -> np.ndarray:
        """patch_vote 用：返回 [15,768]"""
        v1542 = self.to_1542_or_none(raw)
        if v1542 is None:
            raise ValueError("patch_vote 需要能还原到 [15,4,2,768]")
        return v1542.mean(axis=(1,2))  # [15,768]

    def multiview_8x768(self, raw: np.ndarray) -> Optional[np.ndarray]:
        v1542 = self.to_1542_or_none(raw)
        if v1542 is None:
            return None
        return v1542.mean(axis=0).reshape(8,768)

    # ---------- reservoir for viz ----------
    def _reservoir_append(self, feats: list, idxs: list, vec, idx: int, cap: int):
        self._rs_count += 1
        if len(feats) < cap:
            feats.append(vec); idxs.append(idx)
        else:
            j = self.rng.randrange(0, self._rs_count)
            if j < cap:
                feats[j] = vec; idxs[j] = idx

    # ---------- save labels+index ----------
    def _save_labels_and_index(self, save_dir: str, labels: np.ndarray, order_index: List[Tuple[str,int]]):
        if len(labels) != len(order_index):
            raise ValueError("labels length != order_index length")
        np.save(os.path.join(save_dir, "labels.npy"), labels)
        with open(os.path.join(save_dir, "index.csv"), "w", encoding="utf-8") as fw:
            fw.write("subject,epoch\n")
            for sid,eid in order_index:
                fw.write(f"{sid},{eid}\n")
        np.save(os.path.join(save_dir, "subject.npy"), np.array([s for s,_ in order_index], dtype=object))
        np.save(os.path.join(save_dir, "epoch.npy"),   np.array([e for _,e in order_index], dtype=np.int32))

    # ---------- pipelines (CPU streaming) ----------
    def _pipeline_generic_epoch(self, vec_fn, out_dim, k=2, pca_dim=128, batch_rows=200_000, normalize=True):
        """通用：epoch→向量（由 vec_fn 生成）→ scaler → IPCA → MiniBatchKMeans"""
        total = self.count_epochs()
        scaler = StandardScaler() if normalize else None

        # Pass1: Scaler
        if scaler is not None:
            buf=[]
            self.log(f"\n[Pass1] Fitting StandardScaler (dim={out_dim})")
            for _sid,_eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="Scaler", ncols=100):
                buf.append(vec_fn(raw))
                if len(buf) >= batch_rows:
                    scaler.partial_fit(np.stack(buf,0)); buf=[]
            if buf: scaler.partial_fit(np.stack(buf,0)); buf=[]

        # Pass2: IPCA
        ipca = IncrementalPCA(n_components=min(pca_dim, out_dim))
        buf=[]
        self.log("[Pass2] Fitting IncrementalPCA")
        for _sid,_eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="PCA", ncols=100):
            buf.append(vec_fn(raw))
            if len(buf) >= batch_rows:
                X=np.stack(buf,0);  X=scaler.transform(X) if scaler is not None else X
                ipca.partial_fit(X); buf=[]
        if buf:
            X=np.stack(buf,0);  X=scaler.transform(X) if scaler is not None else X
            ipca.partial_fit(X); buf=[]

        # Pass3: KMeans
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=self.seed)
        buf=[]
        self.log("[Pass3] Fitting MiniBatchKMeans")
        for _sid,_eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="KMeans", ncols=100):
            buf.append(vec_fn(raw))
            if len(buf) >= batch_rows:
                X=np.stack(buf,0); X=scaler.transform(X) if scaler is not None else X
                Xp=ipca.transform(X); km.partial_fit(Xp); buf=[]
        if buf:
            X=np.stack(buf,0); X=scaler.transform(X) if scaler is not None else X
            Xp=ipca.transform(X); km.partial_fit(Xp)

        # Pass4: Predict + order_index
        labels=np.empty(total, np.int8)
        order_index=[]; buf=[]; i=0
        self.log("[Pass4] Predicting & recording order_index")
        for sid,eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="Predict", ncols=100):
            buf.append(vec_fn(raw)); order_index.append((sid,eid))
            if len(buf) >= batch_rows:
                X=np.stack(buf,0); X=scaler.transform(X) if scaler is not None else X
                Xp=ipca.transform(X); labels[i:i+len(buf)]=km.predict(Xp); i+=len(buf); buf=[]
        if buf:
            X=np.stack(buf,0); X=scaler.transform(X) if scaler is not None else X
            Xp=ipca.transform(X); labels[i:i+len(buf)]=km.predict(Xp)

        return scaler, ipca, km, labels, order_index

    def _pipeline_mean768(self, **kw):
        return self._pipeline_generic_epoch(self.vec_mean768, out_dim=768, **kw)

    def _pipeline_concat1536(self, **kw):
        return self._pipeline_generic_epoch(self.vec_concat1536, out_dim=1536, **kw)

    def _pipeline_epoch_cls(self, **kw):
        def vec_cls(raw: np.ndarray) -> np.ndarray:
            cls, patches = self.get_cls_and_patches(raw)
            if cls is None:
                raise ValueError("epoch_cls 需要 (61,1536) 格式且包含 CLS。")
            return cls
        return self._pipeline_generic_epoch(vec_cls, out_dim=1536, **kw)

    def _pipeline_patch_vote(self, k=2, pca_dim=64, batch_patches=300_000,
                             vote_threshold=0.5, normalize=True):
        total = self.count_epochs()
        scaler = StandardScaler() if normalize else None
        ipca = IncrementalPCA(n_components=pca_dim)
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=self.seed)

        # 1) Scaler on patches
        if scaler is not None:
            cur, rows = [], 0
            self.log("\n[Pass1] Fitting StandardScaler (patch 15×768)")
            for _sid,_eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="Scaler(P)", ncols=100):
                P = self.patches_15x768(raw)   # [15,768]
                cur.append(P); rows += P.shape[0]
                if rows >= batch_patches:
                    X = np.vstack(cur); scaler.partial_fit(X); cur=[]; rows=0
            if rows>0:
                X=np.vstack(cur); scaler.partial_fit(X)

        # 2) IPCA on patches
        self.log("[Pass2] Fitting IncrementalPCA (patch)")
        for _sid,_eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="PCA(P)", ncols=100):
            X = self.patches_15x768(raw)
            X = scaler.transform(X) if scaler is not None else X
            ipca.partial_fit(X)

        # 3) KMeans on patches
        self.log("[Pass3] Fitting MiniBatchKMeans (patch)")
        for _sid,_eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="KMeans(P)", ncols=100):
            X = self.patches_15x768(raw)
            X = scaler.transform(X) if scaler is not None else X
            Xp = ipca.transform(X)
            km.partial_fit(Xp)

        # 4) Predict+Vote
        labels=np.empty(total, np.int8)
        order_index=[]; i=0
        self.log("[Pass4] Predict+Vote (epoch)")
        for sid,eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="Predict(P)", ncols=100):
            X = self.patches_15x768(raw)
            X = scaler.transform(X) if scaler is not None else X
            Xp = ipca.transform(X)
            pl = km.predict(Xp)
            labels[i] = 1 if (pl.mean() >= vote_threshold) else 0
            order_index.append((sid,eid)); i+=1

        return scaler, ipca, km, labels, order_index

    def _pipeline_multiview(self, k=2, pca_dim=64, batch_rows=100_000, normalize=True):
        total = self.count_epochs()
        scalers = [StandardScaler() if normalize else None for _ in range(8)]
        cache = [[] for _ in range(8)]

        # Pass1: fit 8 scalers
        self.log("\n[Pass1] Fitting 8-view StandardScaler (multiview)")
        valid_total = 0
        for _sid,_eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="Scaler(8v)", ncols=100):
            V = self.multiview_8x768(raw)
            if V is None: continue
            valid_total += 1
            for i in range(8): cache[i].append(V[i])
            if len(cache[0]) >= batch_rows:
                for i in range(8):
                    if scalers[i] is not None:
                        scalers[i].partial_fit(np.stack(cache[i],0))
                cache = [[] for _ in range(8)]
        if len(cache[0]):
            for i in range(8):
                if scalers[i] is not None:
                    scalers[i].partial_fit(np.stack(cache[i],0))
            cache = [[] for _ in range(8)]

        # weights by variance
        self.log("[Pass1b] Estimating variance weights")
        var_sum = np.zeros(8)
        cache = [[] for _ in range(8)]
        for _sid,_eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="Var(8v)", ncols=100):
            V = self.multiview_8x768(raw)
            if V is None: continue
            for i in range(8): cache[i].append(V[i])
            if len(cache[0]) >= batch_rows:
                for i in range(8):
                    arr = np.stack(cache[i],0)
                    if scalers[i] is not None: arr = scalers[i].transform(arr)
                    var_sum[i] += arr.var()
                cache = [[] for _ in range(8)]
        if len(cache[0]):
            for i in range(8):
                arr = np.stack(cache[i],0)
                if scalers[i] is not None: arr = scalers[i].transform(arr)
                var_sum[i] += arr.var()
        w = var_sum / (var_sum.sum() + 1e-8)

        # Pass2: IPCA + KMeans on weighted 768
        ipca = IncrementalPCA(n_components=min(pca_dim,768))
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=self.seed)

        self.log("[Pass2a] Fitting IncrementalPCA")
        buf=[]
        for _sid,_eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="PCA(8v)", ncols=100):
            V = self.multiview_8x768(raw)
            if V is None: continue
            if any(scalers):
                for i in range(8):
                    if scalers[i] is not None:
                        V[i] = scalers[i].transform(V[i][None])[0]
            emb = (V * w[:,None]).sum(0)   # [768]
            buf.append(emb)
            if len(buf) >= batch_rows:
                ipca.partial_fit(np.stack(buf,0)); buf=[]
        if buf: ipca.partial_fit(np.stack(buf,0)); buf=[]

        self.log("[Pass2b] Fitting MiniBatchKMeans")
        buf=[]
        for _sid,_eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="KMeans(8v)", ncols=100):
            V = self.multiview_8x768(raw)
            if V is None: continue
            if any(scalers):
                for i in range(8):
                    if scalers[i] is not None:
                        V[i] = scalers[i].transform(V[i][None])[0]
            emb = (V * w[:,None]).sum(0)
            buf.append(emb)
            if len(buf) >= batch_rows:
                xp = ipca.transform(np.stack(buf,0)); km.partial_fit(xp); buf=[]
        if buf:
            xp = ipca.transform(np.stack(buf,0)); km.partial_fit(xp)

        # Predict + order_index（只对可 multiview 的样本）
        labels=[]; order_index=[]
        self.log("[Pass3] Predicting (multiview-only) & recording order_index")
        for sid,eid,raw in tqdm(self.iter_epochs_raw(), total=total, desc="Predict(8v)", ncols=100):
            V = self.multiview_8x768(raw)
            if V is None: continue
            if any(scalers):
                for i in range(8):
                    if scalers[i] is not None:
                        V[i] = scalers[i].transform(V[i][None])[0]
            emb = (V * w[:,None]).sum(0)
            xp = ipca.transform(emb[None])
            labels.append(int(km.predict(xp)[0]))
            order_index.append((sid,eid))

        return (scalers, w), ipca, km, np.array(labels, dtype=np.int8), order_index

    # ---------- summarize ----------
    def summarize(self, save_dir: str, subject_to_group: Optional[Dict[str,str]]=None):
        labels = np.load(os.path.join(save_dir, "labels.npy"))
        subs = np.load(os.path.join(save_dir, "subject.npy"), allow_pickle=True).tolist()
        vals, cnts = np.unique(labels, return_counts=True)
        self.log("== Cluster counts (overall) ==")
        for v,c in zip(vals,cnts):
            self.log(f"  cluster {v}: {c}")
        if subject_to_group:
            stat: Dict[str, Dict[int,int]] = {}
            for lab, sid in zip(labels, subs):
                grp = subject_to_group.get(sid, "Unknown")
                stat.setdefault(grp, {}).setdefault(int(lab), 0)
                stat[grp][int(lab)] += 1
            self.log("== Cluster counts by group ==")
            for grp, d in stat.items():
                total = sum(d.values())
                frac = {k: f"{v} ({v/total:.1%})" for k,v in d.items()}
                self.log(f"  {grp}: {frac}")

    def diagnose(self,
                 save_dir: str,
                 mode: str,
                 sample_n: int = 100_000,
                 pca_dim_for_umap: int = 64,
                 metrics_sample_n: int = 100_000):
        """
        聚类诊断（不依赖标签的外部真值）：
          - 读取已保存的 scaler/pca/kmeans/labels
          - 以“训练用 PCA 空间”为准做抽样
          - 计算：簇规模、惯性(inertia)、中心间最小距离、Silhouette/CH/DB（在抽样上）
          - 生成：PCA-2D 散点（着色=labels）、KMeans 2D 决策区域（近似）、UMAP(与训练空间一致)
          - 输出：metrics.json + pca2d.png + kmeans_regions.png + umap2d.png（若安装了umap）
        """
        import json, os
        import numpy as np
        import joblib
        from tqdm import tqdm
        from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

        # 必要文件
        pca_path = os.path.join(save_dir, "pca.pkl")
        km_path = os.path.join(save_dir, "kmeans.pkl")
        lab_path = os.path.join(save_dir, "labels.npy")
        if not (os.path.exists(pca_path) and os.path.exists(km_path) and os.path.exists(lab_path)):
            self.log("[diagnose] Missing pca/kmeans/labels; skip.");
            return

        pca = joblib.load(pca_path)
        km = joblib.load(km_path)
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None
        labels = np.load(lab_path)

        # —— 抽同一批样本，并投影到训练用 PCA 空间（与聚类一致） ——
        feats, idxs = [], []
        self._rs_count = 0
        total = self.count_epochs()

        def _vec(raw):
            if mode == "mean768":
                return self.vec_mean768(raw)
            elif mode == "concat1536":
                return self.vec_concat1536(raw)
            elif mode == "epoch_cls":
                cls, _ = self.get_cls_and_patches(raw)
                return cls if cls is not None else self.vec_concat1536(raw)
            elif mode in ("patch_vote", "multiview"):
                return self.vec_mean768(raw)
            else:
                raise ValueError(mode)

        for i, (_sid, _eid, raw) in enumerate(tqdm(self.iter_epochs_raw(), total=total,
                                                   desc="Diag: sampling", ncols=100)):
            v = _vec(raw)
            self._reservoir_append(feats, idxs, v, i, cap=max(sample_n, metrics_sample_n))

        X = np.stack(feats, 0).astype(np.float32, copy=False)
        idxs = np.array(idxs, dtype=np.int64)
        if scaler is not None:
            X = scaler.transform(X)
        Xp = pca.transform(X)  # 与训练/聚类完全一致的空间
        y = labels[idxs] if len(labels) > idxs.max() else km.predict(Xp)

        # —— 指标（在 metrics_sample_n 上算，避免内存爆炸） ——
        msel = min(metrics_sample_n, Xp.shape[0])
        Xm, ym = Xp[:msel], y[:msel]

        metrics = {}
        metrics["n_samples_total"] = int(total)
        metrics["n_samples_sampled"] = int(Xp.shape[0])
        metrics["n_metrics_used"] = int(msel)
        metrics["k"] = int(km.n_clusters)
        # 簇规模
        uniq, cnts = np.unique(y, return_counts=True)
        metrics["cluster_counts"] = {int(k): int(v) for k, v in zip(uniq, cnts)}
        metrics["balance_ratio"] = float(cnts.min() / cnts.max()) if len(cnts) > 1 else 1.0
        # 惯性（训练时的km.inertia_是全量/拟合统计，这里再在抽样上估算一次）
        metrics["inertia_sample"] = float(
            np.mean(np.min(((Xm[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1), axis=1)))
        # 中心间最小距离（PCA空间）
        cd = ((km.cluster_centers_[:, None, :] - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
        cd = cd + np.eye(cd.shape[0]) * 1e9
        metrics["min_center_dist_pca"] = float(np.sqrt(cd.min()))
        # 质量指标
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

        # —— 可视化：PCA-2D散点 + 决策边界 + UMAP(一致空间) ——
        # 1) PCA-2D 散点
        try:
            import matplotlib.pyplot as plt
            Xp2 = Xp[:, :2]
            plt.figure(figsize=(8, 8))
            plt.scatter(Xp2[:, 0], Xp2[:, 1], c=y, s=3, alpha=0.8, cmap="tab10")
            plt.title(f"PCA-2D (train space) · k={km.n_clusters} · n={Xp2.shape[0]}")
            plt.tight_layout()
            out = os.path.join(save_dir, f"pca2d_{mode}.png")
            plt.savefig(out, dpi=300);
            plt.close()
            self.log(f"[plot] {out}")
        except Exception as e:
            self.log(f"[warn] PCA-2D plot failed: {e}")

        # 2) 决策区域（2D近似）
        try:
            import matplotlib.pyplot as plt
            centers2 = km.cluster_centers_[:, :2]
            x_min, x_max = Xp2[:, 0].min() - 0.5, Xp2[:, 0].max() + 0.5
            y_min, y_max = Xp2[:, 1].min() - 0.5, Xp2[:, 1].max() + 0.5
            grid = 500
            xx, yy = np.meshgrid(np.linspace(x_min, x_max, grid), np.linspace(y_min, y_max, grid))
            grid2 = np.c_[xx.ravel(), yy.ravel()]
            d = ((grid2[:, None, :] - centers2[None, :, :]) ** 2).sum(-1)
            Z = d.argmin(axis=1).reshape(xx.shape)
            plt.figure(figsize=(8, 8))
            plt.contourf(xx, yy, Z, levels=km.n_clusters, alpha=0.25, cmap="tab10")
            plt.scatter(Xp2[:, 0], Xp2[:, 1], c=y, s=3, alpha=0.85, cmap="tab10", edgecolors="none")
            plt.scatter(centers2[:, 0], centers2[:, 1], c=range(km.n_clusters), s=120, marker="X", edgecolors="k",
                        linewidths=1.0, cmap="tab10")
            plt.title("KMeans regions on PCA-2D (approx.)")
            plt.tight_layout()
            out = os.path.join(save_dir, f"kmeans_regions_pca2d_{mode}.png")
            plt.savefig(out, dpi=300);
            plt.close()
            self.log(f"[plot] {out}")
        except Exception as e:
            self.log(f"[warn] regions plot failed: {e}")

        # 3) UMAP（与训练空间一致）
        try:
            if HAS_UMAP:
                self.visualize_umap(save_dir=save_dir, mode=mode,
                                    sample_n=min(sample_n, 200_000),
                                    pca_dim=pca_dim_for_umap,
                                    use_training_space=True)
            else:
                self.log("[diagnose] umap-learn not installed; skip UMAP.")
        except Exception as e:
            self.log(f"[warn] UMAP plot failed: {e}")

    # ---------- PCA-space 可视化 ----------
    def visualize_kmeans_pca2d(self, save_dir: str, mode: str, sample_n: int = 200_000):
        import matplotlib.pyplot as plt
        labels = np.load(os.path.join(save_dir, "labels.npy"))
        pca = joblib.load(os.path.join(save_dir, "pca.pkl"))
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        feats, idxs = [], []
        self._rs_count = 0
        self.log("\n[Plot] PCA 2D (same space as KMeans)")
        total = self.count_epochs()
        for i,(sid,eid,raw) in enumerate(tqdm(self.iter_epochs_raw(), total=total, desc="Sampling", ncols=100)):
            if   mode=="mean768":      v = self.vec_mean768(raw)
            elif mode=="concat1536":   v = self.vec_concat1536(raw)
            elif mode=="epoch_cls":    v = self.get_cls_and_patches(raw)[0] or self.vec_concat1536(raw)
            elif mode=="patch_vote":   v = self.vec_mean768(raw)
            elif mode=="multiview":    v = self.vec_mean768(raw)
            else: raise ValueError(mode)
            self._reservoir_append(feats, idxs, v, i, cap=sample_n)
        X = np.stack(feats,0)
        if scaler is not None: X = scaler.transform(X)
        Xp = pca.transform(X)[:, :2]
        y = labels[np.array(idxs, dtype=int)]
        plt.figure(figsize=(8,8))
        plt.scatter(Xp[:,0], Xp[:,1], c=y, s=3, alpha=0.7, cmap="tab10")
        plt.title(f"KMeans@PCA · {mode} · n={len(y)}")
        plt.tight_layout()
        out = os.path.join(save_dir, f"kmeans_pca2d_{mode}.png")
        plt.savefig(out, dpi=300); plt.close()
        self.log(f"[plot] Saved: {out}")

    # ---------- UMAP 仅展示 ----------
    def visualize_umap(self,
                       save_dir: str,
                       mode: str = "concat1536",
                       sample_n: int = 100_000,
                       pca_dim: int = 64,
                       use_training_space: bool = True,
                       n_neighbors: int = 25,
                       min_dist: float = 0.1):
        """
        UMAP 可视化：
          - use_training_space=True：加载训练阶段保存的 scaler.pkl / pca.pkl，
            在相同特征空间上做 UMAP（推荐，和 KMeans 决策空间一致）。
          - False：仅为可视化单独拟合 Scaler/PCA（更自由但与训练空间可能不同）。

        输出：
          - {save_dir}/umap2d_{mode}.png
          - {save_dir}/umap_coords_{mode}.npy        # [M, 2]，UMAP坐标（M=抽样个数）
          - {save_dir}/umap_indices_{mode}.npy       # [M]，这些点在全体遍历顺序中的索引
          - 若存在 labels.npy，会用作颜色上色
        """
        if not HAS_UMAP:
            self.log("[warn] umap-learn not installed; skip UMAP.")
            return

        import matplotlib.pyplot as plt

        # 1) 采样同一批点（与其它可视化保持一致）
        feats, idxs = [], []
        self._rs_count = 0
        total = self.count_epochs()
        self.log(f"\n[Plot] UMAP 2D (mode={mode}, sample_n={sample_n}, "
                 f"use_training_space={use_training_space})")

        # 按模式取特征向量
        def _vec(raw):
            if mode == "mean768":
                return self.vec_mean768(raw)
            elif mode == "concat1536":
                return self.vec_concat1536(raw)
            elif mode == "epoch_cls":
                cls, _patches = self.get_cls_and_patches(raw)
                if cls is None:
                    # 兼容无 CLS 的样本：退化到 concat1536
                    return self.vec_concat1536(raw)
                return cls
            elif mode == "patch_vote":
                # 只是可视化，取 epoch 级 mean768 表示
                return self.vec_mean768(raw)
            elif mode == "multiview":
                # 可视化层面用 mean768 近似
                return self.vec_mean768(raw)
            else:
                raise ValueError(mode)

        for i, (_sid, _eid, raw) in enumerate(tqdm(self.iter_epochs_raw(), total=total,
                                                   desc="Sampling(UMAP)", ncols=100)):
            v = _vec(raw)
            self._reservoir_append(feats, idxs, v, i, cap=sample_n)

        X = np.stack(feats, 0).astype(np.float32, copy=False)
        sample_indices = np.array(idxs, dtype=np.int64)

        # 2) 构造 UMAP 输入特征：两条路径
        # 2a) 与训练空间一致（优先）
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        pca_path = os.path.join(save_dir, "pca.pkl")

        X_for_umap = X
        used_space = "raw"

        if use_training_space and os.path.exists(pca_path):
            # 尝试还原训练时的空间（scaler → pca）
            try:
                pca = joblib.load(pca_path)
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    X_for_umap = scaler.transform(X_for_umap)
                    used_space = "scaler+pca"
                else:
                    used_space = "pca-only"

                X_for_umap = pca.transform(X_for_umap)  # 到训练用的 PCA 空间
                # 注：这里不会截取前2维；UMAP会在完整 PCA 空间上学习邻域
            except Exception as e:
                self.log(f"[warn] load training scaler/pca failed: {e} ; "
                         f"fallback to local fit.")
                use_training_space = False  # 退化到 2b

        # 2b) 本地拟合（仅用于展示，与训练不完全一致）
        if not use_training_space:
            from sklearn.preprocessing import StandardScaler
            from sklearn.decomposition import PCA
            scaler_local = StandardScaler().fit(X_for_umap)
            Xs = scaler_local.transform(X_for_umap)
            if pca_dim and pca_dim < Xs.shape[1]:
                pca_local = PCA(n_components=pca_dim, random_state=self.seed).fit(Xs)
                X_for_umap = pca_local.transform(Xs)
                used_space = f"local(scaler+pca{pca_dim})"
            else:
                X_for_umap = Xs
                used_space = "local(scaler-only)"

        # 3) 跑 UMAP（在上述特征空间上）
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=self.seed)
        emb = reducer.fit_transform(X_for_umap)  # [M,2]

        # 4) 上色（若有 labels，则按 KMeans 标签上色；否则单色）
        labels_path = os.path.join(save_dir, "labels.npy")
        labels = None
        if os.path.exists(labels_path):
            try:
                labels = np.load(labels_path)
            except Exception as e:
                self.log(f"[warn] load labels.npy failed: {e}")

        plt.figure(figsize=(8, 8))
        if labels is not None and len(labels) > sample_indices.max():
            c = labels[sample_indices]
            plt.scatter(emb[:, 0], emb[:, 1], c=c, s=3, alpha=0.7, cmap="tab10")
        else:
            plt.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.7)

        plt.title(f"UMAP 2D · {mode} · n={emb.shape[0]} · space={used_space}")
        plt.tight_layout()

        out_png = os.path.join(save_dir, f"umap2d_{mode}.png")
        out_npy = os.path.join(save_dir, f"umap_coords_{mode}.npy")
        out_idx = os.path.join(save_dir, f"umap_indices_{mode}.npy")

        plt.savefig(out_png, dpi=300)
        plt.close()

        np.save(out_npy, emb)
        np.save(out_idx, sample_indices)

        self.log(f"[plot] Saved: {out_png}")
        self.log(f"[dump] Saved coords: {out_npy}")
        self.log(f"[dump] Saved indices: {out_idx}")

    # ---------- run ----------
    def run(self, save_dir: str, mode: str, k: int = 2, pca_dim: int = 128,
            batch_rows: int = 200_000, vote_threshold: float = 0.5,
            umap_sample: int = 0, normalize: bool = True, auto_plot: bool = True):
        self._init_logger(save_dir)
        t0 = time.time()
        self.log(f"Start run: mode={mode}, k={k}, pca_dim={pca_dim}, normalize={normalize}, seed={self.seed}")
        self.log(f"features_dir={self.features_dir}, files={len(self.paths)}, total_epochs={self.count_epochs()}")

        if mode == "mean768":
            scaler, ipca, km, labels, order_index = self._pipeline_mean768(
                k=k, pca_dim=min(64, pca_dim), batch_rows=batch_rows, normalize=normalize)
        elif mode == "concat1536":
            scaler, ipca, km, labels, order_index = self._pipeline_concat1536(
                k=k, pca_dim=pca_dim, batch_rows=batch_rows, normalize=normalize)
        elif mode == "epoch_cls":
            scaler, ipca, km, labels, order_index = self._pipeline_epoch_cls(
                k=k, pca_dim=pca_dim, batch_rows=batch_rows, normalize=normalize)
        elif mode == "patch_vote":
            scaler, ipca, km, labels, order_index = self._pipeline_patch_vote(
                k=k, pca_dim=min(64, pca_dim), batch_patches=15_000*20,
                vote_threshold=vote_threshold, normalize=normalize)
        elif mode == "multiview":
            (scalers, w), ipca, km, labels, order_index = self._pipeline_multiview(
                k=k, pca_dim=min(64, pca_dim), batch_rows=min(100_000, batch_rows), normalize=normalize)
            joblib.dump(scalers, os.path.join(save_dir, "multiview_scalers.pkl"))
            np.save(os.path.join(save_dir, "multiview_weights.npy"), w)
            scaler = None
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # save models
        if scaler is not None:
            joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
        joblib.dump(ipca, os.path.join(save_dir, "pca.pkl"))
        joblib.dump(km,  os.path.join(save_dir, "kmeans.pkl"))

        # save labels+index
        self._save_labels_and_index(save_dir, labels, order_index)

        # save config
        cfg = dict(mode=mode, k=k, pca_dim=pca_dim, batch_rows=batch_rows,
                   vote_threshold=vote_threshold, features_dir=self.features_dir,
                   pattern=self.pattern, order=self.order, normalize=normalize, seed=self.seed)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # 统一诊断（指标 + 可视化）
        try:
            self.diagnose(save_dir=save_dir,
                          mode=mode,
                          sample_n=min(umap_sample if umap_sample > 0 else 100_000, 200_000),
                          pca_dim_for_umap=pca_dim,
                          metrics_sample_n=min(100_000, self.count_epochs()))
        except Exception as e:
            self.log(f"[warn] diagnose failed: {e}")


# ---------------- CLI ----------------
def build_argparser():
    ap = argparse.ArgumentParser(description="REM clustering (CPU streaming).")
    ap.add_argument("--features_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="*.h5")
    ap.add_argument("--order", type=str, default="patch_channel", choices=["patch_channel","channel_patch"])
    ap.add_argument("--save_dir", type=str, required=True)

    ap.add_argument("--mode", type=str, default="concat1536",
                    choices=["mean768","concat1536","epoch_cls","patch_vote","multiview"])
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--batch_rows", type=int, default=200_000)
    ap.add_argument("--vote_threshold", type=float, default=0.5)
    ap.add_argument("--normalize", action="store_true")
    ap.add_argument("--no-normalize", dest="normalize", action="store_false")
    ap.set_defaults(normalize=True)

    ap.add_argument("--auto_plot", action="store_true")
    ap.add_argument("--umap_sample", type=int, default=0)
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    am = AnalysisManager(features_dir=args.features_dir, pattern=args.pattern, order=args.order)
    am.run(save_dir=args.save_dir,
           mode=args.mode,
           k=args.k,
           pca_dim=args.pca_dim,
           batch_rows=args.batch_rows,
           vote_threshold=args.vote_threshold,
           umap_sample=args.umap_sample,
           normalize=args.normalize,
           auto_plot=args.auto_plot)