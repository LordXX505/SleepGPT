# analysis_manager_mode.py  (order_index + tqdm + logging)
import os, re, glob, h5py, json, time, logging, random
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
    REM 聚类分析（流式 + tqdm + logging + 显式顺序记录）：

    - 输入：每 subject 一个 .h5；每 epoch dataset 形如 (60,1536) 或 [15,4,2,768]
    - mode:
        - 'mean768'     : (channel,domain)均值→patch均值→[768]→IPCA→MiniBatchKMeans
        - 'concat1536'  : 保留time/freq concat→[1536]→IPCA→MiniBatchKMeans
        - 'multiview'   : 8视图(4通道×2域)标准化+方差加权→[768]→IPCA→MiniBatchKMeans
        - 'patch_vote'  : patch级聚类(15×[768])→epoch投票
    - 所有 pipeline 的预测阶段都会同步记录 order_index=[(sid,eid), ...]
      保存时用 order_index 与 labels 逐项对齐，彻底避免顺序假设。
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

        # 蓄水池采样计数器（给可视化抽样用）
        self._rs_count = 0

        # logger 占位
        self.logger: Optional[logging.Logger] = None

    # ---------- 日志 ----------
    def _init_logger(self, save_dir: str):
        os.makedirs(save_dir, exist_ok=True)
        logger = logging.getLogger("AnalysisManager")
        logger.setLevel(logging.INFO)
        logger.handlers.clear()

        # 控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s", "%H:%M:%S"))

        # 文件
        fh = logging.FileHandler(os.path.join(save_dir, "run.log"), encoding="utf-8")
        fh.setLevel(logging.INFO)
        fh.setFormatter(logging.Formatter(
            "[%(asctime)s] %(levelname)s: %(message)s", "%Y-%m-%d %H:%M:%S"
        ))

        logger.addHandler(ch)
        logger.addHandler(fh)
        self.logger = logger

    def log(self, msg: str):
        print(msg)  # 兜底
        if self.logger:
            self.logger.info(msg)

    # ---------- 基础 I/O ----------
    def _decode_epoch(self, arr: np.ndarray) -> np.ndarray:
        """ (60,1536) or [15,4,2,768] -> [15,4,2,768] """
        if arr.shape == (60, 1536):
            if self.order == "patch_channel":
                x = arr.reshape(15, 4, 1536)
            elif self.order == "channel_patch":
                x = arr.reshape(4, 15, 1536).transpose(1, 0, 2)
            else:
                raise ValueError("order must be 'patch_channel' or 'channel_patch'")
            return x.reshape(15, 4, 2, 768)
        if arr.shape == (15, 4, 2, 768):
            return arr
        raise ValueError(f"Unknown epoch shape: {arr.shape}")

    def iter_epochs(self):
        """ yield (subject_id, epoch_id, epoch_array[15,4,2,768]) in a deterministic order """
        for p in self.paths:
            sid = os.path.splitext(os.path.basename(p))[0]
            with h5py.File(p, "r") as f:
                # 只取匹配 epoch_ 的 key，并按数字排序，保证确定性
                ekeys = []
                for k in f.keys():
                    m = self._epoch_pat.match(k)
                    if m: ekeys.append((int(m.group(1)), k))
                ekeys.sort(key=lambda t: t[0])
                for eid, kk in ekeys:
                    arr = f[kk][()]
                    yield sid, eid, self._decode_epoch(arr)

    def count_epochs(self) -> int:
        if self._total_epochs is not None:
            return self._total_epochs
        total = 0
        for p in self.paths:
            with h5py.File(p, "r") as f:
                total += sum(1 for k in f.keys() if self._epoch_pat.match(k))
        self._total_epochs = total
        return total

    # ---------- 特征构造 ----------
    @staticmethod
    def epoch_mean768(x1542: np.ndarray) -> np.ndarray:
        return x1542.mean(axis=(1, 2)).mean(axis=0)  # [768]

    @staticmethod
    def epoch_concat1536(x1542: np.ndarray) -> np.ndarray:
        dom = x1542.mean(axis=(0, 1))               # [2,768]
        return dom.reshape(-1)                       # [1536]

    @staticmethod
    def epoch_multiview8(x1542: np.ndarray) -> np.ndarray:
        v = x1542.mean(axis=0)                       # [4,2,768]
        return v.reshape(8, 768)

    @staticmethod
    def patch_mean768(x1542: np.ndarray) -> np.ndarray:
        return x1542.mean(axis=(1, 2))               # [15,768]

    # ---------- 蓄水池采样（可视化） ----------
    def _reservoir_append(self, feats: list, idxs: list, vec, idx: int, cap: int):
        self._rs_count += 1
        if len(feats) < cap:
            feats.append(vec); idxs.append(idx)
        else:
            j = self.rng.randrange(0, self._rs_count)
            if j < cap:
                feats[j] = vec; idxs[j] = idx

    # ---------- 保存（labels + index） ----------
    def _save_labels_and_index(self, save_dir: str, labels: np.ndarray, order_index: List[Tuple[str, int]]) -> None:
        if len(labels) != len(order_index):
            raise ValueError(f"labels({len(labels)}) 与 order_index({len(order_index)}) 长度不一致")
        np.save(os.path.join(save_dir, "labels.npy"), labels)
        with open(os.path.join(save_dir, "index.csv"), "w", encoding="utf-8") as fw:
            fw.write("subject,epoch\n")
            for sid, eid in order_index:
                fw.write(f"{sid},{eid}\n")
        # 额外方便载入
        np.save(os.path.join(save_dir, "subject.npy"), np.array([s for s, _ in order_index], dtype=object))
        np.save(os.path.join(save_dir, "epoch.npy"),   np.array([e for _, e in order_index], dtype=np.int32))

    # ---------- 各 Pipeline ----------
    def _pipeline_mean768(self, k=2, pca_dim=64, batch_rows=200_000,
                          return_labels=True, normalize=True):
        total = self.count_epochs()
        scaler = StandardScaler() if normalize else None

        # pass1: scaler
        if scaler is not None:
            buf = []
            self.log("\n[Pass1] Fitting StandardScaler (mean768)")
            for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="Scaler", ncols=100):
                buf.append(self.epoch_mean768(x))
                if len(buf) >= batch_rows:
                    scaler.partial_fit(np.stack(buf, 0)); buf = []
            if buf: scaler.partial_fit(np.stack(buf, 0)); buf = []

        # pass2: ipca
        ipca = IncrementalPCA(n_components=pca_dim)
        buf = []
        self.log("[Pass2] Fitting IncrementalPCA")
        for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="PCA", ncols=100):
            buf.append(self.epoch_mean768(x))
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
        self.log("[Pass3] Fitting MiniBatchKMeans")
        for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="KMeans", ncols=100):
            buf.append(self.epoch_mean768(x))
            if len(buf) >= batch_rows:
                X = np.stack(buf, 0)
                if scaler is not None: X = scaler.transform(X)
                Xp = ipca.transform(X); km.partial_fit(Xp); buf = []
        if buf:
            X = np.stack(buf, 0)
            if scaler is not None: X = scaler.transform(X)
            Xp = ipca.transform(X); km.partial_fit(Xp)

        if not return_labels:
            return scaler, ipca, km, None, None

        # pass4: predict + order_index
        N = total
        labels, i, buf = np.empty(N, np.int8), 0, []
        order_index: List[Tuple[str, int]] = []
        self.log("[Pass4] Predicting labels & recording order_index")
        for sid, eid, x in tqdm(self.iter_epochs(), total=total, desc="Predict", ncols=100):
            buf.append(self.epoch_mean768(x))
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

    def _pipeline_concat1536(self, k=2, pca_dim=128, batch_rows=200_000,
                             return_labels=True, normalize=True):
        total = self.count_epochs()
        scaler = StandardScaler() if normalize else None

        # pass1
        if scaler is not None:
            buf = []
            self.log("\n[Pass1] Fitting StandardScaler (concat1536)")
            for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="Scaler", ncols=100):
                buf.append(self.epoch_concat1536(x))
                if len(buf) >= batch_rows:
                    scaler.partial_fit(np.stack(buf, 0)); buf = []
            if buf: scaler.partial_fit(np.stack(buf, 0)); buf = []

        # pass2
        ipca = IncrementalPCA(n_components=pca_dim)
        buf = []
        self.log("[Pass2] Fitting IncrementalPCA")
        for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="PCA", ncols=100):
            buf.append(self.epoch_concat1536(x))
            if len(buf) >= batch_rows:
                X = np.stack(buf, 0)
                if scaler is not None: X = scaler.transform(X)
                ipca.partial_fit(X); buf = []
        if buf:
            X = np.stack(buf, 0)
            if scaler is not None: X = scaler.transform(X)
            ipca.partial_fit(X); buf = []

        # pass3
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=self.seed)
        buf = []
        self.log("[Pass3] Fitting MiniBatchKMeans")
        for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="KMeans", ncols=100):
            buf.append(self.epoch_concat1536(x))
            if len(buf) >= batch_rows:
                X = np.stack(buf, 0)
                if scaler is not None: X = scaler.transform(X)
                Xp = ipca.transform(X); km.partial_fit(Xp); buf = []
        if buf:
            X = np.stack(buf, 0)
            if scaler is not None: X = scaler.transform(X)
            Xp = ipca.transform(X); km.partial_fit(Xp)

        if not return_labels:
            return scaler, ipca, km, None, None

        # pass4
        N = total
        labels, i, buf = np.empty(N, np.int8), 0, []
        order_index: List[Tuple[str, int]] = []
        self.log("[Pass4] Predicting labels & recording order_index")
        for sid, eid, x in tqdm(self.iter_epochs(), total=total, desc="Predict", ncols=100):
            buf.append(self.epoch_concat1536(x))
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

    def _pipeline_multiview(self, k=2, pca_dim=64, batch_rows=100_000,
                            return_labels=True, normalize=True):
        total = self.count_epochs()
        scalers = [StandardScaler() if normalize else None for _ in range(8)]
        cache = [[] for _ in range(8)]

        # pass1: scaler(8)
        self.log("\n[Pass1] Fitting 8-view StandardScaler (multiview)")
        for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="Scaler(8 views)", ncols=100):
            V = self.epoch_multiview8(x)  # [8,768]
            for i in range(8): cache[i].append(V[i])
            if len(cache[0]) >= batch_rows:
                for i in range(8):
                    if scalers[i] is not None:
                        scalers[i].partial_fit(np.stack(cache[i], 0))
                cache = [[] for _ in range(8)]
        if len(cache[0]):
            for i in range(8):
                if scalers[i] is not None:
                    scalers[i].partial_fit(np.stack(cache[i], 0))
            cache = [[] for _ in range(8)]

        # pass1b: 方差权重
        self.log("[Pass1b] Estimating variance weights")
        var_sum = np.zeros(8)
        cache = [[] for _ in range(8)]
        for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="Var(8 views)", ncols=100):
            V = self.epoch_multiview8(x)
            for i in range(8): cache[i].append(V[i])
            if len(cache[0]) >= batch_rows:
                for i in range(8):
                    arr = np.stack(cache[i], 0)
                    if scalers[i] is not None: arr = scalers[i].transform(arr)
                    var_sum[i] += arr.var()
                cache = [[] for _ in range(8)]
        if len(cache[0]):
            for i in range(8):
                arr = np.stack(cache[i], 0)
                if scalers[i] is not None: arr = scalers[i].transform(arr)
                var_sum[i] += arr.var()
            cache = [[] for _ in range(8)]
        w = var_sum / (var_sum.sum() + 1e-8)

        # pass2a: ipca
        ipca = IncrementalPCA(n_components=pca_dim)
        buf = []
        self.log("[Pass2a] Fitting IncrementalPCA")
        for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="PCA", ncols=100):
            V = self.epoch_multiview8(x)
            if any(scalers):
                Vs = []
                for i in range(8):
                    v = V[i][None]
                    if scalers[i] is not None: v = scalers[i].transform(v)
                    Vs.append(v[0])
                Vs = np.stack(Vs, 0)
            else:
                Vs = V
            emb = (Vs * w[:, None]).sum(0)  # [768]
            buf.append(emb)
            if len(buf) >= batch_rows:
                ipca.partial_fit(np.stack(buf, 0)); buf = []
        if buf: ipca.partial_fit(np.stack(buf, 0)); buf = []

        # pass2b: kmeans
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=self.seed)
        buf = []
        self.log("[Pass2b] Fitting MiniBatchKMeans")
        for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="KMeans", ncols=100):
            V = self.epoch_multiview8(x)
            if any(scalers):
                Vs = []
                for i in range(8):
                    v = V[i][None]
                    if scalers[i] is not None: v = scalers[i].transform(v)
                    Vs.append(v[0])
                Vs = np.stack(Vs, 0)
            else:
                Vs = V
            emb = (Vs * w[:, None]).sum(0)
            buf.append(emb)
            if len(buf) >= batch_rows:
                xp = ipca.transform(np.stack(buf, 0)); km.partial_fit(xp); buf = []
        if buf:
            xp = ipca.transform(np.stack(buf, 0)); km.partial_fit(xp)

        if not return_labels:
            return (scalers, w), ipca, km, None, None

        # pass3: predict + order_index
        N = total
        labels, i, buf = np.empty(N, np.int8), 0, []
        order_index: List[Tuple[str, int]] = []
        self.log("[Pass3] Predicting labels & recording order_index")
        for sid, eid, x in tqdm(self.iter_epochs(), total=total, desc="Predict", ncols=100):
            V = self.epoch_multiview8(x)
            if any(scalers):
                Vs = []
                for j in range(8):
                    v = V[j][None]
                    if scalers[j] is not None: v = scalers[j].transform(v)
                    Vs.append(v[0])
                Vs = np.stack(Vs, 0)
            else:
                Vs = V
            emb = (Vs * w[:, None]).sum(0)
            buf.append(emb)
            order_index.append((sid, eid))
            if len(buf) >= batch_rows:
                xp = ipca.transform(np.stack(buf, 0))
                labels[i:i+len(buf)] = km.predict(xp); i += len(buf); buf = []
        if buf:
            xp = ipca.transform(np.stack(buf, 0))
            labels[i:i+len(buf)] = km.predict(xp)
        return (scalers, w), ipca, km, labels, order_index

    def _pipeline_patch_vote(self, k=2, pca_dim=64, batch_patches=300_000,
                             vote_threshold=0.5, return_labels=True, normalize=True):
        total = self.count_epochs()
        scaler = StandardScaler() if normalize else None
        ipca = IncrementalPCA(n_components=pca_dim)
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=self.seed)

        # 1a scaler (patch)
        if scaler is not None:
            cur, rows = [], 0
            self.log("\n[Pass1] Fitting StandardScaler (patchx15)")
            for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="Scaler(P)", ncols=100):
                pm = self.patch_mean768(x)  # [15,768]
                cur.append(pm); rows += 15
                if rows >= batch_patches:
                    X = np.concatenate(cur, 0); scaler.partial_fit(X); cur = []; rows = 0
            if rows > 0:
                X = np.concatenate(cur, 0); scaler.partial_fit(X); cur = []; rows = 0

        # 1b ipca
        self.log("[Pass2] Fitting IncrementalPCA (patch)")
        for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="PCA(P)", ncols=100):
            X = self.patch_mean768(x)
            if scaler is not None: X = scaler.transform(X)
            ipca.partial_fit(X)

        # 1c kmeans
        self.log("[Pass3] Fitting MiniBatchKMeans (patch)")
        for _sid, _eid, x in tqdm(self.iter_epochs(), total=total, desc="KMeans(P)", ncols=100):
            X = self.patch_mean768(x)
            if scaler is not None: X = scaler.transform(X)
            Xp = ipca.transform(X)
            km.partial_fit(Xp)

        if not return_labels:
            return scaler, ipca, km, None, None

        # 2 predict + vote + order_index
        N = total
        epoch_labels, i = np.empty(N, np.int8), 0
        order_index: List[Tuple[str, int]] = []
        self.log("[Pass4] Predicting epoch labels (vote) & recording order_index")
        for sid, eid, x in tqdm(self.iter_epochs(), total=total, desc="Predict(P)", ncols=100):
            X = self.patch_mean768(x)
            if scaler is not None: X = scaler.transform(X)
            Xp = ipca.transform(X)
            pl = km.predict(Xp)
            epoch_labels[i] = 1 if (pl.mean() >= vote_threshold) else 0
            order_index.append((sid, eid))
            i += 1
        return scaler, ipca, km, epoch_labels, order_index

    # ---------- 统一入口 ----------
    def run(self, mode: str, save_dir: str, k: int = 2, pca_dim: int = 128,
            batch_rows: int = 200_000, vote_threshold: float = 0.5,
            umap_sample: int = 0, normalize: bool = True, auto_plot: bool = True):
        self._init_logger(save_dir)
        t0 = time.time()
        self.log(f"Start run: mode={mode}, k={k}, pca_dim={pca_dim}, normalize={normalize}, seed={self.seed}")
        self.log(f"features_dir={self.features_dir}, files={len(self.paths)}, total_epochs={self.count_epochs()}")

        if mode == "mean768":
            scaler, ipca, km, labels, order_index = self._pipeline_mean768(
                k=k, pca_dim=min(64, pca_dim), batch_rows=batch_rows,
                return_labels=True, normalize=normalize)

        elif mode == "concat1536":
            scaler, ipca, km, labels, order_index = self._pipeline_concat1536(
                k=k, pca_dim=pca_dim, batch_rows=batch_rows,
                return_labels=True, normalize=normalize)

        elif mode == "multiview":
            (scalers, w), ipca, km, labels, order_index = self._pipeline_multiview(
                k=k, pca_dim=min(64, pca_dim), batch_rows=min(100_000, batch_rows),
                return_labels=True, normalize=normalize)
            joblib.dump(scalers, os.path.join(save_dir, "multiview_scalers.pkl"))
            np.save(os.path.join(save_dir, "multiview_weights.npy"), w)
            scaler = None  # multiview 单独保存

        elif mode == "patch_vote":
            scaler, ipca, km, labels, order_index = self._pipeline_patch_vote(
                k=k, pca_dim=min(64, pca_dim), batch_patches=15_000 * 20,
                vote_threshold=vote_threshold, return_labels=True, normalize=normalize)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 保存模型
        if scaler is not None:
            joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
        joblib.dump(ipca,   os.path.join(save_dir, "pca.pkl"))
        joblib.dump(km,     os.path.join(save_dir, "kmeans.pkl"))

        # 保存标签与顺序索引（严格同序）
        self._save_labels_and_index(save_dir, labels, order_index)

        # 保存配置
        cfg = dict(mode=mode, k=k, pca_dim=pca_dim, batch_rows=batch_rows,
                   vote_threshold=vote_threshold, features_dir=self.features_dir,
                   pattern=self.pattern, order=self.order, normalize=normalize, seed=self.seed)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # UMAP 抽样（可选）
        if umap_sample > 0 and HAS_UMAP:
            self.log("\n[UMAP] Sampling & projecting (for visualization only)")
            emb2d = self._umap_sample(mode=mode, sample_epochs=umap_sample, pca_dim=pca_dim)
            np.save(os.path.join(save_dir, "umap_sample.npy"), emb2d)

        # 自动画图（可选）
        if auto_plot:
            try:
                self.visualize_kmeans_pca2d(save_dir=save_dir, mode=mode,
                                            sample_n=min(200_000, self.count_epochs()))
            except Exception as e:
                self.log(f"[warn] PCA plot failed: {e}")
            if umap_sample > 0 and HAS_UMAP:
                try:
                    self.visualize_umap(save_dir=save_dir, mode=mode,
                                        sample_n=min(umap_sample, 200_000))
                except Exception as e:
                    self.log(f"[warn] UMAP plot failed: {e}")

        self.log(f"[Done] Total time: {time.time() - t0:.1f}s")
        return labels

    # ---------- 汇总 ----------
    def summarize(self, save_dir: str, subject_to_group: Optional[Dict[str, str]] = None):
        labels = np.load(os.path.join(save_dir, "labels.npy"))
        subs = np.load(os.path.join(save_dir, "subject.npy"), allow_pickle=True).tolist()
        assert len(labels) == len(subs), "labels and subject length mismatch"

        vals, cnts = np.unique(labels, return_counts=True)
        self.log("== Cluster counts (overall) ==")
        for v, c in zip(vals, cnts):
            self.log(f"  cluster {v}: {c}")

        if subject_to_group:
            stat: Dict[str, Dict[int, int]] = {}
            for lab, sid in zip(labels, subs):
                grp = subject_to_group.get(sid, "Unknown")
                stat.setdefault(grp, {}).setdefault(int(lab), 0)
                stat[grp][int(lab)] += 1
            self.log("== Cluster counts by group ==")
            for grp, d in stat.items():
                total = sum(d.values())
                frac = {k: f"{v} ({v/total:.1%})" for k, v in d.items()}
                self.log(f"  {grp}: {frac}")

    # ---------- KMeans@PCA 可视化（同空间） ----------
    def visualize_kmeans_pca2d(self, save_dir: str, mode: str = "concat1536", sample_n: int = 200_000):
        import matplotlib.pyplot as plt
        scaler_path = os.path.join(save_dir, "scaler.pkl")
        pca = joblib.load(os.path.join(save_dir, "pca.pkl"))
        labels = np.load(os.path.join(save_dir, "labels.npy"))
        scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

        feats, idxs = [], []
        self._rs_count = 0
        self.log("\n[Plot] PCA 2D (same space as KMeans)")
        for i, (_sid, _eid, x) in enumerate(tqdm(self.iter_epochs(), total=self.count_epochs(),
                                                 desc="Sampling", ncols=100)):
            if   mode=="mean768":      v = self.epoch_mean768(x)
            elif mode=="concat1536":   v = self.epoch_concat1536(x)
            elif mode=="multiview":    v = self.epoch_mean768(x)   # 近似可视化即可
            elif mode=="patch_vote":   v = self.epoch_mean768(x)
            else: raise ValueError(mode)
            self._reservoir_append(feats, idxs, v, i, cap=sample_n)

        X = np.stack(feats, 0)
        if scaler is not None: X = scaler.transform(X)
        Xp = pca.transform(X)[:, :2]
        y  = labels[np.array(idxs, dtype=int)]

        plt.figure(figsize=(8,8))
        plt.scatter(Xp[:,0], Xp[:,1], c=y, s=3, alpha=0.7, cmap="tab10")
        plt.title(f"KMeans@PCA · {mode} · n={len(y)}")
        plt.tight_layout()
        out = os.path.join(save_dir, f"kmeans_pca2d_{mode}.png")
        plt.savefig(out, dpi=300); plt.close()
        self.log(f"[plot] Saved: {out}")

    # ---------- UMAP 可视化（仅展示） ----------
    def visualize_umap(self, save_dir: str, mode: str = "concat1536", sample_n: int = 100_000, pca_dim: int = 64):
        if not HAS_UMAP:
            self.log("[warn] umap-learn is not installed; skip UMAP plot.")
            return
        import matplotlib.pyplot as plt

        feats, idxs = [], []
        self._rs_count = 0
        self.log("\n[Plot] UMAP 2D (visualization only)")
        for i, (_sid, _eid, x) in enumerate(tqdm(self.iter_epochs(), total=self.count_epochs(),
                                                 desc="Sampling", ncols=100)):
            if   mode=="mean768":      v = self.epoch_mean768(x)
            elif mode=="concat1536":   v = self.epoch_concat1536(x)
            elif mode=="multiview":    v = self.epoch_mean768(x)
            elif mode=="patch_vote":   v = self.epoch_mean768(x)
            else: raise ValueError(mode)
            self._reservoir_append(feats, idxs, v, i, cap=sample_n)

        X = np.stack(feats, 0)
        Xs = StandardScaler().fit_transform(X)
        if pca_dim and pca_dim < Xs.shape[1]:
            Xs = PCA(n_components=pca_dim, random_state=self.seed).fit_transform(Xs)
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=self.seed)
        emb = reducer.fit_transform(Xs)

        plt.figure(figsize=(8,8))
        plt.scatter(emb[:,0], emb[:,1], s=3, alpha=0.7)
        plt.title(f"UMAP 2D · {mode} · n={emb.shape[0]}")
        plt.tight_layout()
        out = os.path.join(save_dir, f"umap2d_{mode}.png")
        plt.savefig(out, dpi=300); plt.close()
        self.log(f"[plot] Saved: {out}")

    # ---------- UMAP 抽样（仅保存坐标，非必需） ----------
    def _umap_sample(self, mode: str, sample_epochs: int = 100_000, pca_dim: int = 128):
        assert HAS_UMAP, "pip install umap-learn"
        sample = []
        n = 0
        for _, _, x in self.iter_epochs():
            if mode == "mean768":      v = self.epoch_mean768(x)
            elif mode == "concat1536": v = self.epoch_concat1536(x)
            elif mode == "multiview":  v = self.epoch_mean768(x)   # 简化
            elif mode == "patch_vote": v = self.epoch_mean768(x)
            else: raise ValueError(mode)
            n += 1
            if len(sample) < sample_epochs:
                sample.append(v)
            else:
                j = self.rng.randint(0, n - 1)
                if j < sample_epochs:
                    sample[j] = v
        X = np.stack(sample, 0)

        Xs = StandardScaler().fit_transform(X)
        pca = PCA(n_components=min(pca_dim, Xs.shape[1]), random_state=self.seed).fit(Xs)
        Xp = pca.transform(Xs)
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=self.seed)
        return reducer.fit_transform(Xp)


if __name__ == '__main__':
    am = AnalysisManager(features_dir="/home/user/Sleep/result/no_ckpt", pattern="*.h5", order="patch_channel")
    labels = am.run(mode="concat1536",
                    save_dir="/data/rem_feat_cluster/out_concat1536",
                    k=2,
                    pca_dim=128,
                    batch_rows=200_000,
                    umap_sample=100_000,
                    normalize=True,
                    auto_plot=True)
    am.summarize("/data/rem_feat_cluster/out_concat1536", subject_to_group=None)