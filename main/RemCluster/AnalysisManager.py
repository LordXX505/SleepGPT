# analysis_manager_mode.py
# 统一版：epoch(cls/patch_mean) / patch_vote / multiview
# 抽取公共逻辑：CPU/GPU 拟合 & 分批预测；显式顺序保存；tqdm & logging；可视化
import os, re, glob, h5py, json, time, logging, random, argparse
import numpy as np
from typing import List, Tuple, Dict, Optional, Iterable, Callable

from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.cluster import MiniBatchKMeans
import joblib

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


# --------------------------- 工具：日志/环境 ---------------------------
def setup_logger(save_dir: str) -> logging.Logger:
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
    return logger


def detect_gpu_backend(use_gpu_flag: bool, logger: logging.Logger):
    if use_gpu_flag:
        try:
            import cupy as cp
            from cuml.decomposition import PCA as cuPCA
            from cuml.cluster import KMeans as cuKMeans
            logger.info("[backend] Using GPU (RAPIDS cuML).")
            return dict(kind="gpu", cp=cp, cuPCA=cuPCA, cuKMeans=cuKMeans)
        except Exception as e:
            logger.info(f"[backend] GPU requested but cuML not available: {e}. Fallback to CPU.")
    logger.info("[backend] Using CPU (scikit-learn).")
    return dict(kind="cpu", cp=None, cuPCA=None, cuKMeans=None)


# --------------------------- 主类 ---------------------------
class AnalysisManager:
    """
    输入 epoch 形状：
      (61,1536): [0] = CLS(time+freq)，[1:] = 60 个 patch
      (60,1536): 60 个 patch
      (15,4,2,768): 可还原 60×1536；multiview 仅此形状可用
    模式：
      mode = "epoch"       + token_mode in {"cls", "patch_mean"} → 每 epoch 1×1536
      mode = "patch_vote"  → 60×1536 patch 级聚类，epoch 投票
      mode = "multiview"   → [15,4,2,768] → [8,768] 标准化+方差加权 → 768
    """
    def __init__(self, features_dir: str, pattern: str, order: str, seed: int = 42):
        self.features_dir = features_dir
        self.pattern = pattern
        self.order = order
        self.seed = seed
        self.rng = random.Random(seed)

        self.paths: List[str] = sorted(glob.glob(os.path.join(features_dir, pattern)))
        if not self.paths:
            raise FileNotFoundError(f"No files matched: {features_dir}/{pattern}")
        self._epoch_pat = re.compile(r"^epoch_(\d+)$")
        self._total_epochs = None

        self.logger: Optional[logging.Logger] = None
        self.backend: Dict = dict(kind="cpu", cp=None, cuPCA=None, cuKMeans=None)

        # reservoir counter for sampling
        self._rs_count = 0

    # ---------- 基础迭代 ----------
    def count_epochs(self) -> int:
        if self._total_epochs is not None:
            return self._total_epochs
        total = 0
        for p in self.paths:
            with h5py.File(p, "r") as f:
                total += sum(1 for k in f.keys() if self._epoch_pat.match(k))
        self._total_epochs = total
        return total

    def iter_epochs_raw(self):
        """yield (sid, eid, arr_raw) 按确定顺序"""
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

    # ---------- 各种形状转换 ----------
    def _1542_to_60x1536(self, arr1542: np.ndarray) -> np.ndarray:
        # [15,4,2,768] → [15,4,1536] → [60,1536]
        v = np.concatenate([arr1542[...,0,:], arr1542[...,1,:]], axis=-1)
        return v.reshape(15*4, 1536)

    def _extract_cls_and_patches(self, arr: np.ndarray) -> Tuple[Optional[np.ndarray], np.ndarray]:
        if arr.shape == (61, 1536):
            return arr[0], arr[1:]
        if arr.shape == (60, 1536):
            return None, arr
        if arr.shape == (15, 4, 2, 768):
            return None, self._1542_to_60x1536(arr)
        raise ValueError(f"Unknown epoch shape: {arr.shape}")

    def _to_multiview8(self, arr: np.ndarray) -> Optional[np.ndarray]:
        """返回 [8,768]（仅 arr 为 [15,4,2,768] 时可用），否则 None"""
        if arr.shape != (15,4,2,768):
            return None
        v = arr.mean(axis=0)  # [4,2,768]
        return v.reshape(8, 768)

    # ---------- 公共：蓄水池抽样 ----------
    def _reservoir_append(self, feats: list, idxs: list, vec, idx: int, cap: int):
        self._rs_count += 1
        if len(feats) < cap:
            feats.append(vec); idxs.append(idx)
        else:
            j = self.rng.randrange(0, self._rs_count)
            if j < cap:
                feats[j] = vec; idxs[j] = idx

    # ---------- 公共：顺序安全保存 ----------
    def _save_labels_index(self, save_dir: str, labels: np.ndarray, order_index: List[Tuple[str,int]]):
        if len(labels) != len(order_index):
            raise ValueError("labels length != order_index length")
        np.save(os.path.join(save_dir, "labels.npy"), labels)
        with open(os.path.join(save_dir, "index.csv"), "w", encoding="utf-8") as fw:
            fw.write("subject,epoch\n")
            for sid, eid in order_index:
                fw.write(f"{sid},{eid}\n")
        np.save(os.path.join(save_dir, "subject.npy"), np.array([s for s,_ in order_index], dtype=object))
        np.save(os.path.join(save_dir, "epoch.npy"),   np.array([e for _,e in order_index], dtype=np.int32))

    # --------------------------- 抽象：特征生成器 ---------------------------
    def gen_epoch_vector(self, token_mode: str) -> Iterable[Tuple[str,int,np.ndarray]]:
        """
        根据 token_mode 生成 (sid, eid, vec1536)
        - cls        : 需要 (61,1536)，否则抛错
        - patch_mean : 60×1536 平均
        """
        for sid, eid, raw in self.iter_epochs_raw():
            cls, patches = self._extract_cls_and_patches(raw)
            if token_mode == "cls":
                if cls is None:
                    raise ValueError("token_mode='cls' but no CLS in this epoch.")
                yield sid, eid, cls
            elif token_mode == "patch_mean":
                yield sid, eid, patches.mean(axis=0)
            else:
                raise ValueError(f"unknown token_mode {token_mode}")

    def gen_patch_matrix(self) -> Iterable[Tuple[str,int,np.ndarray]]:
        """生成 patch 矩阵 (sid, eid, X[60,1536]) 供 patch_vote 使用"""
        for sid, eid, raw in self.iter_epochs_raw():
            yield sid, eid, self._extract_cls_and_patches(raw)[1]

    def gen_multiview_vec(self, logger: logging.Logger) -> Iterable[Tuple[str,int,np.ndarray]]:
        """
        multiview：仅当 raw 为 [15,4,2,768] 时生效。
        步骤：
          1) 先做 8 视图：V=[8,768]，为减少重复计算，标准化/权重估计写在外部 fit 函数里
          2) 返回 V（不加权），由上层 fit 里做：标准化→估权→加权求和→768
        """
        skipped = 0
        for sid, eid, raw in self.iter_epochs_raw():
            if raw.shape == (15,4,2,768):
                V = raw.mean(axis=0).reshape(8,768)  # [8,768]
                yield sid, eid, V
            else:
                skipped += 1
        if skipped:
            logger.info(f"[multiview] skipped {skipped} epochs (not [15,4,2,768])")

    # --------------------------- 公共：CPU 流式拟合/预测 ---------------------------
    def cpu_fit_epoch_stream(self,
                             vec_iter: Iterable[Tuple[str,int,np.ndarray]],
                             total: int,
                             pca_dim: int,
                             batch_rows: int,
                             normalize: bool,
                             logger: logging.Logger):
        scaler = StandardScaler() if normalize else None
        if scaler is not None:
            buf = []
            logger.info("[CPU] Pass1: StandardScaler")
            for _sid,_eid,v in tqdm(vec_iter, total=total, desc="Scaler", ncols=100):
                buf.append(v)
                if len(buf) >= batch_rows:
                    scaler.partial_fit(np.stack(buf,0)); buf=[]
            if buf: scaler.partial_fit(np.stack(buf,0)); buf=[]

        ipca = IncrementalPCA(n_components=min(pca_dim, 1536))
        # 重新遍历
        logger.info("[CPU] Pass2: IncrementalPCA")
        for _sid,_eid,v in tqdm(self.gen_epoch_vector(token_mode="patch_mean") if False else self.gen_epoch_vector, total=0):
            pass  # 只是避免静态检查，这行不会执行

        # 需要再来一遍 vec_iter，因此把其实现传进来时要传“函数”，这里再取一次
        # 我们改成传函数而不是迭代器本身：
        pass

    # 为了简洁，我们把 CPU/GPU 公共逻辑做成“高阶函数”，由外层传入“如何取向量”的函数。
    # --------------------------- 核心：统一训练/预测（CPU） ---------------------------
    def cpu_fit_predict(self,
                        vec_fn: Callable[[], Iterable[Tuple[str,int,np.ndarray]]],
                        total: int,
                        pca_dim: int,
                        k: int,
                        batch_rows: int,
                        normalize: bool,
                        logger: logging.Logger):
        scaler = StandardScaler() if normalize else None
        # Pass1 Scaler
        if scaler is not None:
            buf = []
            logger.info("[CPU] Pass1: StandardScaler")
            for _sid,_eid,v in tqdm(vec_fn(), total=total, desc="Scaler", ncols=100):
                buf.append(v)
                if len(buf) >= batch_rows:
                    scaler.partial_fit(np.stack(buf,0)); buf=[]
            if buf: scaler.partial_fit(np.stack(buf,0)); buf=[]

        # Pass2 IPCA
        ipca = IncrementalPCA(n_components=min(pca_dim, 1536))
        buf = []
        logger.info("[CPU] Pass2: IncrementalPCA")
        for _sid,_eid,v in tqdm(vec_fn(), total=total, desc="PCA", ncols=100):
            buf.append(v)
            if len(buf) >= batch_rows:
                X = np.stack(buf,0)
                if scaler is not None: X = scaler.transform(X)
                ipca.partial_fit(X); buf=[]
        if buf:
            X = np.stack(buf,0)
            if scaler is not None: X = scaler.transform(X)
            ipca.partial_fit(X); buf=[]

        # Pass3 KMeans
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=self.seed)
        buf = []
        logger.info("[CPU] Pass3: MiniBatchKMeans")
        for _sid,_eid,v in tqdm(vec_fn(), total=total, desc="KMeans", ncols=100):
            buf.append(v)
            if len(buf) >= batch_rows:
                X = np.stack(buf,0)
                if scaler is not None: X = scaler.transform(X)
                Xp = ipca.transform(X)
                km.partial_fit(Xp); buf=[]
        if buf:
            X = np.stack(buf,0)
            if scaler is not None: X = scaler.transform(X)
            Xp = ipca.transform(X); km.partial_fit(Xp)

        # Predict
        labels = np.empty(total, np.int8)
        order_index: List[Tuple[str,int]] = []
        buf, buf_idx, i = [], [], 0
        logger.info("[CPU] Predict & record order_index")
        for sid,eid,v in tqdm(vec_fn(), total=total, desc="Predict", ncols=100):
            buf.append(v); buf_idx.append((sid,eid))
            if len(buf) >= batch_rows:
                X = np.stack(buf,0)
                if scaler is not None: X = scaler.transform(X)
                Xp = ipca.transform(X)
                labels[i:i+len(buf)] = km.predict(Xp)
                i += len(buf)
                order_index.extend(buf_idx)
                buf=[]; buf_idx=[]
        if buf:
            X = np.stack(buf,0)
            if scaler is not None: X = scaler.transform(X)
            Xp = ipca.transform(X)
            labels[i:i+len(buf)] = km.predict(Xp)
            order_index.extend(buf_idx)

        return scaler, ipca, km, labels, order_index

    # --------------------------- 核心：统一训练/预测（GPU） ---------------------------
    def gpu_fit_predict(self,
                        vec_fn: Callable[[], Iterable[Tuple[str,int,np.ndarray]]],
                        total: int,
                        pca_dim: int,
                        k: int,
                        batch_rows: int,
                        normalize: bool,
                        logger: logging.Logger,
                        gpu_fit_samples: int):
        cp = self.backend["cp"]
        cuPCA = self.backend["cuPCA"]
        cuKMeans = self.backend["cuKMeans"]

        # collect sample for fit
        logger.info(f"[GPU] Collecting up to {gpu_fit_samples} samples for PCA/KMeans fit")
        feats = []
        for _sid,_eid,v in tqdm(vec_fn(), total=total, desc="Collect(Fit)", ncols=100):
            feats.append(v.astype(np.float32, copy=False))
            if len(feats) >= gpu_fit_samples:
                break
        X_fit = np.stack(feats,0).astype(np.float32)

        scaler_mean_std = None
        if normalize:
            m = X_fit.mean(axis=0)
            s = X_fit.std(axis=0) + 1e-8
            scaler_mean_std = (m, s)
            X_fit = (X_fit - m) / s

        X_fit_gpu = cp.asarray(X_fit)
        pca = cuPCA(n_components=min(pca_dim, X_fit.shape[1]), random_state=self.seed)
        X_fit_pca = pca.fit_transform(X_fit_gpu)

        km = cuKMeans(n_clusters=k, random_state=self.seed)
        km.fit(X_fit_pca)

        # predict in batches
        labels = np.empty(total, np.int32)
        order_index: List[Tuple[str,int]] = []
        logger.info("[GPU] Predicting in batches")
        buf, buf_idx, i = [], [], 0
        for sid,eid,v in tqdm(vec_fn(), total=total, desc="Predict", ncols=100):
            buf.append(v.astype(np.float32, copy=False)); buf_idx.append((sid,eid))
            if len(buf) >= batch_rows:
                Xb = np.stack(buf,0).astype(np.float32)
                if normalize and scaler_mean_std is not None:
                    m,s = scaler_mean_std
                    Xb = (Xb - m) / s
                Xb_gpu = cp.asarray(Xb)
                Xbp = pca.transform(Xb_gpu)
                pred = km.predict(Xbp).get()
                labels[i:i+len(pred)] = pred
                order_index.extend(buf_idx)
                i += len(pred); buf=[]; buf_idx=[]
        if buf:
            Xb = np.stack(buf,0).astype(np.float32)
            if normalize and scaler_mean_std is not None:
                m,s = scaler_mean_std
                Xb = (Xb - m) / s
            Xb_gpu = cp.asarray(Xb)
            Xbp = pca.transform(Xb_gpu)
            pred = km.predict(Xbp).get()
            labels[i:i+len(pred)] = pred
            order_index.extend(buf_idx)

        return None, pca, km, labels.astype(np.int8), order_index, scaler_mean_std

    # --------------------------- multiview 专用：权重估计与向量生成 ---------------------------
    @staticmethod
    def _multiview_fit_weights(vec_fn_mv: Callable[[], Iterable[Tuple[str,int,np.ndarray]]],
                               total: int,
                               batch_rows: int,
                               normalize: bool,
                               logger: logging.Logger):
        """
        输入 V=[8,768] 多视图，不做加权；这里做两件事：
         1) 为每个视图拟合 scaler（可选）
         2) 基于标准化后的方差估权重 w[8]
        返回：scalers(list 或 None[8]), weights(np.ndarray[8])
        """
        scalers = [StandardScaler() if normalize else None for _ in range(8)]
        caches = [[] for _ in range(8)]

        logger.info("[multiview] Pass1: fit 8 scalers (optional)")
        cnt = 0
        for _sid,_eid,V in tqdm(vec_fn_mv(), total=total, desc="MV-Scaler", ncols=100):
            for i in range(8): caches[i].append(V[i])
            cnt += 1
            if len(caches[0]) * 8 >= batch_rows:
                for i in range(8):
                    if scalers[i] is not None:
                        scalers[i].partial_fit(np.stack(caches[i],0))
                    caches[i] = []
        if len(caches[0]):
            for i in range(8):
                if scalers[i] is not None:
                    scalers[i].partial_fit(np.stack(caches[i],0))
                caches[i] = []

        logger.info("[multiview] Pass1b: estimate variance weights")
        var_sum = np.zeros(8, dtype=np.float64)
        caches = [[] for _ in range(8)]
        for _sid,_eid,V in tqdm(vec_fn_mv(), total=total, desc="MV-Var", ncols=100):
            for i in range(8): caches[i].append(V[i])
            if len(caches[0]) * 8 >= batch_rows:
                for i in range(8):
                    arr = np.stack(caches[i],0)
                    if scalers[i] is not None:
                        arr = scalers[i].transform(arr)
                    var_sum[i] += arr.var()
                    caches[i] = []
        if len(caches[0]):
            for i in range(8):
                arr = np.stack(caches[i],0)
                if scalers[i] is not None:
                    arr = scalers[i].transform(arr)
                var_sum[i] += arr.var()
                caches[i] = []

        w = var_sum / (var_sum.sum() + 1e-12)
        return scalers, w

    def _multiview_epoch_vec_fn(self,
                                scalers: List[Optional[StandardScaler]],
                                weights: np.ndarray) -> Callable[[], Iterable[Tuple[str,int,np.ndarray]]]:
        """返回一个 vec_fn：将 V[8,768] → 标准化 → 加权求和 → 768"""
        def _fn():
            for sid, eid, raw in self.iter_epochs_raw():
                if raw.shape != (15,4,2,768):
                    continue
                V = raw.mean(axis=0).reshape(8,768)  # [8,768]
                if any(scalers):
                    Vn = []
                    for i in range(8):
                        v = V[i][None]
                        if scalers[i] is not None:
                            v = scalers[i].transform(v)
                        Vn.append(v[0])
                    V = np.stack(Vn,0)
                emb = (V * weights[:,None]).sum(0)  # [768]
                yield sid, eid, emb
        return _fn

    # --------------------------- 可视化 ---------------------------
    def _load_saved_scaler(self, save_dir: str):
        sp = os.path.join(save_dir, "scaler.pkl")
        if os.path.exists(sp):
            try:
                return ("cpu", joblib.load(sp))
            except Exception:
                pass
        mp = os.path.join(save_dir, "gpu_scaler_mean.npy")
        sp2 = os.path.join(save_dir, "gpu_scaler_std.npy")
        if os.path.exists(mp) and os.path.exists(sp2):
            return ("gpu", {"mean": np.load(mp), "std": np.load(sp2)})
        return (None, None)

    def _pca_transform_saved(self, save_dir: str, X: np.ndarray) -> np.ndarray:
        pp = os.path.join(save_dir, "pca.pkl")
        if os.path.exists(pp):
            try:
                pca = joblib.load(pp); return pca.transform(X)
            except Exception: pass
        cp = os.path.join(save_dir, "pca_components.npy")
        mp = os.path.join(save_dir, "pca_mean.npy")
        if os.path.exists(cp) and os.path.exists(mp):
            comp = np.load(cp); mean = np.load(mp)
            return (X - mean) @ comp.T
        raise FileNotFoundError("No saved PCA (pca.pkl or pca_components.npy+pca_mean.npy)")

    def _kmeans_predict_saved(self, save_dir: str, Xp: np.ndarray) -> np.ndarray:
        kp = os.path.join(save_dir, "kmeans.pkl")
        if os.path.exists(kp):
            try:
                km = joblib.load(kp); return km.predict(Xp)
            except Exception: pass
        cp = os.path.join(save_dir, "kmeans_centers.npy")
        if os.path.exists(cp):
            centers = np.load(cp)
            diff = Xp[:,None,:] - centers[None,:,:]
            dist2 = np.sum(diff*diff, axis=2)
            return np.argmin(dist2, axis=1)
        raise FileNotFoundError("No saved KMeans (kmeans.pkl or kmeans_centers.npy).")

    def _save_gpu_models(self, save_dir: str, pca, km, scaler_mean_std: Optional[Tuple[np.ndarray,np.ndarray]], logger: logging.Logger):
        try:
            comp = pca.components_.get() if hasattr(pca.components_, "get") else np.asarray(pca.components_)
            mean = pca.mean_.get()       if hasattr(pca.mean_, "get")       else np.asarray(pca.mean_)
            np.save(os.path.join(save_dir, "pca_components.npy"), comp)
            np.save(os.path.join(save_dir, "pca_mean.npy"), mean)
            logger.info("[GPU] Saved pca_components.npy & pca_mean.npy")
        except Exception as e:
            logger.info(f"[warn] cannot save PCA arrays: {e}")
        try:
            centers = km.cluster_centers_.get() if hasattr(km.cluster_centers_, "get") else np.asarray(km.cluster_centers_)
            np.save(os.path.join(save_dir, "kmeans_centers.npy"), centers)
            logger.info("[GPU] Saved kmeans_centers.npy")
        except Exception as e:
            logger.info(f"[warn] cannot save KMeans centers: {e}")
        if scaler_mean_std is not None:
            m,s = scaler_mean_std
            np.save(os.path.join(save_dir, "gpu_scaler_mean.npy"), m)
            np.save(os.path.join(save_dir, "gpu_scaler_std.npy"),  s)
            logger.info("[GPU] Saved gpu_scaler_mean/std.npy")

    def visualize_kmeans_pca2d(self, save_dir: str, mode: str, token_mode: str, sample_n: int = 200_000):
        import matplotlib.pyplot as plt
        feats, idxs = [], []
        self._rs_count = 0
        self.logger.info("\n[Plot] KMeans@PCA 2D")
        total = self.count_epochs()

        if mode == "epoch":
            src_iter = self.gen_epoch_vector(token_mode=token_mode)
            getter = lambda: self.gen_epoch_vector(token_mode=token_mode)
        elif mode == "patch_vote":
            # 展示用 patch_mean
            getter = lambda: self.gen_epoch_vector(token_mode="patch_mean")
            src_iter = getter()
        elif mode == "multiview":
            # 展示用 multiview 的加权向量（需要读取权重&scaler；这里退化为未加权的简展示）
            src_iter = ( (sid,eid, raw.mean(axis=0).reshape(8,768).mean(0))
                         for sid,eid,raw in self.iter_epochs_raw() if raw.shape==(15,4,2,768) )
            getter = lambda: src_iter  # 简化
        else:
            raise ValueError(mode)

        for i,(sid,eid,v) in enumerate(tqdm(src_iter, total=total, desc="Sampling", ncols=100)):
            self._reservoir_append(feats, idxs, v, i, cap=sample_n)
        if not feats:
            self.logger.info("[Plot] No samples for plotting."); return

        X = np.stack(feats,0)
        kind, scaler_obj = self._load_saved_scaler(save_dir)
        if kind=="cpu" and scaler_obj is not None:
            Xs = scaler_obj.transform(X)
        elif kind=="gpu" and scaler_obj is not None:
            m,s = scaler_obj["mean"], scaler_obj["std"]; Xs = (X-m)/(s+1e-8)
        else:
            Xs = X

        Xp = self._pca_transform_saved(save_dir, Xs)
        X2 = Xp[:, :2]
        labels = np.load(os.path.join(save_dir, "labels.npy"))
        y = labels[np.array(idxs, dtype=int)]
        plt.figure(figsize=(8,8))
        plt.scatter(X2[:,0], X2[:,1], c=y, s=3, alpha=0.7, cmap="tab10")
        plt.title(f"KMeans@PCA · {mode}:{token_mode} · n={len(y)}")
        plt.tight_layout()
        out = os.path.join(save_dir, f"kmeans_pca2d_{mode}_{token_mode}.png")
        plt.savefig(out, dpi=300); plt.close()
        self.logger.info(f"[plot] Saved: {out}")

    def visualize_umap(self, save_dir: str, mode: str, token_mode: str, sample_n: int = 100_000, pca_dim: int = 64):
        if not HAS_UMAP:
            self.logger.info("[warn] umap-learn not installed; skip UMAP."); return
        import matplotlib.pyplot as plt
        feats, idxs = [], []
        self._rs_count = 0
        self.logger.info("\n[Plot] UMAP 2D (viz only)")
        total = self.count_epochs()
        if mode == "epoch":
            src_iter = self.gen_epoch_vector(token_mode=token_mode)
        elif mode == "patch_vote":
            src_iter = self.gen_epoch_vector(token_mode="patch_mean")
        elif mode == "multiview":
            src_iter = ( (sid,eid, raw.mean(axis=0).reshape(8,768).mean(0))
                         for sid,eid,raw in self.iter_epochs_raw() if raw.shape==(15,4,2,768) )
        else:
            raise ValueError(mode)

        n = 0
        for (sid,eid,v) in tqdm(src_iter, total=total, desc="Sampling", ncols=100):
            n += 1
            if len(feats) < sample_n:
                feats.append(v); idxs.append(n-1)
            else:
                j = self.rng.randint(0, n-1)
                if j < sample_n:
                    feats[j] = v; idxs[j] = n-1
        if not feats:
            self.logger.info("[Plot] No samples for UMAP."); return

        X = np.stack(feats,0)
        Xs = StandardScaler().fit_transform(X)
        if pca_dim and pca_dim < Xs.shape[1]:
            Xs = PCA(n_components=pca_dim, random_state=self.seed).fit_transform(Xs)
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=self.seed)
        emb = reducer.fit_transform(Xs)
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8,8))
        plt.scatter(emb[:,0], emb[:,1], s=3, alpha=0.7)
        plt.title(f"UMAP 2D · {mode}:{token_mode} · n={emb.shape[0]}")
        plt.tight_layout()
        out = os.path.join(save_dir, f"umap2d_{mode}_{token_mode}.png")
        plt.savefig(out, dpi=300); plt.close()
        self.logger.info(f"[plot] Saved: {out}")

    # --------------------------- 入口 ---------------------------
    def run(self,
            save_dir: str,
            mode: str,                # "epoch" | "patch_vote" | "multiview"
            token_mode: str,          # "cls" | "patch_mean" (仅 mode="epoch")
            k: int,
            pca_dim: int,
            batch_rows: int,
            normalize: bool,
            vote_threshold: float,
            auto_plot: bool,
            umap_sample: int,
            use_gpu: bool,
            gpu_fit_samples: int):
        self.logger = setup_logger(save_dir)
        self.backend = detect_gpu_backend(use_gpu, self.logger)

        total = self.count_epochs()
        self.logger.info(f"Start run: mode={mode}, token_mode={token_mode}, k={k}, pca_dim={pca_dim}, normalize={normalize}, seed={self.seed}")
        self.logger.info(f"features_dir={self.features_dir}, files={len(self.paths)}, total_epochs={total}")

        if mode == "epoch":
            vec_fn = lambda: self.gen_epoch_vector(token_mode=token_mode)
            if self.backend["kind"] == "gpu":
                scaler, pca, km, labels, order_index, gpu_scaler = self.gpu_fit_predict(
                    vec_fn, total, pca_dim, k, batch_rows, normalize, self.logger, gpu_fit_samples
                )
                # 保存 GPU 模型
                self._save_gpu_models(save_dir, pca, km, gpu_scaler, self.logger)
            else:
                scaler, pca, km, labels, order_index = self.cpu_fit_predict(
                    vec_fn, total, pca_dim, k, batch_rows, normalize, self.logger
                )
                if scaler is not None:
                    joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
                joblib.dump(pca, os.path.join(save_dir, "pca.pkl"))
                joblib.dump(km, os.path.join(save_dir, "kmeans.pkl"))

        elif mode == "patch_vote":
            # 直接用 CPU 流式（partial_fit 完整）
            scaler = StandardScaler() if normalize else None
            ipca = IncrementalPCA(n_components=min(pca_dim, 1536))
            km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=self.seed)

            # 1 scaler
            if scaler is not None:
                cur, rows = [], 0
                self.logger.info("[PV] Pass1: Scaler(patch)")
                for _sid,_eid,P in tqdm(self.gen_patch_matrix(), total=total, desc="Scaler(P)", ncols=100):
                    cur.append(P); rows += P.shape[0]
                    if rows >= batch_rows:
                        X = np.vstack(cur); scaler.partial_fit(X); cur=[]; rows=0
                if rows>0:
                    X = np.vstack(cur); scaler.partial_fit(X)

            # 2 ipca
            self.logger.info("[PV] Pass2: IPCA(patch)")
            for _sid,_eid,P in tqdm(self.gen_patch_matrix(), total=total, desc="PCA(P)", ncols=100):
                X = P
                if scaler is not None: X = scaler.transform(X)
                ipca.partial_fit(X)

            # 3 kmeans
            self.logger.info("[PV] Pass3: KMeans(patch)")
            for _sid,_eid,P in tqdm(self.gen_patch_matrix(), total=total, desc="KMeans(P)", ncols=100):
                X = P
                if scaler is not None: X = scaler.transform(X)
                Xp = ipca.transform(X)
                km.partial_fit(Xp)

            # 4 predict + vote
            labels = np.empty(total, np.int8)
            order_index = []
            self.logger.info("[PV] Pass4: Predict+Vote")
            i=0
            for sid,eid,P in tqdm(self.gen_patch_matrix(), total=total, desc="Predict(P)", ncols=100):
                X = P
                if scaler is not None: X = scaler.transform(X)
                Xp = ipca.transform(X)
                pl = km.predict(Xp)
                labels[i] = 1 if (pl.mean() >= vote_threshold) else 0
                order_index.append((sid, eid))
                i += 1

            if scaler is not None:
                joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
            joblib.dump(ipca, os.path.join(save_dir, "pca.pkl"))
            joblib.dump(km, os.path.join(save_dir, "kmeans.pkl"))

        elif mode == "multiview":
            # 仅 [15,4,2,768] 可用
            # 先拟合 8 个视图的 scaler + 方差权重
            vec_mv = lambda: self.gen_multiview_vec(self.logger)
            scalers, w = self._multiview_fit_weights(vec_mv, total, batch_rows, normalize, self.logger)
            np.save(os.path.join(save_dir, "multiview_weights.npy"), w)
            joblib.dump(scalers, os.path.join(save_dir, "multiview_scalers.pkl"))

            # 将多视图映射到 768 作为统一向量，再复用 CPU/GPU 统一训练/预测
            vec_fn = self._multiview_epoch_vec_fn(scalers, w)
            if self.backend["kind"] == "gpu":
                scaler, pca, km, labels, order_index, gpu_scaler = self.gpu_fit_predict(
                    vec_fn, total, min(pca_dim, 768), k, batch_rows, False, self.logger, gpu_fit_samples
                )
                # multiview 模式我们已经做过标准化，故此处 normalize=False
                self._save_gpu_models(save_dir, pca, km, None, self.logger)
            else:
                scaler, pca, km, labels, order_index = self.cpu_fit_predict(
                    vec_fn, total, min(pca_dim, 768), k, batch_rows, False, self.logger
                )
                joblib.dump(pca, os.path.join(save_dir, "pca.pkl"))
                joblib.dump(km,  os.path.join(save_dir, "kmeans.pkl"))

        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 保存标签与顺序索引
        self._save_labels_index(save_dir, labels, order_index)

        # 保存配置
        cfg = dict(mode=mode, token_mode=token_mode, k=k, pca_dim=pca_dim, batch_rows=batch_rows,
                   normalize=normalize, vote_threshold=vote_threshold, features_dir=self.features_dir,
                   pattern=self.pattern, order=self.order, seed=self.seed,
                   backend=self.backend["kind"], gpu_fit_samples=gpu_fit_samples)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # 可视化
        if auto_plot:
            try:
                self.visualize_kmeans_pca2d(save_dir, mode=mode, token_mode=token_mode, sample_n=min(200_000, self.count_epochs()))
            except Exception as e:
                self.logger.info(f"[warn] PCA plot failed: {e}")
        if umap_sample>0 and HAS_UMAP:
            try:
                self.visualize_umap(save_dir, mode=mode, token_mode=token_mode, sample_n=min(umap_sample, 200_000))
            except Exception as e:
                self.logger.info(f"[warn] UMAP plot failed: {e}")

        self.logger.info(f"[Done] Total time: {time.time()-t0:.1f}s")


# --------------------------- CLI ---------------------------
def build_argparser():
    ap = argparse.ArgumentParser(description="REM clustering: epoch/patch_vote/multiview; CPU/GPU; streaming.")
    ap.add_argument("--features_dir", type=str, required=True)
    ap.add_argument("--pattern", type=str, default="*.h5")
    ap.add_argument("--order", type=str, default="patch_channel", choices=["patch_channel","channel_patch"])
    ap.add_argument("--save_dir", type=str, required=True)

    ap.add_argument("--mode", type=str, default="epoch", choices=["epoch","patch_vote","multiview"])
    ap.add_argument("--token_mode", type=str, default="cls", choices=["cls","patch_mean"], help="only for mode=epoch")

    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--pca_dim", type=int, default=128)
    ap.add_argument("--batch_rows", type=int, default=200_000)
    ap.add_argument("--normalize", action="store_true"); ap.add_argument("--no-normalize", dest="normalize", action="store_false"); ap.set_defaults(normalize=True)
    ap.add_argument("--vote_threshold", type=float, default=0.5)

    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--gpu_fit_samples", type=int, default=1_000_000)

    ap.add_argument("--auto_plot", action="store_true")
    ap.add_argument("--umap_sample", type=int, default=0)
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    am = AnalysisManager(features_dir=args.features_dir, pattern=args.pattern, order=args.order)
    am.run(save_dir=args.save_dir,
           mode=args.mode,
           token_mode=args.token_mode,
           k=args.k,
           pca_dim=args.pca_dim,
           batch_rows=args.batch_rows,
           normalize=args.normalize,
           vote_threshold=args.vote_threshold,
           auto_plot=args.auto_plot,
           umap_sample=args.umap_sample,
           use_gpu=args.gpu,
           gpu_fit_samples=args.gpu_fit_samples)