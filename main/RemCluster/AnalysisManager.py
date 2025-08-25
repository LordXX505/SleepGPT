# analysis_manager_mode.py
import os, re, glob, h5py, random, json
import numpy as np
from typing import List, Tuple, Dict, Optional

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA, PCA
from sklearn.cluster import MiniBatchKMeans, KMeans
import joblib

try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False


class AnalysisManager:
    """
    超大规模 REM 聚类分析（流式）：
    - 输入：每 subject 一个 .h5；每 epoch dataset 形如 (60,1536) 或 [15,4,2,768]
    - mode:
        - 'mean768'     : (channel,domain)均值→patch均值→[768]→IPCA→MiniBatchKMeans
        - 'concat1536'  : 保留time/freq concat→[1536]→IPCA→MiniBatchKMeans
        - 'multiview'   : 8视图(4通道×2域)标准化+方差加权→[768]→IPCA→MiniBatchKMeans
        - 'patch_vote'  : patch级聚类(15×[768])→epoch投票
    - run(): 按模式完成 拟合→预测→保存
    - summarize(): 读取保存结果并统计簇分布，可分组比较（如 OSA vs Normal）
    """

    def __init__(self,
                 features_dir: str,
                 pattern: str = "*.h5",
                 order: str = "patch_channel",   # 或 "channel_patch"
                 seed: int = 42):
        self.features_dir = features_dir
        self.pattern = pattern
        self.order = order
        self.rng = random.Random(seed)

        self.paths: List[str] = sorted(glob.glob(os.path.join(features_dir, pattern)))
        if not self.paths:
            raise FileNotFoundError(f"No files matched: {features_dir}/{pattern}")
        self._epoch_pat = re.compile(r"^epoch_(\d+)$")

    # ------------- 基础 I/O -------------
    def _decode_epoch(self, arr: np.ndarray) -> np.ndarray:
        """ (60,1536) or [15,4,2,768] -> [15,4,2,768] """
        if arr.shape == (60, 1536):
            if self.order == "patch_channel":
                x = arr.reshape(15, 4, 1536)
            else:
                x = arr.reshape(4, 15, 1536).transpose(1, 0, 2)
            return x.reshape(15, 4, 2, 768)
        if arr.shape == (15, 4, 2, 768):
            return arr
        raise ValueError(f"Unknown epoch shape: {arr.shape}")

    def iter_epochs(self):
        """ yield (subject_id, epoch_id, epoch_array[15,4,2,768]) """
        for p in self.paths:
            sid = os.path.splitext(os.path.basename(p))[0]
            with h5py.File(p, "r") as f:
                for k in f.keys():
                    m = self._epoch_pat.match(k)
                    if not m:
                        continue
                    eid = int(m.group(1))
                    arr = f[k][()]
                    yield sid, eid, self._decode_epoch(arr)

    def count_epochs(self) -> int:
        total = 0
        for p in self.paths:
            with h5py.File(p, "r") as f:
                total += sum(1 for k in f.keys() if self._epoch_pat.match(k))
        return total

    def build_epoch_index(self) -> Tuple[List[str], np.ndarray]:
        """ 返回 (unique_subjects, epoch_subject_idx[N]) """
        subjects, s2i, idx = [], {}, []
        for p in self.paths:
            sid = os.path.splitext(os.path.basename(p))[0]
            if sid not in s2i:
                s2i[sid] = len(subjects); subjects.append(sid)
            with h5py.File(p, "r") as f:
                n = sum(1 for k in f.keys() if self._epoch_pat.match(k))
                idx.extend([s2i[sid]] * n)
        return subjects, np.array(idx, dtype=np.int64)

    # ------------- 各模式的特征构造 -------------
    @staticmethod
    def epoch_mean768(x1542: np.ndarray) -> np.ndarray:
        # [15,4,2,768] -> [768]
        return x1542.mean(axis=(1, 2)).mean(axis=0)

    @staticmethod
    def epoch_concat1536(x1542: np.ndarray) -> np.ndarray:
        # [15,4,2,768] -> [2,768] (保留域) -> concat [1536]
        dom = x1542.mean(axis=(0, 1))  # [2,768]
        return dom.reshape(-1)

    @staticmethod
    def epoch_multiview8(x1542: np.ndarray) -> np.ndarray:
        # [15,4,2,768] -> 8视图 [8,768]（仅做平均，不加权；权重训练在管线里）
        v = x1542.mean(axis=0)  # [4,2,768]
        return v.reshape(8, 768)

    @staticmethod
    def patch_mean768(x1542: np.ndarray) -> np.ndarray:
        # [15,4,2,768] -> [15,768]
        return x1542.mean(axis=(1, 2))

    # ------------- 管线：mean768 -------------
    def _pipeline_mean768(self, k=2, pca_dim=64, batch_rows=200_000, return_labels=False):
        scaler = StandardScaler()
        # pass1
        buf = []
        for _, _, x in self.iter_epochs():
            buf.append(self.epoch_mean768(x))
            if len(buf) >= batch_rows:
                scaler.partial_fit(np.stack(buf, 0)); buf = []
        if buf: scaler.partial_fit(np.stack(buf, 0)); buf = []
        # pass2
        ipca = IncrementalPCA(n_components=pca_dim)
        buf = []
        for _, _, x in self.iter_epochs():
            buf.append(self.epoch_mean768(x))
            if len(buf) >= batch_rows:
                xb = scaler.transform(np.stack(buf, 0)); ipca.partial_fit(xb); buf = []
        if buf:
            xb = scaler.transform(np.stack(buf, 0)); ipca.partial_fit(xb); buf = []
        # pass3
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=42)
        buf = []
        for _, _, x in self.iter_epochs():
            buf.append(self.epoch_mean768(x))
            if len(buf) >= batch_rows:
                xp = ipca.transform(scaler.transform(np.stack(buf, 0)))
                km.partial_fit(xp); buf = []
        if buf:
            xp = ipca.transform(scaler.transform(np.stack(buf, 0))); km.partial_fit(xp)
        if not return_labels:
            return scaler, ipca, km, None
        # predict
        N = self.count_epochs()
        labels, i, buf = np.empty(N, np.int8), 0, []
        for _, _, x in self.iter_epochs():
            buf.append(self.epoch_mean768(x))
            if len(buf) >= batch_rows:
                xp = ipca.transform(scaler.transform(np.stack(buf, 0)))
                labels[i:i+len(buf)] = km.predict(xp); i += len(buf); buf = []
        if buf:
            xp = ipca.transform(scaler.transform(np.stack(buf, 0)))
            labels[i:i+len(buf)] = km.predict(xp)
        return scaler, ipca, km, labels

    # ------------- 管线：concat1536 -------------
    def _pipeline_concat1536(self, k=2, pca_dim=128, batch_rows=200_000, return_labels=False):
        scaler = StandardScaler()
        # pass1
        buf = []
        for _, _, x in self.iter_epochs():
            buf.append(self.epoch_concat1536(x))
            if len(buf) >= batch_rows:
                scaler.partial_fit(np.stack(buf, 0)); buf = []
        if buf: scaler.partial_fit(np.stack(buf, 0)); buf = []
        # pass2
        ipca = IncrementalPCA(n_components=pca_dim)
        buf = []
        for _, _, x in self.iter_epochs():
            buf.append(self.epoch_concat1536(x))
            if len(buf) >= batch_rows:
                xb = scaler.transform(np.stack(buf, 0)); ipca.partial_fit(xb); buf = []
        if buf:
            xb = scaler.transform(np.stack(buf, 0)); ipca.partial_fit(xb); buf = []
        # pass3
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=42)
        buf = []
        for _, _, x in self.iter_epochs():
            buf.append(self.epoch_concat1536(x))
            if len(buf) >= batch_rows:
                xp = ipca.transform(scaler.transform(np.stack(buf, 0)))
                km.partial_fit(xp); buf = []
        if buf:
            xp = ipca.transform(scaler.transform(np.stack(buf, 0))); km.partial_fit(xp)
        if not return_labels:
            return scaler, ipca, km, None
        # predict
        N = self.count_epochs()
        labels, i, buf = np.empty(N, np.int8), 0, []
        for _, _, x in self.iter_epochs():
            buf.append(self.epoch_concat1536(x))
            if len(buf) >= batch_rows:
                xp = ipca.transform(scaler.transform(np.stack(buf, 0)))
                labels[i:i+len(buf)] = km.predict(xp); i += len(buf); buf = []
        if buf:
            xp = ipca.transform(scaler.transform(np.stack(buf, 0)))
            labels[i:i+len(buf)] = km.predict(xp)
        return scaler, ipca, km, labels

    # ------------- 管线：multiview (8视图方差加权) -------------
    def _pipeline_multiview(self, k=2, pca_dim=64, batch_rows=100_000, return_labels=False):
        # pass1: 拟合8个view的scaler + 估计方差权重
        scalers = [StandardScaler() for _ in range(8)]
        cache = [[] for _ in range(8)]
        for _, _, x in self.iter_epochs():
            V = self.epoch_multiview8(x)  # [8,768]
            for i in range(8): cache[i].append(V[i])
            if len(cache[0]) >= batch_rows:
                for i in range(8):
                    scalers[i].partial_fit(np.stack(cache[i], 0))
                    cache[i] = []
        if len(cache[0]):
            for i in range(8):
                scalers[i].partial_fit(np.stack(cache[i], 0))
                cache[i] = []

        # 方差权重
        var_sum = np.zeros(8); chunks = 0
        cache = [[] for _ in range(8)]
        for _, _, x in self.iter_epochs():
            V = self.epoch_multiview8(x)
            for i in range(8): cache[i].append(V[i])
            if len(cache[0]) >= batch_rows:
                for i in range(8):
                    arrs = scalers[i].transform(np.stack(cache[i], 0))
                    var_sum[i] += arrs.var()
                    cache[i] = []
                chunks += 1
        if len(cache[0]):
            for i in range(8):
                arrs = scalers[i].transform(np.stack(cache[i], 0))
                var_sum[i] += arrs.var()
                cache[i] = []
            chunks += 1
        w = var_sum / (var_sum.sum() + 1e-8)  # [8]

        # pass2: 拟合 IPCA + KMeans（在加权后的768上）
        ipca = IncrementalPCA(n_components=pca_dim)
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=42)

        # 2a fit ipca
        buf = []
        for _, _, x in self.iter_epochs():
            V = self.epoch_multiview8(x)  # [8,768]
            Vs = np.stack([scalers[i].transform(V[i][None])[0] for i in range(8)], 0)
            emb = (Vs * w[:, None]).sum(0)  # [768]
            buf.append(emb)
            if len(buf) >= batch_rows:
                ipca.partial_fit(np.stack(buf, 0)); buf = []
        if buf: ipca.partial_fit(np.stack(buf, 0)); buf = []

        # 2b fit kmeans
        buf = []
        for _, _, x in self.iter_epochs():
            V = self.epoch_multiview8(x)
            Vs = np.stack([scalers[i].transform(V[i][None])[0] for i in range(8)], 0)
            emb = (Vs * w[:, None]).sum(0)
            buf.append(emb)
            if len(buf) >= batch_rows:
                xp = ipca.transform(np.stack(buf, 0)); km.partial_fit(xp); buf = []
        if buf:
            xp = ipca.transform(np.stack(buf, 0)); km.partial_fit(xp)

        if not return_labels:
            return (scalers, w), ipca, km, None

        # predict
        N = self.count_epochs()
        labels, i, buf = np.empty(N, np.int8), 0, []
        for _, _, x in self.iter_epochs():
            V = self.epoch_multiview8(x)
            Vs = np.stack([scalers[i].transform(V[i][None])[0] for i in range(8)], 0)
            emb = (Vs * w[:, None]).sum(0)
            buf.append(emb)
            if len(buf) >= batch_rows:
                xp = ipca.transform(np.stack(buf, 0))
                labels[i:i+len(buf)] = km.predict(xp); i += len(buf); buf = []
        if buf:
            xp = ipca.transform(np.stack(buf, 0))
            labels[i:i+len(buf)] = km.predict(xp)
        return (scalers, w), ipca, km, labels

    # ------------- 管线：patch_vote -------------
    def _pipeline_patch_vote(self, k=2, pca_dim=64, batch_patches=300_000, vote_threshold=0.5, return_labels=True):
        scaler = StandardScaler()
        ipca = IncrementalPCA(n_components=pca_dim)
        km = MiniBatchKMeans(n_clusters=k, batch_size=4096, random_state=42)

        # 1a fit scaler
        cur, rows = [], 0
        for _, _, x in self.iter_epochs():
            pm = self.patch_mean768(x)  # [15,768]
            cur.append(pm); rows += 15
            if rows >= batch_patches:
                X = np.concatenate(cur, 0); scaler.partial_fit(X); cur = []; rows = 0
        if rows > 0:
            X = np.concatenate(cur, 0); scaler.partial_fit(X); cur = []; rows = 0

        # 1b fit ipca
        for _, _, x in self.iter_epochs():
            X = scaler.transform(self.patch_mean768(x))  # [15,768]
            ipca.partial_fit(X)

        # 1c fit kmeans
        for _, _, x in self.iter_epochs():
            Xp = ipca.transform(scaler.transform(self.patch_mean768(x)))
            km.partial_fit(Xp)

        if not return_labels:
            return scaler, ipca, km, None

        # 2 predict + vote
        N = self.count_epochs()
        epoch_labels, i = np.empty(N, np.int8), 0
        for _, _, x in self.iter_epochs():
            Xp = ipca.transform(scaler.transform(self.patch_mean768(x)))
            pl = km.predict(Xp)  # [15]
            epoch_labels[i] = 1 if (pl.mean() >= vote_threshold) else 0
            i += 1
        return scaler, ipca, km, epoch_labels

    # ------------- 统一入口：run -------------
    def run(self,
            mode: str,
            save_dir: str,
            k: int = 2,
            pca_dim: int = 128,
            batch_rows: int = 200_000,
            vote_threshold: float = 0.5,
            umap_sample: int = 0):
        """
        统一运行：拟合→预测→保存
        保存内容：
          - config.json
          - labels.npy
          - index.csv (subject,epoch)
          - (可选) umap_sample.npy  (仅抽样)
          - 模型：scaler.pkl / pca.pkl / kmeans.pkl / multiview_scalers.pkl / multiview_weights.npy
        """
        os.makedirs(save_dir, exist_ok=True)

        if mode == "mean768":
            scaler, ipca, km, labels = self._pipeline_mean768(k=k, pca_dim=min(64, pca_dim),
                                                              batch_rows=batch_rows, return_labels=True)
            joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
            joblib.dump(ipca,   os.path.join(save_dir, "pca.pkl"))
            joblib.dump(km,     os.path.join(save_dir, "kmeans.pkl"))

        elif mode == "concat1536":
            scaler, ipca, km, labels = self._pipeline_concat1536(k=k, pca_dim=pca_dim,
                                                                 batch_rows=batch_rows, return_labels=True)
            joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
            joblib.dump(ipca,   os.path.join(save_dir, "pca.pkl"))
            joblib.dump(km,     os.path.join(save_dir, "kmeans.pkl"))

        elif mode == "multiview":
            (scalers, w), ipca, km, labels = self._pipeline_multiview(k=k, pca_dim=min(64, pca_dim),
                                                                      batch_rows=min(100_000, batch_rows),
                                                                      return_labels=True)
            joblib.dump(scalers, os.path.join(save_dir, "multiview_scalers.pkl"))
            np.save(os.path.join(save_dir, "multiview_weights.npy"), w)
            joblib.dump(ipca, os.path.join(save_dir, "pca.pkl"))
            joblib.dump(km,   os.path.join(save_dir, "kmeans.pkl"))

        elif mode == "patch_vote":
            scaler, ipca, km, labels = self._pipeline_patch_vote(k=k, pca_dim=min(64, pca_dim),
                                                                 batch_patches=15_000 * 20,  # 可调
                                                                 vote_threshold=vote_threshold,
                                                                 return_labels=True)
            joblib.dump(scaler, os.path.join(save_dir, "scaler.pkl"))
            joblib.dump(ipca,   os.path.join(save_dir, "pca.pkl"))
            joblib.dump(km,     os.path.join(save_dir, "kmeans.pkl"))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # 保存标签与索引
        np.save(os.path.join(save_dir, "labels.npy"), labels)
        subs, sub_idx = self.build_epoch_index()
        with open(os.path.join(save_dir, "index.csv"), "w", encoding="utf-8") as fw:
            fw.write("subject,epoch\n")
            # epoch顺序与 iter_epochs 顺序一致 → 与 labels 对齐
            for p in self.paths:
                sid = os.path.splitext(os.path.basename(p))[0]
                with h5py.File(p, "r") as f:
                    eids = sorted(int(m.group(1)) for k in f.keys() if (m:=self._epoch_pat.match(k)))
                    for eid in eids:
                        fw.write(f"{sid},{eid}\n")

        # 保存配置
        cfg = dict(mode=mode, k=k, pca_dim=pca_dim, batch_rows=batch_rows,
                   vote_threshold=vote_threshold, features_dir=self.features_dir,
                   pattern=self.pattern, order=self.order)
        with open(os.path.join(save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)

        # （可选）抽样UMAP
        if umap_sample > 0 and HAS_UMAP:
            emb2d = self._umap_sample(mode=mode, sample_epochs=umap_sample, pca_dim=pca_dim)
            np.save(os.path.join(save_dir, "umap_sample.npy"), emb2d)

        return labels

    # ------------- 统一展示：summarize -------------
    def summarize(self, save_dir: str, subject_to_group: Optional[Dict[str, str]] = None):
        """
        从 save_dir 读 labels + index，汇总簇计数；如提供 group 映射则按组汇总
        """
        labels = np.load(os.path.join(save_dir, "labels.npy"))
        # 读取 index
        subs, eids = [], []
        with open(os.path.join(save_dir, "index.csv"), "r", encoding="utf-8") as f:
            next(f)  # skip header
            for line in f:
                s, e = line.strip().split(",")
                subs.append(s); eids.append(int(e))
        assert len(labels) == len(subs), "labels and index length mismatch"

        # 总体分布
        vals, cnts = np.unique(labels, return_counts=True)
        print("== Cluster counts (overall) ==")
        for v, c in zip(vals, cnts):
            print(f"  cluster {v}: {c}")

        # 分组分布
        if subject_to_group:
            stat: Dict[str, Dict[int, int]] = {}
            for lab, sid in zip(labels, subs):
                grp = subject_to_group.get(sid, "Unknown")
                stat.setdefault(grp, {}).setdefault(int(lab), 0)
                stat[grp][int(lab)] += 1
            print("== Cluster counts by group ==")
            for grp, d in stat.items():
                total = sum(d.values())
                frac = {k: f"{v} ({v/total:.1%})" for k, v in d.items()}
                print(f"  {grp}: {frac}")

    # ------------- UMAP 抽样（按所选 mode） -------------
    def _umap_sample(self, mode: str, sample_epochs: int = 100_000, pca_dim: int = 128):
        assert HAS_UMAP, "pip install umap-learn"
        sample = []
        n = 0
        for _, _, x in self.iter_epochs():
            if mode == "mean768":
                v = self.epoch_mean768(x)
            elif mode == "concat1536":
                v = self.epoch_concat1536(x)
            elif mode == "multiview":
                # 先简化用 mean768 采样可视化；多视图严格需要权重和scaler，建议用 run 后的 pca/kmeans 再做
                v = self.epoch_mean768(x)
            elif mode == "patch_vote":
                v = self.epoch_mean768(x)
            else:
                raise ValueError(mode)
            n += 1
            if len(sample) < sample_epochs:
                sample.append(v)
            else:
                j = self.rng.randint(0, n - 1)
                if j < sample_epochs:
                    sample[j] = v
        X = np.stack(sample, 0)

        scaler = StandardScaler().fit(X)
        Xs = scaler.transform(X)
        pca = PCA(n_components=min(pca_dim, Xs.shape[1]), random_state=42).fit(Xs)
        Xp = pca.transform(Xs)
        reducer = umap.UMAP(n_neighbors=25, min_dist=0.1, random_state=42)
        return reducer.fit_transform(Xp)

if __name__ == '__main__':
    am = AnalysisManager(features_dir="./features_all_subjects", pattern="*.h5", order="patch_channel")

    # 1) 选择模式并运行（会自动保存模型与结果）
    #   'mean768' | 'concat1536' | 'multiview' | 'patch_vote'
    labels = am.run(mode="concat1536",
                    save_dir="./out_concat1536",
                    k=2,
                    pca_dim=128,
                    batch_rows=200_000,
                    umap_sample=100_000)  # 抽样保存 UMAP

    # 2) 展示（总体 + 分组可选）
    # subject_to_group = {"subj0001":"OSA", "subj0002":"Normal", ...}
    am.summarize("./out_concat1536", subject_to_group=None)