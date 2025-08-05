
import cv2
import glob
import os
import h5py
import numpy as np
from scipy.signal import welch
from matplotlib import cm
import matplotlib.pyplot as plt
import io
def resize_img(img, max_wh=1800):
    """等比例缩放，让宽或高 <= max_wh"""
    h, w = img.shape[:2]
    if max(h, w) <= max_wh:
        return img                           # 不用缩
    scale = max_wh / max(h, w)
    return cv2.resize(img, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

def spec_from_h5(h5_path, fs=100, nperseg=256):
    """读取 data.h5，逐 epoch 计算 Welch PSD 并串联成时-频图 (H,W,3 BGR)."""
    with h5py.File(h5_path, 'r') as f:
        # 假设只存一个数据集；如需指定，请改 'eeg' -> f['eeg'] ...
        data = f['signal'][:]
    # 逐 epoch 做 Welch，取 0–50 Hz
    pxx_list, f_list = [], None
    for epoch in data:
        f, pxx = welch(epoch, fs=fs, nperseg=nperseg)
        if f_list is None:
            f_list = f
        pxx_list.append(pxx)
    spec = 10 * np.log10(np.stack(pxx_list, axis=0) + 1e-12)  # (N, F)
    spec = np.flipud(spec.T)                                  # F×N, 低频在底
    fig, ax = plt.subplots(figsize=(6, 2.2), dpi=100)
    im = ax.imshow(spec, aspect='auto',
                   origin='upper',
                   cmap='viridis',
                   extent=[0, spec.shape[1], 0, fs / 2])
    ax.set_ylabel("Freq (Hz)")
    ax.set_xlabel("Epoch index")
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_ylim(0, 50)
    ax.set_title("Spectrogram (0–50 Hz)")
    fig.tight_layout()

    buf = io.BytesIO()
    fig.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)

    # Matplotlib PNG → OpenCV BGR
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    spec_img = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)  # BGR
    return resize_img(spec_img)
def image_viewer(root_folder):
    files = sorted(glob.glob(os.path.join(root_folder, "**", "plot_figures",
                                          "*manual_vs_auto_hypnogram.png"), recursive=True))
    if not files:
        print("未找到图片"); return

    idx, show_spec = 0, False
    while True:
        img_path = files[idx]
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        h5_path = os.path.join('/', *img_path.split('/')[:-2], 'data.h5')
        print(h5_path)
        if img.dtype != 'uint8':
            img_show = cv2.convertScaleAbs(img, alpha=255.0/img.max())
        else:
            img_show = img
        img_show = resize_img(img_show)

        # —— 同目录 data.h5 ——
        # h5_path = os.path.join(os.path.dirname(img_path), "data.h5")
        spec_show = None
        if show_spec and os.path.isfile(h5_path):
            try:
                spec_show = spec_from_h5(h5_path)             # 计算时-频图
            except Exception as e:
                print(f"[Warn] 处理 {h5_path} 出错: {e}")
        # —— 显示 ——（PNG 在左，Spectrogram 在右）
        if spec_show is not None:
            h, w = img_show.shape[:2]
            spec_show = cv2.resize(spec_show, (w, h))
            if img_show.shape[2] == 4:
                img_show = cv2.cvtColor(img_show, cv2.COLOR_BGRA2BGR)
            if spec_show.shape[2] == 4:
                spec_show = cv2.cvtColor(spec_show, cv2.COLOR_BGRA2BGR)
            concat = np.hstack([img_show, spec_show])
            cv2.imshow("viewer", concat)
        else:
            cv2.imshow("viewer", img_show)

        key = cv2.waitKey(0) & 0xFF
        if key in (ord('q'), 27):
            break
        elif key in (ord('d'), 83):          # next
            idx = (idx + 1) % len(files)
        elif key in (ord('a'), 81):          # prev
            idx = (idx - 1) % len(files)
        elif key in (ord('s'), ord('S')):    # toggle spectrogram
            show_spec = not show_spec

    cv2.destroyAllWindows()
def read_valid():
    res_dict = {}
    for name in ['第一批', '第二批']:
        root_path = os.path.join('/Users/hwx_admin/Downloads', name)
        data_path = os.path.join(root_path, 'prep', 'valid_ods.npy')
        data = np.load(data_path, allow_pickle=True).item()
        print(data.keys())
        for k, v in data.items():
            res_dict[k] = v
    print(sum(list(res_dict.values())), len(res_dict))
    np.save('/Users/hwx_admin/Downloads/valid.npy', arr=res_dict, allow_pickle=True)
if __name__ == '__main__':
    name = '第二批'
    root_path = os.path.join('/Users/hwx_admin/Downloads', name)
    # read_valid()
    folder = '/Volumes/T7/data/ums/pic'
    image_viewer(os.path.join(root_path, 'data_new'))
    # 137 140 149 170 182 185 202 054 060 081 094 zzj