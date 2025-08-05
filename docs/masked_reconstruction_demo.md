
# Masked Reconstruction Demo

This demo provides a step-by-step example to visualize the reconstruction performance of the pretrained SleepGPT model under masked time and frequency domain inputs. It showcases the model's ability to recover missing segments in both the raw EEG signal and its spectrogram representation.

## 🧠 What It Does

The script performs the following steps:

1. **Loads the pretrained SleepGPT model.**
2. **Loads a downstream EEG dataset** (e.g., SHHS or SleepEDF).
3. **Applies random masking** to parts of the EEG in both time and frequency domains.
4. **Feeds the masked EEG** into the model to reconstruct the missing segments.
5. **Plots and saves the reconstruction results**:
   - Original vs. masked vs. predicted EEG (time-domain).
   - Original vs. masked vs. predicted spectrogram (frequency-domain).

## 📁 File Location

The script can be found in the repository as:

```
main/Visualization/visual_mask.py 
or 
main/Visualization/visual_mask.ipynb 
```

You can modify this script to suit your custom model checkpoint and dataset.

## ⚙️ Configuration

Make sure your config (`main/config.py`) contains the following keys:

```python
{
   'load_path': '/path/to/your/checkpoints/epoch=xx-step=xxxx.ckpt',
   'datasets': ['shhs1'],  # or your selected dataset
   'kfold': False,
   'mask_ratio' = [0.75],
   'visual_setting' = {'mask_same': True, 'mode': 'mask_same'},
  ...  # other standard keys
}
```

You can launch the script using `python` with `Sacred`:

```bash
python main/Visualization/visual_mask.py with load_path=/path/to/checkpoint.ckpt SHHS1_datasets visualization_mask_same
or
using jupyter notebook main/Visualization/visual_mask.ipynb
```

## 📊 Output

For each tested EEG segment, the script will output:

- A figure comparing the **original signal**, **masked signal**, and **reconstructed signal** for each PSG channel.
- A figure showing the **original and reconstructed spectrograms**.
- Saved figures under:

```
/root/Sleep/result/<dataset>/epoch_<x>/predict_<id>_<loss>.svg
```

## 🧩 Channels and Colors

The channels are selected using:

```python
['C3', 'C4', 'EMG', 'EOG', 'F3', 'Fpz', 'O1', 'Pz']
```

Colors are predefined using a fixed color palette.

## 📌 Notes

- The demo stops after generating 20 examples.
- Be sure to install dependencies (`pytorch_lightning`, `matplotlib`, etc.).
- This script uses `matplotlib` to generate `.svg` plots for high-quality publication figures.
- The model should be in **evaluation** mode and assumes the checkpoint is properly trained.

---

For any issues, feel free to raise an [Issue](https://github.com/LordXX505/SleepGPT/issues) or reach out to the maintainers.
