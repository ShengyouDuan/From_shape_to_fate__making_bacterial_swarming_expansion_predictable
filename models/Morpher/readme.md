# Morpher

| Item | Description |
|---|---|
| Name | **Morpher**: an autoregressive model for temporal binary mask forecasting, aligned with the training and evaluation protocol described in our paper. |
| What this repo provides | 1) Load sequence-organized YOLO-seg polygon annotations (`.txt`) and rasterize them into binary masks;<br>2) Temporal downsampling with a fixed stride `step`;<br>3) Autoregressive forecasting with **Morpher (GRU/LSTM/RNN/TransformerEncoder)**;<br>4) Report **mIoU, mAP@[.50:.95], HD, HD95, ASSD**, with optional physics-consistency statistics via `--phys_stats`. |
| Method overview (paper-consistent) | **SpatialEncoder** encodes each binary mask frame into a latent vector `z_t` and retains multi-scale features for skip connections in the decoder;<br>**Morphon** performs attention-based aggregation over observed latent states with a gated fusion (`alpha`) to form a compact history summary;<br>Temporal modeling uses `arch ∈ {gru, lstm, rnn, transformer}` with sinusoidal temporal positional encoding;<br>Inference is **strict autoregressive**: each predicted frame is fed back (sigmoid → re-encode → append to history) until all future frames are generated. |
| Requirements | Python 3.9+ (3.10 / 3.11 recommended);<br>PyTorch 2.0+ (2.1+ recommended when enabling `torch.compile`);<br>CUDA optional (GPU automatically enables AMP mixed precision);<br>Works on Windows / Linux. |
| Installation | Use a virtual environment if possible. Minimal dependencies:<br><pre><code>pip install numpy scipy pillow torchvision opencv-python tqdm timm</code></pre> |
| Dataset | Training and testing use the **Prediction** subset of **[SwarmEvo](https://huggingface.co/datasets/SwarmEvo)**. See **Dataset layout** below. |

Training and testing only require downloading the **Prediction** subset from **[SwarmEvo](https://huggingface.co/datasets/SwarmEvo)**  
(use the `SwarmEvo/prediction` directory).

Default paths: `dataset/train` and `dataset/test` (overridable via CLI).

Each sequence is a folder containing time-ordered `.txt` files:

```text
dataset/
├── train/
│   ├── 1/
│   │   ├── 1_1.txt
│   │   ├── 1_2.txt
│   │   └── ...
│   └── 2/
│       ├── 2_1.txt
│       └── ...
└── test/
    └── 3/
        ├── 3_1.txt
        └── ...
```


**Images_for_prediction** provides all corresponding raw images and is used **only for visualization**, not required for training or testing. |
| Annotation format (YOLO-seg polygons) | Each `.txt` may contain multiple lines. Each line: `cls x1 y1 x2 y2 ...` with **0–1 normalized** coordinates.<br>Implementation detail: polygons are filled into a single binary mask (multiple instances are unioned); `cls` is kept as a placeholder.<br>Sorting: the trailing number in the filename is used for ordering (use names like `0001.txt`). |
| Training-time data augmentation | During training, geometric augmentations are applied to mask sequences, including random rotation, translation, and horizontal/vertical flipping. Validation and test sets are evaluated **without augmentation**. |
| Optimization objective | Training optimizes a composite loss consisting of **Focal loss**, **Soft IoU loss**, and a **boundary-aware loss** (computed on downsampled masks for efficiency). |
| Key hyperparameters (defaults from Config) | `img_size=640`;<br>`step=25`;<br>`obs_ratio=0.8`;<br>`batch_size=2`, `epochs=300`, `lr=5e-5`;<br>Training uses AdamW (`weight_decay=1e-4`) with warmup + cosine schedule;<br>GPU: AMP mixed precision + optional `torch.compile`. |
| Training command (Transformer example) | <pre><code>python Morpher.py train ^
  --arch transformer ^
  --train_path dataset\train ^
  --val_path dataset\test ^
  --img_size 640 ^
  --step 25 ^
  --obs_ratio 0.8 ^
  --batch_size 2 ^
  --epochs 300 ^
  --lr 5e-5 ^
  --results_dir results ^
  --save_name best_transformer.pth ^
  --log_csv results\train_log_transformer.csv ^
  --torch_compile ^
  --torch_compile_mode max-autotune</code></pre> |
| Training outputs | 1) Best checkpoint selected by **validation mIoU**;<br>2) Optional CSV training log with per-epoch metrics;<br>3) Console summaries and best-model notifications. |
| Test command (Transformer example) | <pre><code>python Morpher.py test ^
  --arch transformer ^
  --weights results\best_transformer.pth ^
  --test_path dataset\test ^
  --img_size 640 ^
  --step 25 ^
  --obs_ratio 0.8 ^
  --results_dir results ^
  --out_csv results\test_metrics.csv ^
  --torch_compile ^
  --torch_compile_mode max-autotune</code></pre> |
| Test outputs | 1) Frame-level metrics: `results/test_outputs/frame_metrics.csv`;<br>2) Sequence-level summary CSV;<br>3) Console summary statistics. |
| Physics consistency (optional) | Enable via `--phys_stats`. Reported fields include `vel_RMSE`, `AI_abs_err`, `H2_abs_err`, `NAS_abs_err`, and `TCI`. |
| Reproducibility notes | Stable filename sorting is required; ensure `T/step ≥ 2`; reduce batch size or disable `torch.compile` if GPU memory is limited. |
| Common issues | Boundary metrics may become NaN when masks are empty; `torch.compile` may slow the first epoch; missing `timm` triggers an internal fallback. |
| Maintainer note | Single-file entry point: `Morpher.py`. Training and testing are invoked via `train` / `test` subcommands. |
---

## Citation

If you use **Morpher** in your research, please cite the accompanying paper:

> *[From shape to fate: making bacterial swarming expansion predictable](https://arxiv.org/abs/2602.01056)*

```bibtex
@article{duan2026shapetofate,
  title     = {From shape to fate: making bacterial swarming expansion predictable},
  author    = {Duan, Shengyou and Wang, Zhaoyang and Xiong, Kaiyi and Zhu, Jin and Gu, Pengxi and Chen, Weijie and Xin, Hongyi and Qu, Zijie},
  journal   = {arXiv preprint arXiv:2602.01056},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.01056}
}
```

---

## License

This code is released for **academic research use only**.
