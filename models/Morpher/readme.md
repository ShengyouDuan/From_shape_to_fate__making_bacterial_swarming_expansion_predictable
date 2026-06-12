# Morpher-F

| Item | Description |
|---|---|
| Name | **Morpher-F**: an autoregressive model for temporal binary mask forecasting, aligned with the training and evaluation protocol described in our paper. |
| What this repo provides | 1) Load sequence-organized YOLO-seg polygon annotations (`.txt`) and rasterize them into binary masks;<br>2) Temporal downsampling with a fixed stride `step`;<br>3) Autoregressive forecasting with **Morpher-F (GRU/LSTM/RNN/TransformerEncoder)**;<br>4) Report **mIoU, mAP@[.50:.95], HD, HD95, ASSD**, with optional physics-consistency statistics via `--phys_stats`. |
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


| Training command (Transformer example) | See command below |

```bash
python Morpher-F.py train ^
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
  --torch_compile_mode max-autotune
```
| Test command (Transformer example) | See command below |
```
python Morpher-F.py test ^
  --arch transformer ^
  --weights results\best_transformer.pth ^
  --test_path dataset\test ^
  --img_size 640 ^
  --step 25 ^
  --obs_ratio 0.8 ^
  --results_dir results ^
  --out_csv results\test_metrics.csv ^
  --torch_compile ^
  --torch_compile_mode max-autotune
```

## Citation

If you use **Morpher-F** in your research, please cite the accompanying paper:

> *[Population-Scale Advancing Interface Modeling Reveals How Bacterial Swarms Encode Future Spatial Architecture](https://arxiv.org/abs/2602.01056)*

```bibtex
@article{duan2026shapetofate,
  title     = {Population-Scale Advancing Interface Modeling Reveals How Bacterial Swarms Encode Future Spatial Architecture},
  author    = {Duan, Shengyou and Wang, Zhaoyang and Xiong, Kaiyi and Zhu, Jin and Gu, Pengxi and Chen, Weijie and Xin, Hongyi and Qu, Zijie},
  journal   = {arXiv preprint arXiv:2602.01056},
  year      = {2026},
  url       = {https://arxiv.org/abs/2602.01056}
}
```

---

## License

This code is released for **academic research use only**.
