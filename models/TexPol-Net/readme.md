# TexPol-Net
## Training Configuration

| Item               | Description                                                  |
| ------------------ | ------------------------------------------------------------ |
| Framework backend  | PyTorch with Ultralytics YOLO interface. The training pipeline is built on top of the Ultralytics training engine. |
| Determinism        | `torch.use_deterministic_algorithms(False)`. Allows non-deterministic operations to improve training efficiency. This setting can be adjusted to enforce strict reproducibility if required. |
| cuDNN settings     | `cudnn.deterministic=False`, `cudnn.benchmark=True`. Disables deterministic convolution selection and enables cuDNN benchmarking for faster kernel selection. Both options are configurable depending on reproducibility or performance requirements. |
| Model definition   | `ultralytics/cfg/models/Texpol-Net/Texpol-Net.yaml`. Specifies the network architecture of Texpol-Net. Users may modify this YAML file to adjust model depth, width, or module composition. |
| Dataset config     | `data.yaml`. Defines dataset paths, class names, and data splits. The path and dataset content can be freely changed for different experiments. |
| Image size         | `imgsz = 640`. Input image resolution used during training. This value can be adjusted based on dataset characteristics and available computational resources. |
| Training epochs    | `epochs = 800`. Total number of training epochs. Can be increased or decreased depending on convergence behavior and dataset size. |
| Batch size         | `batch = 16`. Number of samples per training batch. This parameter is user-configurable and typically constrained by available memory. |
| Single-class mode  | `single_cls = True`. Treats all instances as a single foreground class. This option can be disabled for multi-class segmentation tasks. |
| DataLoader workers | `workers = 0`. Number of parallel worker processes for data loading. This can be increased to accelerate data loading on systems with sufficient CPU resources. |
| Random seed        | `seed = 0`. Sets the random seed for training initialization. Different seeds can be used to evaluate training stability and variance. |
| Mixed precision    | `amp = False`. Disables automatic mixed-precision training. This option can be enabled (`True`) to reduce memory usage and accelerate training on supported hardware. |
| Device             | `device = 'cpu'`. Specifies the computation device. Can be set to a CUDA device (e.g., `'0'` or `'cuda'`) for GPU-accelerated training. |
| Training entry     | `YOLO(...).train(...)`. Standard Ultralytics training API used to launch the training process with user-defined hyperparameters. |

> **Note:** Non-deterministic cuDNN behavior is intentionally enabled by default to prioritize training throughput over strict reproducibility. All settings above are configurable and can be adapted to different hardware environments and experimental protocols.

------

## Dataset

Training of **TexPol-Net** uses the **Segmentation** subset of **[SwarmEvo](https://huggingface.co/datasets/SwarmEvo)**.

The dataset provides boundary-resolved polygon annotations of swarming colonies in **YOLO-seg format** (`.txt`), paired with the corresponding raw microscopy images. These annotations capture diffuse and fingered swarm fronts and serve as the supervision signal for texture- and geometry-aware segmentation.

All dataset paths, splits, and class definitions are specified via `data.yaml` and can be freely adapted for different experimental settings or datasets.

------

## Citation

If you use **TexPol-Net** in your research, please cite the accompanying paper:

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

------

## License

This code is released for **academic research use only**.