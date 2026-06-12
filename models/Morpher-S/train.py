import torch
torch.use_deterministic_algorithms(False)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True
from ultralytics import YOLO
if __name__ == '__main__':
    model = YOLO('TexPol-Net/ultralytics/cfg/models/Texpol-Net/Texpol-Net.yaml')
    model.train(data="TexPol-Net/data.yaml",
                imgsz=640,
                epochs=800,
                single_cls=True,
                batch=16,
                workers=0,
                seed=0,
                amp=False,
                device='cpu'
                )