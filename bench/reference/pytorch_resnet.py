import torch
import time

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet18', pretrained=True)
model.eval()

img = torch.randn(1, 3, 224, 224, dtype=torch.float32)

for _ in range(4):
    beg = time.time()
    for _ in range(2000):
        out = model.forward(img)
    end = time.time()
    print(f'{(end - beg) / 2000}s')
