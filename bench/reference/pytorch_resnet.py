import torch
import time
import sys

assert len(sys.argv) == 3, "usage: python pytorch_resnet.py <layer> <repeat>"
n = sys.argv[1]
assert n == "18" or n == "34" or n == "50" or n == "101" or n == "152"
repeat = int(sys.argv[2])

model = torch.hub.load('pytorch/vision:v0.9.0', 'resnet' + n, pretrained=True)
model.eval()

img = torch.randn(1, 3, 224, 224, dtype=torch.float32)

for _ in range(4):
    beg = time.time()
    for _ in range(repeat):
        out = model.forward(img)
    end = time.time()
    print(f'{(end - beg) / repeat}s')
