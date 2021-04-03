import onnx
import struct
import numpy as np
import mxnet as mx
from mxnet.gluon.data.vision import transforms
import onnxruntime as rt
import re

img_path = 'kitten.jpg'
label_path = 'synset.txt'
model_path = 'resnet18-v1-7-opt.onnx'


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_input():
    labels = [x.rstrip() for x in open(label_path).readlines()]

    img = mx.image.imread(img_path)
    img = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])(img)
    img = img.expand_dims(axis=0).asnumpy()
    open('resnet_data/input', 'wb').write(img.tobytes())

    sess = rt.InferenceSession(model_path)
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    pred = softmax(np.squeeze(sess.run([label_name], {input_name: img})[0]))
    for i in np.argsort(pred)[::-1][0:5]:
        print(f'id = {i}, class = {labels[i]}, prob = {pred[i]}')


def cmp_conv_key(node):
    f = re.findall(r'stage(\d+)_conv(\d+)', node.name)
    return int(f[0][0]) * 1000 + int(f[0][1]) if len(f) == 1 else -1


def get_param():
    model = onnx.load(model_path)
    id = 0
    name_map = {}

    for n in sorted(model.graph.node, key=cmp_conv_key):
        if n.op_type == "Conv":
            name_map[n.input[1]] = f'conv{id}_w'
            name_map[n.input[2]] = f'conv{id}_b'
            id += 1
        elif n.op_type == "Gemm":
            name_map[n.input[1]] = f'gemv_w'
            name_map[n.input[2]] = f'gemv_b'

    for i in model.graph.initializer:
        open('resnet_data/' + name_map[i.name], 'wb').write(struct.pack('%sf' % len(i.float_data), *i.float_data))


get_input()
get_param()
