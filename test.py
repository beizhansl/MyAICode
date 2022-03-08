import numpy as np
import cv2
import torch
LAMBDA_DICT = {
            'valid': 1.0, 'hole': 2.0, 'tv': 0.1, 'prc': 0.05, 'style': 120.0}

for key, weight in LAMBDA_DICT.items():
    print('key:',key,'value:',weight)
a = torch.tensor([[[1,2],[3,4]],[[5,6],[7,8]]])
print(a)
# print('---')
# b = torch.flatten(a,1)
# print(b)
# print("---")
# c = a.view(a.size(0),-1)
# print(c
b = a.shape
c = a.size()
print(b,type(b))
print(c,type(c))
