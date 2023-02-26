import os
import math
import torch
import cv2
class SRDetector():
    def __init__(self,SRconfig, my_model, ckp):
        self.args = SRconfig

        self.scale = SRconfig.scale

        self.ckp = ckp
        self.model = my_model

    def prepare(self, *args,round=0):
        device = torch.device(f'cuda:{round}' if self.args.cpu==False and torch.cuda.is_available() else 'cpu')
        print("SR tensor device:",device)
        def _prepare(tensor):
            if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

