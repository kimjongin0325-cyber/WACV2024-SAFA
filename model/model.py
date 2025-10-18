# model/model.py (수정 버전)
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from model.flownet import *
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model:
    def __init__(self, local_rank=-1):
        self.flownet = SAFA()

        # ✅ convimg 존재 여부 확인
        head_params = []
        if hasattr(self.flownet, 'block') and hasattr(self.flownet.block, 'convimg'):
            try:
                head_params.extend(list(map(id, self.flownet.block.convimg.cnn0.parameters())))
                head_params.extend(list(map(id, self.flownet.block.convimg.cnn1.parameters())))
                head_params.extend(list(map(id, self.flownet.block.convimg.cnn2.parameters())))
            except AttributeError:
                print("⚠️ convimg 일부 계층이 누락됨. 최신 SAFA 구조로 감지됨.")
        else:
            print("⚠️ convimg 계층 없음: 최신 Flownet 구조로 감지됨.")
            head_params = []

        base_params = filter(lambda p: id(p) not in head_params, self.flownet.parameters())
        params = [
            {"params": base_params, 'name':'flow', "lr": 3e-4, "weight_decay": 1e-4}
        ]

        # ✅ convimg 계층이 존재할 경우만 추가
        if hasattr(self.flownet, 'block') and hasattr(self.flownet.block, 'convimg'):
            params.extend([
                {"params": self.flownet.block.convimg.cnn0.parameters(), 'name':'head0', "lr": 3e-5, "weight_decay": 1e-4},
                {"params": self.flownet.block.convimg.cnn1.parameters(), 'name':'head1', "lr": 3e-5, "weight_decay": 1e-4},
                {"params": self.flownet.block.convimg.cnn2.parameters(), 'name':'head2', "lr": 3e-5, "weight_decay": 1e-4},
            ])

        self.optimG = AdamW(params)
        self.device()
        if local_rank != -1:
            self.flownet = DDP(self.flownet, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)

    def train(self): self.flownet.train()
    def eval(self): self.flownet.eval()
    def device(self): self.flownet.to(device)
    def inference(self, i0, i1, timestep):
        imgs = torch.cat((i0, i1), 1)
        if isinstance(timestep, (list, tuple)):
            return [self.flownet.inference(imgs, t) for t in timestep]
        return self.flownet.inference(imgs, timestep)
