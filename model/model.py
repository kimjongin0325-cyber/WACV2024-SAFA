# ✅ 최종 수정: nn.Module 상속 + .to() 지원 추가 버전
model_code = """import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.nn.parallel import DistributedDataParallel as DDP
from model.flownet import *
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Model(nn.Module):  # ✅ nn.Module 상속 추가
    def __init__(self, local_rank=-1):
        super(Model, self).__init__()
        self.flownet = SAFA()

        # ✅ convimg 계층 존재 여부 확인 및 안전 처리
        head_params = []
        if hasattr(self.flownet, 'block') and hasattr(self.flownet.block, 'convimg'):
            try:
                head_params.extend(list(map(id, self.flownet.block.convimg.cnn0.parameters())))
                head_params.extend(list(map(id, self.flownet.block.convimg.cnn1.parameters())))
                head_params.extend(list(map(id, self.flownet.block.convimg.cnn2.parameters())))
                print("✅ convimg 계층이 있는 구버전 Flownet 구조로 감지됨.")
            except AttributeError:
                print("⚠️ convimg 일부 계층 누락 — 최신 SAFA 구조로 감지됨.")
        else:
            print("⚠️ convimg 계층 없음 — 최신 Flownet 구조로 감지됨.")
            head_params = []

        base_params = filter(lambda p: id(p) not in head_params, self.flownet.parameters())
        params = [
            {"params": base_params, 'name':'flow', "lr": 3e-4, "weight_decay": 1e-4}
        ]

        # convimg 계층이 존재할 경우만 추가
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

    def train(self, mode=True):
        super(Model, self).train(mode)
        self.flownet.train(mode)

    def eval(self):
        super(Model, self).eval()
        self.flownet.eval()

    def device(self):
        self.flownet.to(device)

    def inference(self, i0, i1, timestep):
        imgs = torch.cat((i0, i1), 1)
        if isinstance(timestep, (list, tuple)):
            return [self.flownet.inference(imgs, t) for t in timestep]
        return self.flownet.inference(imgs, timestep)
"""

with open("/content/WACV2024-SAFA/model/model.py", "w") as f:
    f.write(model_code)

print("✅ model.py 최종 수정 완료 — nn.Module 상속 + .to() 지원 버전 저장됨.")
