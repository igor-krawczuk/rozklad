import torch as pt
from torch.utils.data.dataset import Dataset


class SimpleNoisy(Dataset):
    def __init__(self,inner):
        super().__init__()
        self.inner=inner
    def __getitem__(self, index: int):
        ground_truth,= self.inner[index][:1]
        noisy=ground_truth+pt.randn_like(ground_truth)
        salt=(pt.rand_like(ground_truth)>0.9)
        pepper=(pt.rand_like(ground_truth)>0.)
        noisy[salt]=1.0
        noisy[pepper]=0.0
        return noisy,ground_truth


    def __len__(self) -> int:
        return len(self.inner)