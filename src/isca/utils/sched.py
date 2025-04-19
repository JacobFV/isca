import torch


class WarmupCosine:
    def __init__(self, optimizer, warmup, total_steps, base_lr):
        self.opt = optimizer
        self.w = warmup
        self.t = total_steps
        self.b = base_lr
        self.step_num = 0

    def step(self):
        self.step_num += 1
        if self.step_num < self.w:
            lr = self.b * self.step_num / self.w
        else:
            p = (self.step_num - self.w) / max(1, self.t - self.w)
            lr = 0.5 * self.b * (1 + torch.cos(torch.pi * p))
        for g in self.opt.param_groups:
            g["lr"] = lr
