import torch


class TrainingMonitor:

    def __init__(self):
        self.loss_history = []
        self.gradient_norms = []

    def record_loss(self, loss):
        self.loss_history.append(loss)

    def record_gradient_norm(self, model):

        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
