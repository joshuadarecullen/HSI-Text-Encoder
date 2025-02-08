import torch
from torch import nn, Tensor


# Contrastive Loss
class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07) -> None:
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature
        self.criterion = nn.CrossEntropyLoss()

    def forward(self,
                hsi_embeddings: Tensor,
                text_embeddings: Tensor) -> torch.float:
        logits = torch.matmul(hsi_embeddings, text_embeddings.T) \
                / self.temperature
        labels = torch.arange(hsi_embeddings.shape[0]).to(hsi_embeddings.device)
        return self.criterion(logits, labels)
