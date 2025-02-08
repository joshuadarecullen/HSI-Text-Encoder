from torch import nn, Tensor
from text_encoder import TextEncoder
from HSI_transformer import HSIEncoder
from loss import ContrastiveLoss


# Full Multimodal Model
class HSITextModel(nn.Module):
    def __init__(self,
                 input_channels: int = 220,
                 embed_dim: int = 512):

        super().__init__()

        self.hsi_encoder = HSIEncoder(input_channels, embed_dim)
        self.text_encoder = TextEncoder(embed_dim)
        self.loss_fn = ContrastiveLoss()

    def forward(self,
                hsi: Tensor,
                input_ids: list[int],
                attention_mask: Tensor[int]):

        hsi_embed = self.hsi_encoder(hsi)
        text_embed = self.text_encoder(input_ids, attention_mask)
        loss = self.loss_fn(hsi_embed, text_embed)
        return loss, hsi_embed, text_embed
