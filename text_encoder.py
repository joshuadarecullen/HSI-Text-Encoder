import torch
import torch.nn as nn, Tensor
from transformers import BertModel, BertTokenizer
import torchvision.models as models

# Text Encoder (BERT)
class TextEncoder(nn.Module):
    def __init__(self, embed_dim=512):
        super(TextEncoder, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.fc = nn.Linear(self.bert.config.hidden_size, embed_dim)
    
    def forward(self, input_ids: list[int], attention_mask: Tensor[int]):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)




