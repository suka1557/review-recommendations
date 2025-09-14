import torch.nn as nn
import torch.nn.functional as F


class ReviewClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, num_classes=5, pad_token_id=2):
        super(ReviewClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_token_id)

        # simple encoder: average pooling over embeddings
        self.fc1 = nn.Linear(embed_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, input_ids, attention_mask=None):
        """
        input_ids: [batch_size, seq_len]
        attention_mask: [batch_size, seq_len] (1 for tokens, 0 for padding)
        """
        x = self.embedding(input_ids)  # [B, L, D]

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)  # [B, L, 1]
            x = (x * mask).sum(1) / mask.sum(1).clamp(min=1e-9)  # masked mean pooling
        else:
            x = x.mean(1)

        x = F.relu(self.fc1(x))
        x = self.fc2(x)  # logits [B, num_classes]
        return x
