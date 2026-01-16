import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

LABEL_SMOOTHING = 0.1
FOCAL_GAMMA = 2.0


class WeightedLayerPooling(nn.Module):
    """
    Learnable weighted combination of ALL transformer layers.
    uses all hidden states, not just the last one.
    """
    def __init__(self, num_hidden_layers, layer_start=4, layer_weights=None):
        super().__init__()
        self.layer_start = layer_start
        self.num_layers = num_hidden_layers - layer_start + 1
        # learnable weights for each layer
        self.layer_weights = nn.Parameter(
            torch.ones(self.num_layers) / self.num_layers
        )

    def forward(self, all_hidden_states, attention_mask):
        # all_hidden_states: tuple of (batch, seq_len, hidden) for each layer
        # stack layers from layer_start onwards
        layers = torch.stack(all_hidden_states[self.layer_start:], dim=0)  # (num_layers, batch, seq, hidden)
        
        # normalize weights with softmax for stable combination
        weights = F.softmax(self.layer_weights, dim=0)
        
        # weighted sum: (num_layers, batch, seq, hidden) -> (batch, seq, hidden)
        weighted = (weights.view(-1, 1, 1, 1) * layers).sum(dim=0)
        
        # mean pool the weighted representation
        mask = attention_mask.unsqueeze(-1).float()
        pooled = (weighted * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return pooled


class MultiHeadAttentionPooling(nn.Module):
    """
    Multi-head attention pooling - multiple learned query vectors
    attend to the sequence, giving diverse views of the input.
    """
    def __init__(self, hidden_size, num_heads=8, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_size // num_heads
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by num_heads"
        
        # learnable query vectors (one per head)
        self.queries = nn.Parameter(torch.randn(num_heads, self.head_dim) * 0.02)
        
        # project hidden states to keys
        self.key_proj = nn.Linear(hidden_size, hidden_size)
        self.value_proj = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.orthogonal_(self.key_proj.weight)
        nn.init.orthogonal_(self.value_proj.weight)
        nn.init.orthogonal_(self.out_proj.weight)
        nn.init.zeros_(self.key_proj.bias)
        nn.init.zeros_(self.value_proj.bias)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, hidden_states, attention_mask):
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # project to keys and values
        keys = self.key_proj(hidden_states)  # (batch, seq, hidden)
        values = self.value_proj(hidden_states)  # (batch, seq, hidden)
        
        # reshape for multi-head attention
        keys = keys.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        values = values.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch, heads, seq, head_dim)
        
        # expand queries for batch: (heads, head_dim) -> (batch, heads, 1, head_dim)
        queries = self.queries.unsqueeze(0).unsqueeze(2).expand(batch_size, -1, -1, -1)
        
        # attention scores: (batch, heads, 1, head_dim) @ (batch, heads, head_dim, seq) -> (batch, heads, 1, seq)
        attn_scores = torch.matmul(queries, keys.transpose(-2, -1)) * self.scale
        
        # apply mask
        mask = attention_mask.unsqueeze(1).unsqueeze(2)  # (batch, 1, 1, seq)
        attn_scores = attn_scores.masked_fill(mask == 0, -1e4)
        
        # softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # weighted sum: (batch, heads, 1, seq) @ (batch, heads, seq, head_dim) -> (batch, heads, 1, head_dim)
        context = torch.matmul(attn_weights, values)
        
        # reshape back: (batch, heads, 1, head_dim) -> (batch, hidden)
        context = context.squeeze(2).view(batch_size, hidden_size)
        
        # output projection
        output = self.out_proj(context)
        return output


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism to learn optimal combination of multiple representations.
    Uses sigmoid gates to control contribution of each input.
    """
    def __init__(self, hidden_size, num_inputs):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_size * num_inputs, hidden_size * num_inputs),
            nn.LayerNorm(hidden_size * num_inputs),
            nn.GELU(),
            nn.Linear(hidden_size * num_inputs, num_inputs),
            nn.Sigmoid()
        )
        self.fusion = nn.Sequential(
            nn.Linear(hidden_size * num_inputs, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, *inputs):
        # inputs: tuple of (batch, hidden) tensors
        concat = torch.cat(inputs, dim=-1)  # (batch, hidden * num_inputs)
        
        # compute gates for each input
        gates = self.gate(concat)  # (batch, num_inputs)
        
        # apply gates and sum
        hidden_size = inputs[0].shape[-1]
        gated = torch.zeros_like(inputs[0])
        for i, inp in enumerate(inputs):
            gated = gated + gates[:, i:i+1] * inp
        
        # final fusion
        output = self.fusion(concat) + gated  # residual connection
        return output


class MegaPooling(nn.Module):
    """Just mean pooling for now"""
    def __init__(self, hidden_size, num_hidden_layers, num_attention_heads=8, dropout=0.1):
        super().__init__()
        # nothing needed for mean pooling
        
    def forward(self, last_hidden_state, all_hidden_states, attention_mask):
        # simple mean pooling
        mask = attention_mask.unsqueeze(-1).float()
        output = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        return output


class MultiSampleDropout(nn.Module):
    """Apply multiple dropout masks with varying rates and average predictions"""
    def __init__(self, hidden_size, num_classes, num_dropouts=5, dropout_rate=0.1):
        super().__init__()
        # varying dropout rates for more diverse predictions
        rates = [dropout_rate + i * 0.05 for i in range(num_dropouts)]  # e.g., [0.1, 0.15, 0.2, 0.25, 0.3]
        self.dropouts = nn.ModuleList([nn.Dropout(r) for r in rates])
        self.classifier = nn.Linear(hidden_size, num_classes)
        self._init_weights()

    def _init_weights(self):
        nn.init.orthogonal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x):
        if self.training:
            # average logits from multiple dropout masks
            logits = torch.stack([self.classifier(drop(x)) for drop in self.dropouts], dim=0)
            return logits.mean(dim=0)
        else:
            return self.classifier(x)


class MedicalClassifier(nn.Module):
    def __init__(self, model_name, num_classes, num_dropouts=5, dropout_rate=0.1):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.encoder.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        
        hidden_size = self.encoder.config.hidden_size
        num_hidden_layers = self.encoder.config.num_hidden_layers

        # MEGA POOLING - the ultimate pooling strategy
        self.mega_pool = MegaPooling(
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=8,
            dropout=dropout_rate
        )

        # classification head
        self.pre_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )

        # multi-sample dropout classifier
        self.classifier = MultiSampleDropout(hidden_size, num_classes, num_dropouts, dropout_rate)

        self._init_weights()

    def _init_weights(self):
        for module in self.pre_classifier:
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            output_hidden_states=True  # need all hidden states for weighted layer pooling
        )
        
        last_hidden_state = outputs.last_hidden_state
        all_hidden_states = outputs.hidden_states  # tuple of all layer outputs

        # MEGA pooling
        pooled = self.mega_pool(last_hidden_state, all_hidden_states, attention_mask)

        # classification head
        x = self.pre_classifier(pooled)
        logits = self.classifier(x)

        loss = None
        if labels is not None:
            # Focal Loss
            ce_loss = F.cross_entropy(logits, labels, reduction='none', label_smoothing=LABEL_SMOOTHING)
            pt = torch.exp(-ce_loss)
            loss = (((1 - pt) ** FOCAL_GAMMA) * ce_loss).mean()

        return {"loss": loss, "logits": logits}
