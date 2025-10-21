
# -*- coding: utf-8 -*-
"""
hon_ensemble.py
- GPT2_MoE_Hybrid (from hon_modeling_1.py) + MultiHeadLatentAttention (from hon_modeling_2.py)
- Single-file training/inference + weighted ensemble generation.
- Usage (examples):
  python hon_ensemble.py --alpha 0.6 --max_new_tokens 300
  python hon_ensemble.py --train_steps 2000 --alpha 0.5 --max_new_tokens 500
Notes:
- If "input.txt" is present in the working directory, will use it (char-level) for training.
- If not present, it will skip training and just run ensemble generation from randomly initialized weights.
"""

import os
import argparse
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Device
# -------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =========================================================
# Part A. MultiHeadLatentAttention (from hon_modeling_2.py)
# =========================================================
class MultiHeadLatentAttention(nn.Module):
    """
    MLA의 핵심 원리를 담은 개념적 구현.
    - 고정된 크기의 잠재 KV 캐시를 유지.
    - 순차적인 업데이트 메커니즘을 통해 캐시를 갱신.
    """
    def __init__(self, n_head, n_embd):
        super().__init__()
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head

        # 1. 일반적인 어텐션 프로젝션 레이어
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, n_embd) # 새 토큰 -> Key 후보
        self.v_proj = nn.Linear(n_embd, n_embd) # 새 토큰 -> Value 후보
        self.c_proj = nn.Linear(n_embd, n_embd)

        # 2. 잠재 KV 캐시 업데이트를 위한 게이트 메커니즘
        self.update_gate_proj = nn.Linear(n_embd + n_embd, n_embd) # z_t (Update Gate)

    def forward(self, x):
        B, T, C = x.size() # Batch, Time, Channels

        # 잠재 KV 캐시 초기화 (고정된 크기의 단일 벡터)
        latent_kv = torch.zeros(B, self.n_head, 1, self.head_size, device=x.device)
        
        outputs = []
        for t in range(T):
            xt = x[:, t:t+1, :] # (B, 1, C)
            
            # --- 1. 잠재 KV 캐시 업데이트 ---
            new_k = self.k_proj(xt).view(B, 1, self.n_head, self.head_size).transpose(1, 2) # (B, n_head, 1, hs)
            new_v = self.v_proj(xt).view(B, 1, self.n_head, self.head_size).transpose(1, 2) # (B, n_head, 1, hs)
            new_kv_candidate = new_v

            gate_input = torch.cat([latent_kv.squeeze(2), new_kv_candidate.squeeze(2)], dim=-1)
            z = torch.sigmoid(self.update_gate_proj(gate_input)).view(B, self.n_head, 1, self.head_size)
            
            latent_kv = (1 - z) * latent_kv + z * new_kv_candidate

            # --- 2. 어텐션 계산 ---
            q = self.q_proj(xt).view(B, 1, self.n_head, self.head_size).transpose(1, 2) # (B, n_head, 1, hs)
            att = (q @ latent_kv.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_size))
            att = F.softmax(att, dim=-1)
            y = att @ latent_kv # (B, n_head, 1, hs)
            y = y.transpose(1, 2).contiguous().view(B, 1, C)
            outputs.append(y)
        
        y = torch.cat(outputs, dim=1)
        y = self.c_proj(y)
        return y


# ====================================================================
# Part B. GPT2_MoE_Hybrid and its components (from hon_modeling_1.py)
# ====================================================================
class SimplifiedLatentAttention(nn.Module):
    """
    KV 캐시를 압축하는 개념을 단순화하여 구현한 어텐션.
    k, v는 latent_dim 공간으로 투영.
    """
    def __init__(self, n_head, n_embd, block_size, latent_dim):
        super().__init__()
        assert latent_dim % n_head == 0, "latent_dim must be divisible by n_head"
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        self.latent_head_size = latent_dim // n_head

        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, latent_dim) # Key -> latent
        self.v_proj = nn.Linear(n_embd, latent_dim) # Value -> latent
        
        self.c_proj = nn.Linear(n_embd, n_embd)

        self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()
        q = self.q_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.latent_head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.latent_head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd)
        return self.c_proj(y)


class Expert(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
        )
    def forward(self, x):
        return self.net(x)


class MoE_Layer(nn.Module):
    """
    Multiple experts; route top-k per token.
    """
    def __init__(self, n_embd, num_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])
        self.gate = nn.Linear(n_embd, num_experts) # router
        self.top_k = top_k
        self.num_experts = num_experts  # (fix) needed in forward loop

    def forward(self, x):
        B, T, C = x.size()
        N = B * T
        x_flat = x.view(N, C)

        router_logits = self.gate(x_flat)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1) # (N, top_k)
        routing_weights = F.softmax(routing_weights, dim=-1)

        final_output = torch.zeros_like(x_flat)
        flat_indices = torch.arange(N, device=x.device).unsqueeze(1).expand(-1, self.top_k) # (N, top_k)

        for i in range(self.num_experts):
            expert_mask = (selected_experts == i)  # (N, top_k) boolean
            if expert_mask.any():
                idxs = flat_indices[expert_mask]           # 1D indices for tokens routed to expert i
                w = routing_weights[expert_mask]           # 1D weights
                x_i = x_flat.index_select(0, idxs)         # (M, C)
                out_i = self.experts[i](x_i)               # (M, C)
                final_output.index_add_(0, idxs, out_i * w.unsqueeze(1))

        return final_output.view(B, T, C)


class HybridBlock(nn.Module):
    def __init__(self, n_embd, n_head, block_size, latent_dim, num_experts, top_k):
        super().__init__()
        self.ln_1 = nn.LayerNorm(n_embd)
        self.attn = SimplifiedLatentAttention(n_head, n_embd, block_size, latent_dim)
        self.ln_2 = nn.LayerNorm(n_embd)
        self.moe = MoE_Layer(n_embd, num_experts, top_k)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.moe(self.ln_2(x))
        return x


class GPT2_MoE_Hybrid(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, latent_dim, num_experts, top_k):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[HybridBlock(n_embd, n_head, block_size, latent_dim, num_experts, top_k) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ===================================================
# Part C. Wrapper LM using MultiHeadLatentAttention
# ===================================================
class LatentLMWrapper(nn.Module):
    """
    MultiHeadLatentAttention 블록에 LM 헤드를 붙인 간단 래퍼.
    """
    def __init__(self, vocab_size, n_embd, n_head, block_size):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.attn = MultiHeadLatentAttention(n_head, n_embd)
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        tok_emb = self.token_embedding(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding(pos)
        x = tok_emb + pos_emb
        x = self.attn(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits = self(idx_cond)[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# ==============================================
# Part D. Data utilities (char-level tokenizer)
# ==============================================
def build_char_dataset(path, block_size):
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    chars = sorted(list(set(text)))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    encode = lambda s: [stoi[c] for c in s]
    decode = lambda l: "".join([itos[i] for i in l])
    data = torch.tensor(encode(text), dtype=torch.long)
    n = int(0.9 * len(data))
    train_data = data[:n]
    val_data = data[n:]
    vocab_size = len(chars)

    def get_batch(split, batch_size):
        src = train_data if split == "train" else val_data
        ix = torch.randint(len(src) - block_size, (batch_size,))
        x = torch.stack([src[i:i+block_size] for i in ix])
        y = torch.stack([src[i+1:i+block_size+1] for i in ix])
        return x.to(device), y.to(device)

    return vocab_size, decode, get_batch


# ==============================================
# Part E. Training & Ensemble Inference
# ==============================================
def train_language_model(model, get_batch, steps, lr=1e-3, eval_interval=500, batch_size=32):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    for it in range(steps):
        if eval_interval and (it % eval_interval == 0):
            model.eval()
            with torch.no_grad():
                losses = []
                for _ in range(50):
                    X, Y = get_batch("val", batch_size)
                    logits = model(X)
                    B, T, C = logits.shape
                    loss = F.cross_entropy(logits.view(B*T, C), Y.view(B*T))
                    losses.append(loss.item())
            print(f"[eval] step {it} | val loss {sum(losses)/len(losses):.4f}")
            model.train()

        X, Y = get_batch("train", batch_size)
        logits = model(X)
        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B*T, C), Y.view(B*T))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if (it+1) % 100 == 0:
            print(f"[train] step {it+1} | loss {loss.item():.4f}")
    return model


@torch.no_grad()
def ensemble_generate(model1, model2, start_idx, max_new_tokens, alpha=0.5, block_size=64):
    """
    Weighted probability ensemble: probs = alpha*p1 + (1-alpha)*p2
    """
    idx = start_idx.clone().to(device)
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -block_size:]

        logits1 = model1(idx_cond)[:, -1, :]
        logits2 = model2(idx_cond)[:, -1, :]

        p1 = F.softmax(logits1, dim=-1)
        p2 = F.softmax(logits2, dim=-1)
        probs = alpha * p1 + (1.0 - alpha) * p2

        idx_next = torch.multinomial(probs, num_samples=1)
        idx = torch.cat((idx, idx_next), dim=1)
    return idx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--block_size", type=int, default=64)
    parser.add_argument("--n_embd", type=int, default=128)
    parser.add_argument("--n_head", type=int, default=4)
    parser.add_argument("--n_layer", type=int, default=4)
    parser.add_argument("--latent_dim", type=int, default=128)
    parser.add_argument("--num_experts", type=int, default=8)
    parser.add_argument("--top_k", type=int, default=2)
    parser.add_argument("--train_steps", type=int, default=0, help="set >0 to train on input.txt if available")
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=0.5, help="weight for GPT2_MoE_Hybrid in the ensemble")
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Try to build dataset from input.txt
    data_ok = os.path.exists("input.txt")
    if data_ok:
        print(">> Using 'input.txt' for char-level training.")
        vocab_size, decode, get_batch = build_char_dataset("input.txt", args.block_size)
    else:
        print(">> 'input.txt' not found. Skipping training and using dummy vocab_size=50257.")
        vocab_size = 50257
        decode = None
        get_batch = None

    # Build models
    model_gpt = GPT2_MoE_Hybrid(
        vocab_size=vocab_size, n_embd=args.n_embd, n_head=args.n_head, n_layer=args.n_layer,
        block_size=args.block_size, latent_dim=args.latent_dim, num_experts=args.num_experts, top_k=args.top_k
    ).to(device)

    model_latent = LatentLMWrapper(
        vocab_size=vocab_size, n_embd=args.n_embd, n_head=args.n_head, block_size=args.block_size
    ).to(device)

    # Optionally train both models (very simple loop, same data loader)
    if data_ok and args.train_steps > 0:
        print(">> Training GPT2_MoE_Hybrid ...")
        train_language_model(model_gpt, get_batch, steps=args.train_steps, lr=args.lr,
                             eval_interval=args.eval_interval, batch_size=args.batch_size)
        print(">> Training LatentLMWrapper ...")
        train_language_model(model_latent, get_batch, steps=args.train_steps, lr=args.lr,
                             eval_interval=args.eval_interval, batch_size=args.batch_size)
    else:
        print(">> Training skipped. Models remain randomly initialized.")

    # Ensemble generation
    start_idx = torch.zeros((1, 1), dtype=torch.long, device=device)  # BOS=0 for char-level
    out = ensemble_generate(model_gpt, model_latent, start_idx,
                            max_new_tokens=args.max_new_tokens, alpha=args.alpha, block_size=args.block_size)
    out_list = out[0].tolist()

    if decode is not None:
        txt = decode(out_list)
        print("\n=== Ensemble Generated Text ===")
        print(txt)
    else:
        print("\n=== Ensemble Generated Token IDs ===")
        print(out_list[:100], "... (truncated)")

if __name__ == "__main__":
    main()
