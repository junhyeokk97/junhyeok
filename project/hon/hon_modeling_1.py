import torch
import torch.nn as nn
from torch.nn import functional as F

# GPU 사용 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 하이퍼파라미터 ---
vocab_size = 50257  # 이 값은 실제 데이터에 따라 바뀔 예정
n_embd = 128      # 임베딩 차원
n_layer = 4       # 블록 수
n_head = 4        # 헤드 수
block_size = 64   # 최대 시퀀스 길이
latent_dim = 128  # 어텐션 잠재 공간 차원 (n_embd와 같거나 작아야 함)
num_experts = 8   # 총 전문가 수
top_k = 2         # 활성화할 전문가 수
learning_rate = 1e-3
max_iters = 5000
eval_interval = 500
batch_size = 32

# --- 제공해주신 모델 코드 (수정 없음) ---
# ... (여기에 제공해주신 SimplifiedLatentAttention, Expert, MoE_Layer, HybridBlock 클래스를 그대로 붙여넣습니다) ...
# --- 1. 딥시크의 아이디어를 차용한 어텐션 모듈 ---
class SimplifiedLatentAttention(nn.Module):
    """
    KV 캐시를 압축하는 개념을 단순화하여 구현한 어텐션.
    (수정) k,v의 head_size가 latent_dim을 기반으로 계산되도록 수정했습니다.
    """
    def __init__(self, n_head, n_embd, block_size, latent_dim):
        super().__init__()
        assert latent_dim % n_head == 0
        self.n_head = n_head
        self.n_embd = n_embd
        self.head_size = n_embd // n_head
        self.latent_head_size = latent_dim // n_head

        # Q, K, V를 위한 선형 레이어
        self.q_proj = nn.Linear(n_embd, n_embd)
        self.k_proj = nn.Linear(n_embd, latent_dim) # Key를 잠재 공간으로 압축
        self.v_proj = nn.Linear(n_embd, latent_dim) # Value를 잠재 공간으로 압축
        
        # 출력을 위한 선형 레이어
        self.c_proj = nn.Linear(n_embd, n_embd)

        # 마스크
        self.register_buffer('bias', torch.tril(torch.ones(block_size, block_size))
                                             .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.size()

        q = self.q_proj(x).view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_head, self.latent_head_size).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_head, self.latent_head_size).transpose(1, 2)
        
        # 어텐션 계산
        att = (q @ k.transpose(-2, -1)) * (1.0 / k.size(-1)**0.5)
        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_embd) # C 대신 n_embd로 명시
        
        return self.c_proj(y)


# --- 2. 딥시크의 핵심 아이디어인 MoE 레이어 ---
class Expert(nn.Module):
    """하나의 전문가 모듈 (기본적인 피드포워드 네트워크)"""
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
    여러 전문가 중 일부를 선택하여 계산하는 MoE 레이어.
    FFN을 대체합니다.
    """
    def __init__(self, n_embd, num_experts, top_k):
        super().__init__()
        self.experts = nn.ModuleList([Expert(n_embd) for _ in range(num_experts)])
        self.gate = nn.Linear(n_embd, num_experts) # 라우터(게이트)
        self.top_k = top_k

    def forward(self, x):
        B, T, C = x.size()
        x_flat = x.view(-1, C) # (B*T, C)
        
        router_logits = self.gate(x_flat)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)
        routing_weights = F.softmax(routing_weights, dim=-1)
        
        final_output = torch.zeros_like(x_flat)
        flat_indices = torch.arange(x_flat.size(0), device=x.device).unsqueeze(1).expand(-1, self.top_k)

        for i in range(self.num_experts):
            expert_mask = (selected_experts == i)
            if expert_mask.any():
                token_indices = flat_indices[expert_mask]
                weights = routing_weights[expert_mask]
                expert_output = self.experts[i](x_flat[token_indices])
                final_output.index_add_(0, token_indices, expert_output * weights.unsqueeze(1))
                
        return final_output.view(B, T, C)

# --- 3. 위 모듈들을 결합한 하이브리드 블록 ---
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

# --- 4. 최종 모델 조립 (+ 예측 함수 추가) ---
class GPT2_MoE_Hybrid(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, latent_dim, num_experts, top_k):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, n_embd)
        self.position_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(
            *[HybridBlock(n_embd, n_head, block_size, latent_dim, num_experts, top_k) for _ in range(n_layer)]
        )
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx):
        B, T = idx.size()
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

    # === 예측(텍스트 생성) 함수 ===
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            # 입력 컨텍스트는 block_size를 넘지 않도록 자름
            idx_cond = idx[:, -self.block_size:]
            # 예측
            logits = self(idx_cond)
            # 마지막 time step의 logit만 사용
            logits = logits[:, -1, :]
            # 소프트맥스를 통해 확률로 변환
            probs = F.softmax(logits, dim=-1)
            # 확률 분포에 따라 다음 토큰 샘플링
            idx_next = torch.multinomial(probs, num_samples=1)
            # 기존 시퀀스에 새로운 토큰 추가
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# ======================================================================
# ========= 여기서부터 학습과 예측을 위한 새로운 코드입니다 =========
# ======================================================================

# --- Step 1: 데이터 및 토크나이저 준비 ---
# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# 문자 단위 토크나이저
chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: ''.join([itos[i] for i in l])

# 전체 데이터를 텐서로 변환
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# 데이터 로더 (배치 생성)
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

# --- Step 2: 학습 코드 ---
# 모델 생성 (vocab_size를 실제 데이터 기준으로 다시 설정)
model = GPT2_MoE_Hybrid(vocab_size, n_embd, n_head, n_layer, block_size, latent_dim, num_experts, top_k)
m = model.to(device)

# 옵티마이저 생성
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# 학습 루프
for iter in range(max_iters):
    if iter % eval_interval == 0:
        model.eval()
        losses = torch.zeros(200)
        for k in range(200):
            X, Y = get_batch('val')
            logits = model(X)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            Y = Y.view(B*T)
            loss = F.cross_entropy(logits, Y)
            losses[k] = loss.item()
        print(f"step {iter}: val loss {losses.mean():.4f}")
        model.train()

    # 학습 데이터 배치 가져오기
    xb, yb = get_batch('train')

    # 순전파 및 손실 계산
    logits = model(xb)
    B, T, C = logits.shape
    logits = logits.view(B*T, C)
    yb = yb.view(B*T)
    loss = F.cross_entropy(logits, yb)

    # 역전파 및 파라미터 업데이트
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(f"Final training loss: {loss.item():.4f}")


# --- Step 3: 예측(생성) 코드 실행 ---
print("\n--- 텍스트 생성 시작 ---")
# 시작 컨텍스트 (빈 줄 하나)
start_context = torch.zeros((1, 1), dtype=torch.long, device=device)
# 모델을 사용해 500개의 토큰 생성
generated_tokens = m.generate(start_context, max_new_tokens=500)[0].tolist()
# 생성된 토큰을 텍스트로 디코딩하여 출력
print(decode(generated_tokens))
print("--- 텍스트 생성 완료 ---")