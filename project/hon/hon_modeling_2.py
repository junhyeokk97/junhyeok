import torch
import torch.nn as nn
from torch.nn import functional as F

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
        # 기존 잠재 캐시와 새 토큰 정보를 합쳐 게이트를 계산
        self.update_gate_proj = nn.Linear(n_embd + n_embd, n_embd) # z_t (Update Gate)

    def forward(self, x):
        B, T, C = x.size() # Batch, Time, Channels

        # 잠재 KV 캐시 초기화 (고정된 크기의 단일 벡터)
        # (B, n_head, 1, head_size) -> 1은 시퀀스 길이가 1임을 의미
        latent_kv = torch.zeros(B, self.n_head, 1, self.head_size, device=x.device)
        
        outputs = []
        # 순차적으로 각 타임스텝(단어)을 처리
        for t in range(T):
            xt = x[:, t:t+1, :] # 현재 타임스텝의 입력 (B, 1, C)
            
            # --- 1. 잠재 KV 캐시 업데이트 ---
            # 새로운 토큰으로부터 Key, Value 후보 생성
            new_k = self.k_proj(xt).view(B, 1, self.n_head, self.head_size).transpose(1, 2) # (B, n_head, 1, hs)
            new_v = self.v_proj(xt).view(B, 1, self.n_head, self.head_size).transpose(1, 2) # (B, n_head, 1, hs)
            # 여기서는 K와 V를 동일하게 취급하여 하나의 잠재 캐시로 압축
            new_kv_candidate = new_v # 간단하게 Value 후보를 사용

            # 업데이트 게이트(z_t) 계산: 기존 정보와 새 정보 중 무엇을 얼마나 반영할지 결정
            # (B, n_head, head_size) 와 (B, n_head, head_size)를 합쳐서 계산
            gate_input = torch.cat([latent_kv.squeeze(2), new_kv_candidate.squeeze(2)], dim=-1)
            z = torch.sigmoid(self.update_gate_proj(gate_input)).view(B, self.n_head, 1, self.head_size)
            
            # 게이트를 사용하여 잠재 KV 캐시 업데이트
            # z가 1에 가까우면 새 정보를 많이, 0에 가까우면 기존 정보를 많이 유지
            latent_kv = (1 - z) * latent_kv + z * new_kv_candidate

            # --- 2. 어텐션 계산 ---
            # 현재 토큰의 Query는 '잠재 KV 캐시'와 상호작용
            q = self.q_proj(xt).view(B, 1, self.n_head, self.head_size).transpose(1, 2) # (B, n_head, 1, hs)
            
            # Query가 단 하나의 '잠재 KV 캐시'를 참조
            att = (q @ latent_kv.transpose(-2, -1)) * (1.0 / latent_kv.size(-1)**0.5)
            att = F.softmax(att, dim=-1)
            
            # 현재 타임스텝의 출력 계산
            y = att @ latent_kv # (B, n_head, 1, hs)
            y = y.transpose(1, 2).contiguous().view(B, 1, C)
            
            outputs.append(y)
        
        # 모든 타임스텝의 출력을 하나로 합침
        y = torch.cat(outputs, dim=1)
        # 최종 프로젝션
        y = self.c_proj(y)
        
        return y