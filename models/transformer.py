import torch
import torch.nn as nn
from math import sqrt

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.dim = dim
        self.eps = eps

        self.g = nn.Parameter(torch.ones((self.dim)))
        
    def forward(self, x):
        r_x = torch.clamp(torch.norm(x, dim=-1, keepdim=True) * self.dim ** - 0.5, min=self.eps)
        output = (x / r_x) * self.g

        return output

class SwiGLU(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.a_proj = nn.Linear(self.in_dim, 2 * self.out_dim, bias=True)
        self.silu = nn.SiLU()

    def forward(self, x):
        # Input x: [B, N, D]
        x = self.a_proj(x)
        A, G = torch.chunk(x, dim=-1, chunks=2)
        output = A * self.silu(G)

        return output

class ExpertMLP(nn.Module):
    def __init__(self, hidden_dim=1024, output_dim=4096, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.out_dim = output_dim
        self.p = dropout # dropout probability

        # Define SwiGLU activation
        self.swiglu = SwiGLU(in_dim=self.hidden_dim, out_dim=self.out_dim)

        # Dropout
        self.dropout = nn.Dropout(p=self.p)

        # Linear proj
        self.out_proj = nn.Linear(self.out_dim, self.hidden_dim, bias=False)

    def forward(self, x):
        x = self.out_proj(self.dropout(self.swiglu(x)))
        return x

class NoiseRouter(nn.Module):
    def __init__(self, hidden_dim=1024, num_experts=4, top_k=2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_experts = num_experts
        self.top_k = top_k

        # Linear projections
        self.ffn_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 2)
        self.gelu = nn.GELU()

        # Expert logits projection
        self.log_proj = nn.Linear(self.hidden_dim * 2, num_experts)

        # Softmax
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, tokens, noise_embed):
        # noise_embed: [B, 1, D]
        B, N, D = tokens.shape[0], tokens.shape[1], tokens.shape[2]
        x = self.gelu(self.ffn_proj(noise_embed.expand(B, N, D)))

        # binary routing mask 
        mask = torch.zeros((B, N, self.num_experts))

        # Get expert probs
        expert_probs = self.softmax(self.log_proj(x))

        # Choose experts
        if self.training:
            # Stochastic sampling without replacement
            expert_indices = torch.multinomial(expert_probs.reshape(-1, self.num_experts), num_samples=2, replacement=False).reshape(B, N, self.top_k)
        else:
            # Deterministic top-k selection
            _, expert_indices = torch.topk(expert_probs, self.top_k, dim=-1)
            
        router_mask = torch.scatter(mask, 2, expert_indices, 1.0)
        router_weights = expert_probs * router_mask
        means = torch.sum(router_weights, dim=-1, keepdim=True)

        return expert_probs, expert_indices, router_mask, router_weights / means

class SparseExpertLayer(nn.Module):
    def __init__(self, num_experts=4, hidden_dim=1024, output_dim=4096, dropout=0.1, top_k=2):
        super().__init__()

        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.top_k = top_k
        self.experts = nn.ModuleList()

        for _ in range(num_experts):
            self.experts.append(ExpertMLP(hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout))

    def forward(self, x, router_weights, router_mask):
        # Input Tokens of shape [B, N, D]
        # Router mask is shape [B, N, E]
        # Router weights is shape [B, N, E]
        # E is number of experts
        B, N, D = x.shape[0], x.shape[1], x.shape[2]

        M = torch.where(router_mask == 1.0, True, False).reshape(B * N, 4)
        x = x.reshape(B * N, self.hidden_dim)
        W  = router_weights.reshape(B * N, self.num_experts)

        total_assignments = 0

        # To write values using scatter from each yi.
        z = torch.zeros((B * N, D))
        
        for i in range(self.num_experts):
            # Get a binary mask for expert i
            m_i = M[:, i]

            # Get tokens assigned to expert i
            x_i = x[m_i]

            # Determine if total assignments is correct
            total_assignments += x_i.shape[0]

            # Get the output of expert i on the tokens assigned to it.
            o_i = self.experts[i](x_i)

            assert o_i.shape == x_i.shape

            # Get the weights for expert i and multiply them by expert i outputs
            w_i = W[m_i, i]
            y_i = o_i * w_i.unsqueeze(dim=1)

            assert y_i.shape == x_i.shape

            indices = torch.nonzero(m_i.to(torch.int32))

            print(indices.shape)
            print(y_i.shape)

            # Aggregate expert outputs
            z = z.index_add_(dim=0, index=indices.squeeze(1), source=y_i)
            
        assert total_assignments == B * N * self.top_k

        # Note aggregated output is of shape [B, N, 1024]
        return z.reshape(B, N, D)

class ActionTransformer(nn.Module):
    def __init__(self, fl=512, fv=768, num_heads=8, action_dim=7, horizon=10, hidden_dim=1024, num_blocks=12, causal=True, proprio_dim=15, top_k = 2, num_experts = 4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.horizon = horizon
        self.num_blocks = num_blocks
        self.action_dim = action_dim
        self.causal = causal
        self.fl = fl # language embedding size
        self.fv = fv # vision embedding size for gripper and static images
        self.p = proprio_dim
        self.top_k = top_k
        self.num_experts = num_experts
        self.dropout = dropout


        self.Transformer = Transformer(self.num_heads, self.hidden_dim, self.num_blocks, self.causal, self.top_k, self.num_experts, self.dropout)
        self.action_embedding = nn.Linear(self.action_dim, self.hidden_dim, bias=False)
        self.action_head = nn.Linear(self.hidden_dim, self.action_dim)
        self.position_embeddings = nn.Parameter(torch.zeros((1, self.horizon + 1, self.hidden_dim)))

        # Condition on noise so the network knows how much denoising is required per noise level.
        self.noise_proj = nn.Sequential(
            nn.Linear(1, self.hidden_dim, bias=True),
            nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)
        )

        # Projections for observation 
        self.s_proj = nn.Linear(self.hidden_dim * 2, self.hidden_dim, bias=False)
        self.l_proj = nn.Linear(self.fl, self.hidden_dim, bias=False)

    def forward(self, x, noise_level, s, g, l):
        # Linearly project actions and add positional embeddings to convert into Transformer tokens
        x = self.action_embedding(x)
        action_embed = x + self.position_embeddings[:, 1:11, :]

        # Bernoulli mask (training only)
        if self.training:
            m = torch.empty_like(l)
            m.bernoulli_(p=0.9)
            l = l * m

        # Projections for Observation Conditioning
        # Note that static and gripper camera embeddings come from the Vision Encoder, and language embeddings come
        # from the Language Encoder.
        # Proprioception is [B, 15], language embedding is [B, Fl], Gripper-camera embedding is [B, Fv]
        # # and Static-camera embedding is [B, Fv]
        # all are dim [B, 128] ==> [B, 1, 128]
        s_embed = self.s_proj(s).unsqueeze(1) + self.position_embeddings[:, 1:2, :]
        g_embed = self.s_proj(g).unsqueeze(1) + self.position_embeddings[:, 1:2, :]
        l_embed = self.l_proj(l).unsqueeze(1) + self.position_embeddings[:, 0:1, :]

        # Noise conditioning
        noise_level_transformed = (torch.log(noise_level) / 4.0).unsqueeze(1) # shape is [B, 1]
        noise_embed = self.noise_proj(noise_level_transformed).unsqueeze(dim=1) # shape is [B, 1, 128]

        # Concatenate noise tokekn with action and observation tokens along sequence length.
        # Prepend observation conditioning tokens to action tokens
        input_tokens = torch.cat([noise_embed, l_embed, s_embed, g_embed, action_embed], dim=1)

        # Pass into Transformer architecture.
        output_tokens = self.Transformer(input_tokens, noise_embed)

        # Action head
        actions = self.action_head(output_tokens[:, 4:self.horizon + 4, :])

        return actions

class Transformer(nn.Module):
    def __init__(self, num_heads=4, hidden_dim=128, num_blocks=2, causal=True, top_k=2, num_experts=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_blocks = num_blocks
        self.causal=causal
        self.top_k = top_k
        self.num_experts = num_experts
        self.dropout = dropout

        self.TransformerBlocks = nn.ModuleList()

        for _ in range(num_blocks):
            self.TransformerBlocks.append(
                TransformerBlock(num_heads=self.num_heads, hidden_dim=self.hidden_dim, causal=self.causal, top_k = self.top_k, num_experts = self.num_experts, dropout = self.dropout)
            )

        self.rms_norm = RMSNorm(dim=self.hidden_dim, eps=1e-6)

    def forward(self, x, noise_level):
        for TransformerBlock in self.TransformerBlocks:
            x = TransformerBlock(x, noise_level)

        y = self.rms_norm(x)
        return y

class TransformerBlock(nn.Module):
    def __init__(self, num_heads=4, hidden_dim=128, causal=True, top_k=2, num_experts=4, dropout=0.1):
        super().__init__()
        self.h = num_heads
        self.hidden_dim = hidden_dim
        self.causal = causal
        self.top_k = top_k
        self.num_experts = num_experts
        self.dropout = dropout

        # Multi-Head Self-Attention
        self.mha = MultiHeadSelfAttention(hidden_dim=self.hidden_dim, attn_heads=self.h, causal=self.causal)

        # RMS Norm
        self.rms_norm1 = RMSNorm(dim=self.hidden_dim, eps=1e-6)
        self.rms_norm2 = RMSNorm(dim=self.hidden_dim, eps=1e-6)

        # Noise Router
        self.noise_router = NoiseRouter(
            hidden_dim = self.hidden_dim,
            num_experts = self.num_experts,
            top_k = self.top_k
        )

        # Sparse Expert Layer
        self.experts = SparseExpertLayer(
            num_experts=self.num_experts,
            hidden_dim=self.hidden_dim,
            output_dim=self.hidden_dim * 4,
            dropout = self.dropout,
            top_k = self.top_k
        )

    def forward(self, x, noise_embed):
        # X: [B, N, D]
        # Noise-conditioned Self-Attention
        H = self.mha(self.rms_norm1(x) + noise_embed) + x
        U = self.rms_norm2(H)

        # Mixture-of-Experts
        _, _, router_mask, router_weights = self.noise_router(U, noise_embed)
        Y = U + self.experts(U, router_weights, router_mask)

        return Y

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_dim=128, attn_heads=4, causal=True, device="cpu", attn_dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.norm_dim = hidden_dim // attn_heads # normalization dimensions
        self.h = attn_heads
        self.attn_dropout = attn_dropout

        # Define the QKV projection
        self.qkv_proj = nn.Linear(self.hidden_dim, self.hidden_dim * 3)

        # Activation function
        self.softmax = nn.Softmax(dim=-1)

        # Causal attention
        self.causal = causal

        # Output projection
        self.out_proj = nn.Linear(self.hidden_dim, self.hidden_dim, bias=False)

        # RMS Norm
        self.rmsnorm_q = RMSNorm(dim=self.norm_dim, eps=1e-6)
        self.rmsnorm_k = RMSNorm(dim=self.norm_dim, eps=1e-6)

        # Dropout
        self.attn_dropout = nn.Dropout(p=self.attn_dropout)

        # Device
        self.device = device

    def forward(self, x):
        """
        Perform Multi-Head Self-Atttention on x. 
        """

        B, N, D = x.shape[0], x.shape[1], x.shape[2]

        # Attention Causal Masking
        if self.causal:
            # zeros below and on the diagonal, - infinity above.
            mask = torch.full((1, 1, N, N), fill_value=float("-inf"), device=self.device)
            mask = torch.triu(mask, diagonal=1)
        else:
            mask = torch.zeros((1, 1, N, N), device=self.device)

        # Project to Q, K, V
        x = self.qkv_proj(x)
        Q, K, V = torch.chunk(x, chunks=3, dim=-1)

        # Split across attention heads. [B, h, N, D // h]
        Q = Q.reshape(B, N, self.h, D // self.h).permute((0, 2, 1, 3))
        K = K.reshape(B, N, self.h, D // self.h).permute((0, 2, 1, 3))
        V = V.reshape(B, N, self.h, D // self.h).permute((0, 2, 1, 3))

        # Apply RMS Norm
        Q = self.rmsnorm_q(Q)
        K = self.rmsnorm_k(K)

        # Multi Head Self Attention
        attn_scores =  (Q @ K.transpose(-2, -1) / sqrt(D // self.h)) + mask # apply causal mask
        attn_weights = self.attn_dropout(self.softmax(attn_scores))
        z = attn_weights @ V

        # Concatenate (go from [B, h, N, D // h] to [B, N, h, D // h] to [B, N, D])
        z = z.permute((0, 2, 1, 3)).reshape(B, N, D)

        o = self.out_proj(z)

        return o

class MLP(nn.Module):
    def __init__(self, embed_dim=128, hidden_dim=512):
        super().__init__()

        self.ffn1 = nn.Linear(embed_dim, hidden_dim)
        self.ffn2 = nn.Linear(hidden_dim, embed_dim)
        self.gelu = nn.GELU()

    def forward(self, x):
        x = self.ffn2(self.gelu(self.ffn1(x)))

        return x

