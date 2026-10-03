import torch
import torch.nn as nn


# ============================================================
# TimeEmbedding: sinusoidal positional embedding untuk timestep
# ------------------------------------------------------------
# Mengubah skalar timestep t (0..T) menjadi vektor berdimensi `dim`
# menggunakan frekuensi geometris: ω_i = 1/10000^(2i/dim)
# Output: (B, dim)
# ============================================================
class TimeEmbedding(nn.Module):
    def __init__(self, dim: int, device: str | torch.device = "cpu"):
        super().__init__()
        self.dim = dim
        self.device = device
        half = dim // 2
        # inv_freq[i] = 1 / 10000^(2i/dim) untuk i = 0..half-1
        self.inv_freq = 1.0 / (10000 ** (2 * torch.arange(half).float() / dim))

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) -> emb_time: (B, dim)
        emb_time = torch.zeros(t.size(0), self.dim, device=self.device)
        # indeks genap: sin, indeks ganjil: cos
        emb_time[:, 0::2] = torch.sin(t.unsqueeze(1) * self.inv_freq)
        emb_time[:, 1::2] = torch.cos(t.unsqueeze(1) * self.inv_freq)
        return emb_time


# ============================================================
# FiLM (Feature-wise Linear Modulation)
# ------------------------------------------------------------
# Mengubah vektor kondisi `cond` (B, cond_dim) menjadi
# scale dan bias per channel: (B, channels, 1)
# Lalu memodulasi hidden feature h: out = h * scale + bias
# ============================================================
class FiLM(nn.Module):
    def __init__(self, cond_dim: int, channels: int, device: str | torch.device = "cpu"):
        super().__init__()
        self.cond_encoder = nn.Sequential(
            nn.Mish(),                    # aktivasi non-linear
            nn.Linear(cond_dim, 2 * channels)  # output: scale + bias
        )

    def forward(self, cond, h):
        # cond: (B, cond_dim), h: (B, C, L)
        params = self.cond_encoder(cond)           # (B, 2*C)
        scale, bias = params.chunk(2, dim=1)       # masing-masing (B, C)
        scale = scale.unsqueeze(-1)                # (B, C, 1)
        bias = bias.unsqueeze(-1)                  # (B, C, 1)
        return h * scale + bias                    # broadcast di L -> (B, C, L)


# ============================================================
# Conv1dBlock: Conv1d -> GroupNorm -> Mish
# ------------------------------------------------------------
# Blok bangunan dasar CNN 1D. Padding kernel//2 mempertahankan
# panjang sekuens L. GroupNorm stabil untuk batch kecil.
# ============================================================
class Conv1dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, n_groups=8):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2),
            nn.GroupNorm(n_groups, out_channels),
            nn.Mish()
        )

    def forward(self, x):
        return self.block(x)


# ============================================================
# ConditionalResidualBlock1d
# ------------------------------------------------------------
# Blok residual dengan FiLM conditioning.
# Alur: x -> block1 -> FiLM(cond) -> block2 -> + residual -> out
# Jika in_channels != out_channels, residual lewat Conv1d kernel 1.
# ============================================================
class ConditionalResidualBlock1d(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim,
                 kernel_size=3, n_groups=8, device: str | torch.device = "cpu"):
        super().__init__()
        self.block1 = Conv1dBlock(in_channels, out_channels, kernel_size, n_groups)
        self.film = FiLM(cond_dim, out_channels, device)
        self.block2 = Conv1dBlock(out_channels, out_channels, kernel_size, n_groups)
        # residual: Identity jika channel sama, Conv1d kernel 1 jika beda
        self.residual_conv = (nn.Conv1d(in_channels, out_channels, 1)
                              if in_channels != out_channels else nn.Identity())

    def forward(self, cond, h):
        # cond: (B, cond_dim), h: (B, C, L)
        x = self.block1(h)           # (B, out_channels, L)
        x = self.film(cond, x)       # modulasi FiLM
        x = self.block2(x)           # (B, out_channels, L)
        return x + self.residual_conv(h)  # residual connection


# ============================================================
# Downsample1d: Conv1d stride 2 -> panjang sekuens jadi setengah
# ============================================================
class Downsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv1d(dim, dim, 3, 2, 1)  # kernel=3, stride=2, padding=1

    def forward(self, x):
        return self.conv(x)


# ============================================================
# Upsample1d: ConvTranspose1d stride 2 -> panjang sekuens jadi 2x
# ============================================================
class Upsample1d(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # kernel=4, stride=2, padding=1 -> output length = 2 * input length
        self.conv = nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


# ============================================================
# ConditionalUnet1D: U-Net 1D kondisional untuk noise prediction
# ------------------------------------------------------------
# Arsitektur:
#   input_proj -> Down path (ResBlock + Downsample) x N
#   -> Mid (2x ResBlock) -> Up path (Upsample + ResBlock + skip concat) x N
#   -> output_proj -> prediksi noise ε_θ
# Kondisi: timestep embedding (+ optional global_cond) -> FiLM di tiap ResBlock
# ============================================================
class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim,              # action_dim (mis. 2 untuk x,y)
        global_cond_dim,        # dimensi embedding observasi (state), None jika tidak pakai
        diffusion_step_embed_dim=256,   # dimensi embedding timestep
        down_dims=(256, 512, 1024),     # channel di setiap level downsampling
        kernel_size=3,
        n_groups=8,
    ):
        super().__init__()
        all_dims = [input_dim] + list(down_dims)   # [input_dim, 256, 512, 1024]
        self.down_dims = down_dims

        # ----------------------------------------------------
        # 1) Timestep embedding: t -> sinusoidal -> MLP -> (B, dsed)
        # ----------------------------------------------------
        dsed = diffusion_step_embed_dim
        self.diffusion_step_encoder = nn.Sequential(
            TimeEmbedding(dsed),              # sinusoidal embedding
            nn.Linear(dsed, dsed * 4),
            nn.Mish(),
            nn.Linear(dsed * 4, dsed),        # output: (B, dsed)
        )
        cond_dim = dsed
        if global_cond_dim is not None:
            cond_dim += global_cond_dim  # global_cond dikirim di forward

        # ----------------------------------------------------
        # 2) Input projection: (B, action_dim, L) -> (B, down_dims[0], L)
        # ----------------------------------------------------
        self.input_proj = nn.Conv1d(input_dim, down_dims[0], 1)

        # ----------------------------------------------------
        # 3) Down path: ResBlock + Downsample
        #    skip connection disimpan SETELAH block, SEBELUM downsample
        # ----------------------------------------------------
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        for i in range(len(down_dims) - 1):
            self.down_blocks.append(
                ConditionalResidualBlock1d(
                    down_dims[i], down_dims[i + 1], cond_dim
                )
            )
            self.down_samples.append(Downsample1d(down_dims[i + 1]))

        # ----------------------------------------------------
        # 4) Mid blocks (bottleneck)
        # ----------------------------------------------------
        mid_dim = down_dims[-1]          # channel di bottleneck
        self.mid_block1 = ConditionalResidualBlock1d(mid_dim, mid_dim, cond_dim)
        self.mid_block2 = ConditionalResidualBlock1d(mid_dim, mid_dim, cond_dim)

        # ----------------------------------------------------
        # 5) Up path: Upsample + concat skip + ResBlock
        #    channel input up_block = channel_up * 2 (karena concat skip)
        # ----------------------------------------------------
        self.up_blocks = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        for i in range(len(down_dims) - 1):
            self.up_samples.append(Upsample1d(down_dims[-1 - i]))
            self.up_blocks.append(
                ConditionalResidualBlock1d(
                    down_dims[-1 - i] * 2,   # concat skip -> channel x2
                    down_dims[-2 - i],        # output channel level berikutnya
                    cond_dim,
                )
            )

        # ----------------------------------------------------
        # 6) Output projection: (B, down_dims[0], L) -> (B, action_dim, L)
        # ----------------------------------------------------
        self.output_proj = nn.Conv1d(down_dims[0], input_dim, 1)

    def forward(self, x, timestep, global_cond=None):
        """
        x: (B, action_dim, L)        - action chunk noisy
        timestep: (B,)               - integer 0..T
        global_cond: (B, global_cond_dim) or None - embedding observasi (state)
        return: (B, action_dim, L)   - prediksi noise ε_θ
        """
        # 1) Timestep embedding
        t_emb = self.diffusion_step_encoder(timestep)   # (B, dsed)

        # 2) Gabungkan dengan global_cond (observasi state)
        if global_cond is not None:
            cond = torch.cat([t_emb, global_cond], dim=1)  # (B, cond_dim)
        else:
            cond = t_emb

        # 3) Input projection
        h = self.input_proj(x)      # (B, down_dims[0], L)

        # 4) Down path
        skips = []
        for block, downsample in zip(self.down_blocks, self.down_samples):
            h = block(cond, h)      # ResBlock + FiLM (cond first, then h)
            skips.append(h)         # simpan skip SEBELUM downsample
            h = downsample(h)       # panjang sekuens /2
        
        # 5) Mid (bottleneck)
        h = self.mid_block1(cond, h)
        h = self.mid_block2(cond, h)
        
        # 6) Up path
        for upsample, block in zip(self.up_samples, self.up_blocks):
            h = upsample(h)                     # panjang sekuens x2
            skip = skips.pop()                  # ambil skip connection
            h = torch.cat([h, skip], dim=1)     # concat di channel
            h = block(cond, h)                  # ResBlock + FiLM
        
        # 7) Output projection
        return self.output_proj(h)              # (B, action_dim, L)
    
class StateEncoder(nn.Module):
    def __init__(self, state_dim: int = 5, hidden_dim: int = 256, out_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Mish(),                    
            nn.Linear(hidden_dim, hidden_dim),
            nn.Mish(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)
