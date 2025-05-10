import torch
import torch.nn as nn
import torch.nn.functional as F


class Prototype(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.psi_r = nn.Sequential(
            nn.Linear(input_dim * 2, hidden_dim),
            nn.ReLU()
        )
        self.psi_a = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.ReLU()
        )

    def forward(self, z_t):
        """
        z_t: [batch_size, num_nodes, feature_dim]
        """
        B, N, D = z_t.shape  # [8, 80, 30]

        # expand z_i: [B, N, 1, D]
        z_i = z_t.unsqueeze(2)  # [8, 80, 1, 30]
        # expand z_j: [B, 1, N, D]
        z_j = z_t.unsqueeze(1)  # [8, 1, 80, 30]

        # concat along feature dim
        z_pair = torch.cat([z_i.expand(-1, -1, N, -1), z_j.expand(-1, N, -1, -1)], dim=-1)  # [8, 80, 80, 60]

        # flatten last 2 dims for feeding into FFN
        z_pair_flat = z_pair.view(B * N * N, D * 2)  # [8*80*80, 60]

        r = self.psi_r(z_pair_flat)  # [8*80*80, hidden_dim]

        r = r.view(B, N, N, -1)  # [8, 80, 80, hidden_dim]

        r_sum = r.sum(dim=2)  # sum over neighbors → [8, 80, hidden_dim]

        f_i = self.psi_a(r_sum)  # [8, 80, output_dim]

        return f_i  # f_i^k
K = 5  # 举例
input_dim = 30
hidden_dim = 64
output_dim = 30

prototypes = nn.ModuleList([Prototype(input_dim, hidden_dim, output_dim) for _ in range(K)])


