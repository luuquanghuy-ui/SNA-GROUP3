import torch
import torch.nn as nn
import torch.nn.functional as F


class E1MultiBranch(nn.Module):
    def __init__(
        self,
        tweet_dim=384,
        profile_text_dim=384,
        profile_struct_dim=15,
        neighbor_dim=768,
        branch_dim=128,
        fusion_dim=256,
        projection_dim=128,
        num_classes=2,
        dropout=0.3
    ):
        super().__init__()

        def make_branch(input_dim):
            return nn.Sequential(
                nn.Linear(input_dim, branch_dim),
                nn.LayerNorm(branch_dim),
                nn.GELU(),
                nn.Dropout(dropout)
            )

        self.tweet_branch = make_branch(tweet_dim)
        self.profile_text_branch = make_branch(profile_text_dim)
        self.profile_struct_branch = make_branch(profile_struct_dim)
        self.neighbor_branch = make_branch(neighbor_dim)

        self.fusion = nn.Sequential(
            nn.Linear(branch_dim * 4, fusion_dim),
            nn.LayerNorm(fusion_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

        self.classifier = nn.Linear(fusion_dim, num_classes)

        self.projector = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.GELU(),
            nn.Linear(fusion_dim, projection_dim)
        )

    def forward(self, tweet, profile_text, profile_struct, neighbor):
        t = self.tweet_branch(tweet)
        p = self.profile_text_branch(profile_text)
        s = self.profile_struct_branch(profile_struct)
        n = self.neighbor_branch(neighbor)

        combined = torch.cat([t, p, s, n], dim=1)
        fused = self.fusion(combined)

        logits = self.classifier(fused)
        z = F.normalize(self.projector(fused), dim=1)

        return logits, z
