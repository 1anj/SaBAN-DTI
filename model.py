import torch.nn as nn
from modules import ProtProjector, DrugProjector, MLPClassifier, BAN, Clip

class DTIModel(nn.Module):
    def __init__(
        self,
        prot_dim=1280,
        drug_dim=768, 
        latent_dim=1024,
        num_heads=8,
        dropout=0.05,
    ):
        super().__init__()
        
        self.prot_proj = ProtProjector(
            prot_dim, latent_dim, 
            activation=nn.ELU, 
            num_heads=num_heads,
            attn_drop=dropout, 
            proj_drop=dropout
        )

        self.drug_proj = DrugProjector(
            drug_dim, latent_dim,
            activation=nn.ELU,
            num_heads=num_heads,
            attn_drop=dropout,
            proj_drop=dropout
        )

        self.ban_layer = BAN(v_dim=latent_dim, q_dim=latent_dim, h_dim=latent_dim, h_out=num_heads)
        self.classifier = MLPClassifier(input_dim=latent_dim)
        self.clip = Clip()
        
    def forward(self, prot_embed, drug_embed, prot_seqs=None, drug_smiles=None, labels=None):
        prot_embed = self.prot_proj(prot_embed)
        drug_embed = self.drug_proj(drug_embed)
        cl_loss, _ = self.clip(prot_embed, drug_embed)
        prot_embed = prot_embed.unsqueeze(1)
        drug_embed = drug_embed.unsqueeze(1)
        coembed, _ = self.ban_layer(prot_embed, drug_embed)
        score = self.classifier(coembed)
        return score, cl_loss

class TokenEncoder(nn.Module):
    def __init__(self, prot_encoder, drug_encoder):
        super().__init__()
        self.prot_encoder = prot_encoder
        self.drug_encoder = drug_encoder
        
    def encoding(self, prot_ids, prot_mask, drug_ids, drug_mask):
        prot_embed = self.prot_encoder(
            input_ids=prot_ids, 
            attention_mask=prot_mask,
            output_hidden_states=True, 
            return_dict=True
        ).hidden_states[-1]

        drug_embed = self.drug_encoder(
            input_ids=drug_ids, 
            attention_mask=drug_mask, 
            return_dict=True
        ).last_hidden_state

        return prot_embed, drug_embed