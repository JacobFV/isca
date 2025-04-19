from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F
from transformers import AutoModelForCausalLM
from isca.models.attractor_symbol import AttractorSymbolLayer
from isca.models.operator_flow import OperatorFlow
from isca.models.identity_tracker import IdentityTracker
from isca.utils.graph_utils import infer_soft_graph


class ISCA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Load GPT-2 model
        model = AutoModelForCausalLM.from_pretrained(
            cfg["backbone"], trust_remote_code=True, torch_dtype=torch.float32  # Changed to float32 for MPS compatibility
        )
        # Access GPT-2 transformer directly
        self.backbone = model.transformer
        self.vocab_size = model.config.vocab_size

        # freeze lower layers
        freeze_count = 0
        for name, p in self.backbone.named_parameters():
            if freeze_count < cfg["freeze_layers"] * 2:
                p.requires_grad_(False)
                freeze_count += 1

        dim = cfg["hidden_dim"]
        k = cfg["num_centroids"]
        flows = cfg["num_operator_flows"]

        self.attractor = AttractorSymbolLayer(dim, k)
        self.flows = nn.ModuleList(
            [OperatorFlow(dim, cfg["flow_depth"]) for _ in range(flows)]
        )
        self.identity = IdentityTracker()
        self.tau_role = cfg["tau_role"]
        self.gamma_mem = cfg["gamma_mem"]

        self.lm_head = nn.Linear(dim, self.vocab_size, bias=False)

        # Get number of heads from GPT-2 config
        heads = self.backbone.config.num_attention_heads

        self.role_proj_q = nn.Linear(dim, heads, bias=False)
        self.role_proj_k = nn.Linear(dim, heads, bias=False)

        self.graph_memory: torch.Tensor | None = None

    def _role_similarity(self, h):
        """
        produce role‑conditioned attention pass‑through mask
        """
        # Ensure input is float32
        h = h.to(torch.float32)
        
        # Project and normalize with proper shape handling
        q = self.role_proj_q(h)  # (b,n,h)
        k = self.role_proj_k(h)  # (b,n,h)
        
        # Normalize along last dimension
        q = F.normalize(q, dim=-1)
        k = F.normalize(k, dim=-1)
        
        # Compute similarity with explicit shapes
        sim = torch.matmul(q.transpose(1, 2), k)  # (b,h,n,n)
        
        # Apply temperature scaling and softmax
        sim = sim / self.tau_role
        return F.softmax(sim, dim=-1)

    def forward(self, input_ids, attention_mask, labels, cfg, step):
        # Convert attention mask to float32 for consistent dtype
        attention_mask = attention_mask.to(torch.float32)
        
        # GPT-2 specific embedding
        h = self.backbone.wte(input_ids)
        h = h.to(torch.float32)  # Ensure float32 type
        encoder_blocks = self.backbone.h

        # ----------------- encoder half ------------------ #
        for i, blk in enumerate(encoder_blocks):
            # GPT-2 blocks take (h, attention_mask) and return a tuple
            h = blk(h, attention_mask=attention_mask)[0]

            if i == cfg["freeze_layers"]:
                # ---------- symbol extraction ------------- #
                attract, assign = self.attractor(h)
                h = h + attract  # inject symbols

                # build soft graph from assignments
                A_soft = infer_soft_graph(assign)  # (b,n,n)

                # memory integration - replace in-place operations with new tensor creation
                if self.graph_memory is None:
                    self.graph_memory = A_soft.detach().clone()
                else:
                    new_memory = cfg["gamma_mem"] * self.graph_memory.to(A_soft.dtype) + (1 - cfg["gamma_mem"]) * A_soft.detach()
                    self.graph_memory = new_memory.clone()

                # ------------ operator flows -------------- #
                flow_outs = [f(h) for f in self.flows]  # list[(b,n,d)]
                # closure loss: pairwise composition distance
                comp_loss = torch.tensor(0.0, device=h.device, dtype=torch.float32)
                for a in range(len(flow_outs)):
                    for b in range(len(flow_outs)):
                        c = (a + b) % len(flow_outs)
                        comp_loss += (self.flows[c](h) - flow_outs[b]).pow(2).mean()
                comp_loss = comp_loss / (len(flow_outs) ** 2)

                self.comp_loss = comp_loss
                self.assign = assign
                self.graph_adj = A_soft

                # identity self‑loss
                self_loss = self.identity(A_soft)
                self.self_loss = torch.tensor(self_loss, device=h.device, dtype=torch.float32)

        # ---------------- lm head + role‑atten tuning --------------- #
        logits = self.lm_head(h)

        # role‑similarity regularizer
        role_attn = self._role_similarity(h)
        role_loss = (role_attn * (1 - role_attn)).mean()  # encourage crisp roles

        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100
        )

        total = (
            ce_loss
            + cfg["lambda_sym"] * (h - attract.detach()).pow(2).mean()
            + cfg["lambda_flow"] * self.comp_loss
            + cfg["lambda_self"] * self.self_loss
            + role_loss * 0.1
        )

        return {
            "loss": total,
            "ce": ce_loss.detach(),
            "sym": (h - attract.detach()).pow(2).mean().detach(),
            "flow": self.comp_loss.detach(),
            "self": self.self_loss.detach(),
            "role": role_loss.detach(),
        }
