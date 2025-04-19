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
            cfg["backbone"], trust_remote_code=True, torch_dtype=torch.bfloat16
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
        # Convert role projection weights to match h dtype
        if self.role_proj_q.weight.dtype != h.dtype:
            self.role_proj_q.weight = nn.Parameter(self.role_proj_q.weight.to(h.dtype))
        if self.role_proj_k.weight.dtype != h.dtype:
            self.role_proj_k.weight = nn.Parameter(self.role_proj_k.weight.to(h.dtype))
        
        q = F.normalize(self.role_proj_q(h), dim=-1)  # (b,n,h)
        k = F.normalize(self.role_proj_k(h), dim=-1)  # (b,n,h)
        sim = torch.einsum("bnh,bmh->bhnm", q, k)  # (b,h,n,m)
        return (sim / self.tau_role).softmax(-1)

    def forward(self, input_ids, attention_mask, labels, cfg, step):
        # GPT-2 specific embedding
        h = self.backbone.wte(input_ids)
        encoder_blocks = self.backbone.h

        # Convert attention mask to float or bool to match query dtype
        attention_mask = attention_mask.to(h.dtype)

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

                # memory integration - replace in-place operations
                if self.graph_memory is None:
                    self.graph_memory = A_soft.detach()
                else:
                    self.graph_memory = (
                        cfg["gamma_mem"] * self.graph_memory
                        + (1 - cfg["gamma_mem"]) * A_soft.detach()
                    )

                # ------------ operator flows -------------- #
                flow_outs = [f(h) for f in self.flows]  # list[(b,n,d)]
                # closure loss: pairwise composition distance
                comp_loss = 0.0
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
                self.self_loss = torch.tensor(self_loss, device=h.device, dtype=h.dtype)

        # ---------------- lm head + role‑atten tuning --------------- #
        # Convert lm_head weight to match h dtype
        if self.lm_head.weight.dtype != h.dtype:
            self.lm_head.weight = nn.Parameter(self.lm_head.weight.to(h.dtype))
        
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
