from __future__ import annotations
import torch, torch.nn as nn
import os, tempfile
from transformers import PreTrainedModel, AutoConfig, PretrainedConfig
from typing import Optional, Tuple, Dict, Any, Union, List

from isca.utils.isca import ISCA


class ISCAConfig(PretrainedConfig):
    model_type = "isca"

    def __init__(
        self,
        backbone: str = "meta-llama/Llama-2-7b-hf",
        freeze_layers: int = 6,
        hidden_dim: int = 4096,
        num_centroids: int = 256,
        num_operator_flows: int = 32,
        flow_depth: int = 2,
        tau_role: float = 0.07,
        gamma_mem: float = 0.95,
        lambda_sym: float = 0.5,
        lambda_flow: float = 1.0,
        lambda_self: float = 0.5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.backbone = backbone
        self.freeze_layers = freeze_layers
        self.hidden_dim = hidden_dim
        self.num_centroids = num_centroids
        self.num_operator_flows = num_operator_flows
        self.flow_depth = flow_depth
        self.tau_role = tau_role
        self.gamma_mem = gamma_mem
        self.lambda_sym = lambda_sym
        self.lambda_flow = lambda_flow
        self.lambda_self = lambda_self


class ISCAModelForCausalLM(PreTrainedModel):
    config_class = ISCAConfig
    base_model_prefix = "isca"
    supports_gradient_checkpointing = True

    def __init__(self, config: ISCAConfig):
        super().__init__(config)

        # Convert config to the format ISCA expects
        model_cfg = {
            "backbone": config.backbone,
            "freeze_layers": config.freeze_layers,
            "hidden_dim": config.hidden_dim,
            "num_centroids": config.num_centroids,
            "num_operator_flows": config.num_operator_flows,
            "flow_depth": config.flow_depth,
            "tau_role": config.tau_role,
            "gamma_mem": config.gamma_mem,
            "lambda_sym": config.lambda_sym,
            "lambda_flow": config.lambda_flow,
            "lambda_self": config.lambda_self,
        }

        self.isca = ISCA(model_cfg)

    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        step: int = 0,
        **kwargs,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the ISCA model.

        Args:
            input_ids: Tensor of token ids
            attention_mask: Attention mask for the input
            labels: Target labels for the language modeling task
            step: Current training step

        Returns:
            Dictionary containing loss values and other metrics
        """
        # Prepare the config for the forward pass
        cfg = {
            "backbone": self.config.backbone,
            "freeze_layers": self.config.freeze_layers,
            "hidden_dim": self.config.hidden_dim,
            "num_centroids": self.config.num_centroids,
            "num_operator_flows": self.config.num_operator_flows,
            "flow_depth": self.config.flow_depth,
            "tau_role": self.config.tau_role,
            "gamma_mem": self.config.gamma_mem,
            "lambda_sym": self.config.lambda_sym,
            "lambda_flow": self.config.lambda_flow,
            "lambda_self": self.config.lambda_self,
        }

        # Default attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        # Forward pass through ISCA
        outputs = self.isca(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels if labels is not None else torch.zeros_like(input_ids),
            cfg=cfg,
            step=step,
        )

        return outputs

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation.
        """
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        *model_args,
        **kwargs,
    ) -> "ISCAModelForCausalLM":
        """
        Load a pretrained ISCA model.
        """
        # Load config
        config = kwargs.pop("config", None)
        if config is None:
            config = ISCAConfig.from_pretrained(pretrained_model_name_or_path, **kwargs)

        # Create model instance
        model = cls(config)

        # Load model weights
        model.load_state_dict(
            torch.load(
                pretrained_model_name_or_path, map_location=kwargs.get("device", "cpu")
            )
        )

        return model

    def save_pretrained(
        self,
        save_directory: Union[str, os.PathLike],
        save_config: bool = True,
        **kwargs,
    ):
        """
        Save the model and its configuration.
        """
        # Save configuration if requested
        if save_config:
            self.config.save_pretrained(save_directory)

        # Save model weights
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

    def push_to_hub(
        self,
        repo_id: str,
        commit_message: Optional[str] = None,
        private: bool = False,
        token: Optional[str] = None,
        **kwargs,
    ):
        """
        Push the model to the Hugging Face Hub.
        """
        from huggingface_hub import HfApi

        # Create a temporary directory to save the model
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save the model
            self.save_pretrained(tmpdir)

            # Push the model to the hub
            api = HfApi(token=token)
            api.create_repo(repo_id=repo_id, private=private, exist_ok=True)
            api.upload_folder(
                folder_path=tmpdir,
                repo_id=repo_id,
                commit_message=commit_message or f"Upload {repo_id}",
            )
