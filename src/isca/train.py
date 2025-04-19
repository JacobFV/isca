from __future__ import annotations
import yaml, torch, os
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
import math

# Set tokenizers parallelism before importing any HuggingFace components
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from isca.data.text_dataset import MixedDataset
from isca.models.isca import ISCA
from isca.utils.sched import WarmupCosine


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text())


def main(args):
    cfg_path = args.config
    print(f"Loading configuration from {cfg_path}")

    cfg_all = load_cfg(cfg_path)
    cfg_m, cfg_t, cfg_l = cfg_all["model"], cfg_all["train"], cfg_all["loss"]

    # Create mixed dataset
    print("Loading datasets...")
    ds = MixedDataset.create(
        model_name=cfg_m["backbone"],
        max_len=cfg_t["max_seq"],
        datasets=cfg_t.get("datasets", None)  # Use all datasets if not specified
    )
    dl = DataLoader(
        ds,
        batch_size=cfg_t["batch_size"],
        shuffle=True,
        drop_last=True,
        num_workers=2  # Reduced number of workers to minimize forking issues
    )

    # Calculate epochs based on dataset size and desired steps
    total_steps = cfg_t["steps"]
    steps_per_epoch = len(dl)
    num_epochs = math.ceil(total_steps / steps_per_epoch)
    print(f"Training for {num_epochs} epochs ({total_steps} total steps)")
    print(f"Steps per epoch: {steps_per_epoch}")

    model = ISCA({**cfg_m, **cfg_l}).to(cfg_t["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_t["lr"])
    sched = WarmupCosine(opt, cfg_t["warmup"], cfg_t["steps"], cfg_t["lr"])

    step = 0
    ckpt_dir = Path(cfg_t["ckpt_dir"])
    ckpt_dir.mkdir(exist_ok=True)

    # Main training loop with epoch progress
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs")
    for epoch in epoch_pbar:
        # Progress bar for steps within epoch
        pbar = tqdm(dl, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch in pbar:
            step += 1
            batch = {k: v.to(cfg_t["device"]) for k, v in batch.items()}
            out = model(**batch, cfg={**cfg_m, **cfg_l}, step=step)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()

            # Update progress bars
            if step % cfg_t["log_every"] == 0:
                metrics = {k: f"{v:.3f}" for k, v in out.items()}
                metrics["lr"] = f"{sched.get_last_lr()[0]:.2e}"
                pbar.set_postfix(metrics)
                epoch_pbar.set_postfix(metrics)

            if step % cfg_t["save_every"] == 0:
                checkpoint_path = ckpt_dir / f"isca_{step}.pt"
                torch.save(
                    {
                        "step": step,
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": opt.state_dict(),
                        "scheduler_state_dict": sched.state_dict(),
                    },
                    checkpoint_path
                )
                print(f"\nSaved checkpoint to {checkpoint_path}")

            if step >= total_steps:
                print("\nReached total steps, training complete!")
                return

    print("\nTraining complete!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ISCA model")
    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default="config/default.yaml",
        help="Path to the configuration YAML file",
    )
    args = parser.parse_args()
    main(args)
