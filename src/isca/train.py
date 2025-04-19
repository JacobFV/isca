from __future__ import annotations
import yaml, torch, os
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from isca.data.text_dataset import TextDataset
from isca.models.isca import ISCA
from isca.utils.sched import WarmupCosine


def load_cfg(path):
    return yaml.safe_load(Path(path).read_text())


def main(args):
    cfg_path = args.config
    print(f"Loading configuration from {cfg_path}")

    cfg_all = load_cfg(cfg_path)
    cfg_m, cfg_t, cfg_l = cfg_all["model"], cfg_all["train"], cfg_all["loss"]

    ds = TextDataset(cfg_t["dataset"], cfg_m["backbone"], cfg_t["max_seq"])
    dl = DataLoader(ds, batch_size=cfg_t["batch_size"], shuffle=True, drop_last=True)

    model = ISCA({**cfg_m, **cfg_l}).to(cfg_t["device"])
    opt = torch.optim.AdamW(model.parameters(), lr=cfg_t["lr"])
    sched = WarmupCosine(opt, cfg_t["warmup"], cfg_t["steps"], cfg_t["lr"])

    step = 0
    ckpt_dir = Path(cfg_t["ckpt_dir"])
    ckpt_dir.mkdir(exist_ok=True)
    pbar = tqdm(total=cfg_t["steps"])
    
    while step < cfg_t["steps"]:
        for batch in dl:
            step += 1
            pbar.update(1)
            batch = {k: v.to(cfg_t["device"]) for k, v in batch.items()}
            out = model(**batch, cfg={**cfg_m, **cfg_l}, step=step)
            out["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()

            if step % cfg_t["log_every"] == 0:
                pbar.set_postfix({k: f"{v:.3f}" for k, v in out.items() if k != "loss"})

            if step % cfg_t["save_every"] == 0:
                checkpoint_path = ckpt_dir / f"isca_{step}.pt"
                torch.save(model.state_dict(), checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")

            if step >= cfg_t["steps"]:
                break


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
