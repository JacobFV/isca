# Integrated Symbolic Cognition Architecture (ISCA)

A fully-fleshed implementation of the Integrated Symbolic Cognition Architecture (ISCA).

## Installation and Usage

This project uses [Poetry](https://python-poetry.org/) for dependency management:

```bash
# Install Poetry if you don't have it
curl -sSL https://install.python-poetry.org | python -

# Install dependencies
poetry install

# Run training
poetry run python src/train.py
```

## Training

The training process can be configured through the `config/default.yaml` file:

```bash
# Run with the default configuration
poetry run python src/train.py

# Run with a custom configuration file
poetry run python src/train.py --config config/custom.yaml
```

### Checkpoints

Checkpoints are saved during training based on the following configuration parameters:

- `train.ckpt_dir`: Directory where checkpoints are saved (default: "checkpoints")
- `train.save_every`: How often to save checkpoints in training steps (default: 2000)

The checkpoints follow the naming pattern `isca_{step}.pt` where `step` is the training step.

To modify the checkpoint directory or frequency:

1. Edit `config/default.yaml`:
   ```yaml
   train:
     # Other parameters...
     ckpt_dir: "my_checkpoints"    # Custom checkpoint directory
     save_every: 1000              # Save checkpoint every 1000 steps
   ```

2. Or create a custom config file with your desired settings.

### Finding the Latest Checkpoint

You can find the latest checkpoint by looking at the file with the highest step number in the checkpoint directory:

```bash
# List all checkpoints sorted by modification time
ls -lt checkpoints/

# Or find the checkpoint with the highest step number
ls -1 checkpoints/isca_*.pt | sort -V | tail -n 1
```

## Evaluation

To evaluate a trained model, use the evaluation script with a checkpoint:

```bash
# Basic evaluation with a specific checkpoint
poetry run python src/eval.py --checkpoint checkpoints/isca_10000.pt

# Run with visualization
poetry run python src/eval.py --checkpoint checkpoints/isca_10000.pt --visualize --save_plots plots/

# Evaluate on custom data
poetry run python src/eval.py --checkpoint checkpoints/isca_10000.pt --eval_data path/to/eval/data.txt

# Custom evaluation
# Modify src/isca/eval.py to uncomment and customize the evaluation sections
```

The evaluation infrastructure includes several components for analyzing the model:

1. Basic metrics (loss, perplexity, component-wise losses)
2. Attractor centroid visualization
3. Symbol assignment analysis
4. Operator flow analysis 
5. Graph memory evolution analysis
6. Role similarity patterns

Each evaluation section is commented out and can be customized for specific analysis needs.

## Architecture Overview

ISCA consists of the following components:

1. **Attractor Symbol Layer**: Learnable set of centroids with differentiable assignment and EMA updates.
2. **Identity Tracker**: Tracks persistence of the 'self' subgraph with spectral-persistence loss.
3. **Operator Flow**: Family of learnable vector-field operators with closure regularization.
4. **Graph Memory**: Evolving graph with decay parameter for bounded effective horizon.
5. **Role-Similarity Gating**: Attention heads repurposed as logic edges for interpretable reasoning paths.

## Design Rationale

ISCA aligns with "latent cognition" by:

- Treating symbols as manifolds (centroids & attractor assignments)
- Implementing reasoning as flow (operator flows)
- Modeling self as a graph attractor (identity tracker)
- Using memory as an evolving graph
- Using roles to guide attention

No scaffolding or external interpreter needed - the symbolic machinery lives in the weights and activations.
