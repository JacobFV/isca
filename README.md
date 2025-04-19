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
poetry run python -m src.isca.train
```

## Base Models

ISCA can work with different base language models:

- Default: `moonshotai/Kimi-VL-A3B-Thinking` - An efficient MoE model with 16B total parameters, activating only 3B during inference, optimized for reasoning
- Alternative: `meta-llama/Llama-2-7b-hf` - The original Llama 2 7B model

To change the base model, modify the `backbone` parameter in `config/default.yaml`.

## Training

The training process can be configured through the `config/default.yaml` file:

```bash
# Run with the default configuration
poetry run python -m src.isca.train

# Run with a custom configuration file
poetry run python -m src.isca.train --config config/custom.yaml
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

ISCA includes a modular evaluation system that supports various types of analysis and benchmarks. The main entry point is the evaluation script in `src/isca/eval/main.py`.

```bash
# Basic evaluation with a specific checkpoint
poetry run python -m src.isca.eval.main --checkpoint checkpoints/isca_10000.pt

# Run with a specific evaluation type
poetry run python -m src.isca.eval.main --checkpoint checkpoints/isca_10000.pt --eval_type symbolic

# Run with visualization
poetry run python -m src.isca.eval.main --checkpoint checkpoints/isca_10000.pt --visualize --save_plots plots/

# Evaluate on custom data
poetry run python -m src.isca.eval.main --checkpoint checkpoints/isca_10000.pt --eval_data path/to/eval/data.txt
```

### Evaluation Types

The evaluation system supports multiple types of analysis:

1. `basic` - Standard metrics (loss, perplexity, component-wise losses)
2. `centroids` - Attractor centroid visualization using PCA
3. `symbolic` - Symbol assignment analysis (entropy, sparsity)
4. `operators` - Operator flow analysis (displacement, consistency)
5. `graph` - Graph structure analysis
6. `memory` - Graph memory evolution analysis 
7. `roles` - Role similarity pattern analysis
8. `mmlu` - MMLU benchmark evaluation
9. `mmlu_pro` - MMLU-Pro benchmark evaluation
10. `real_bench` - REAL Bench evaluation for web agents
11. `custom` - Custom evaluation using a specified module

Example:

```bash
# Run symbolic representation analysis
poetry run python -m src.isca.eval.main --checkpoint checkpoints/isca_10000.pt --eval_type symbolic --visualize
```

## Benchmark Evaluations

ISCA supports several standard benchmarks for evaluating model capabilities:

### Supported Benchmarks

| Benchmark | Description | Eval Type |
|-----------|-------------|-----------|
| MMLU | Multiple-choice questions across 57 subjects (STEM, humanities, etc.) | `mmlu` |
| MMLU-Pro | More challenging variant with 10 options and complex reasoning tasks | `mmlu_pro` |
| REAL Bench | Evaluates web agent capabilities on 11 website replicas | `real_bench` |

You can evaluate using either the main evaluation script or call the benchmark modules directly for more options:

```bash
# Using the main evaluation script
poetry run python -m src.isca.eval.main --checkpoint checkpoints/isca_10000.pt --eval_type mmlu

# Calling the benchmark module directly
poetry run python -m src.isca.eval.benchmarks.mmlu_eval --checkpoint checkpoints/isca_10000.pt \
    --subjects high_school_mathematics elementary_mathematics \
    --output_file results/mmlu_results.json
```

### Benchmark-Specific Options

When calling benchmark modules directly, you have access to additional options:

#### MMLU (`mmlu_eval.py`)
- `--subjects`: Specific subjects to evaluate on
- `--model_type`: Type of model to evaluate (isca, huggingface, openai, anthropic, gemini)
- `--model_name`: Name/path of the model (for non-ISCA models)
- `--output_file`: Path to save results as JSON

#### MMLU-Pro (`mmlu_pro_eval.py`)
- `--categories`: Specific categories to evaluate on
- `--use_cot`: Whether to use chain-of-thought reasoning (default: True)
- `--max_new_tokens`: Maximum tokens for CoT responses (default: 200)

#### REAL Bench (`real_bench_eval.py`)
- `--websites`: Specific websites to evaluate on
- `--task_types`: Evaluate on specific task types
- `--max_new_tokens`: Maximum tokens per response (default: 1024)
- `--output_dir`: Directory to save detailed results

For all benchmarks, you can specify model type (`--model_type`), which supports ISCA models, HuggingFace models, and API-based models (OpenAI, Anthropic, Google Gemini).

## HuggingFace Integration

The ISCA model includes a HuggingFace wrapper that allows it to be used with the Transformers library ecosystem.

### Using the HuggingFace Wrapper

```python
from isca.utils.isca_hf import ISCAConfig, ISCAModelForCausalLM

# Load a trained ISCA model with Kimi-VL-A3B-Thinking backbone
config = ISCAConfig(
    backbone="moonshotai/Kimi-VL-A3B-Thinking",
    freeze_layers=6,
    hidden_dim=3072,
    num_centroids=256,
    num_operator_flows=32,
    flow_depth=2,
    tau_role=0.07,
    gamma_mem=0.95,
)

model = ISCAModelForCausalLM.from_pretrained("path/to/checkpoint", config=config)
```

### Converting Checkpoints

To convert an existing ISCA checkpoint to HuggingFace format, use the conversion script:

```bash
# Convert a checkpoint
poetry run python src/isca/utils/convert_checkpoint.py \
    --checkpoint checkpoints/isca_10000.pt \
    --config config/default.yaml \
    --output_dir ./hf_model

# Convert and push to HuggingFace Hub
poetry run python src/isca/utils/convert_checkpoint.py \
    --checkpoint checkpoints/isca_10000.pt \
    --config config/default.yaml \
    --output_dir ./hf_model \
    --push_to_hub "your-username/isca-model"
```

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