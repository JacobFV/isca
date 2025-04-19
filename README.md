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

## MMLU Evaluation

MMLU (Massive Multitask Language Understanding) is a benchmark consisting of multiple-choice questions across 57 subjects including STEM, humanities, social sciences, and more.

To evaluate your ISCA model on the MMLU benchmark:

```bash
# Evaluate on all MMLU subjects
poetry run python -m src.isca.examples.mmlu_eval --checkpoint checkpoints/isca_10000.pt

# Evaluate on specific subjects
poetry run python -m src.isca.examples.mmlu_eval --checkpoint checkpoints/isca_10000.pt \
    --subjects high_school_mathematics elementary_mathematics college_mathematics

# Save results to a file
poetry run python -m src.isca.examples.mmlu_eval --checkpoint checkpoints/isca_10000.pt \
    --output_file results/mmlu_results.json
```

The script will:
1. Load the HuggingFace MMLU dataset
2. Evaluate the model on multiple-choice questions across different subjects
3. Report accuracy per subject and per category (STEM, Humanities, Social Sciences, Other)
4. Calculate overall accuracy across all evaluated subjects

Additional options:
- `--max_seq_len`: Maximum sequence length for inputs (default: 2048)
- `--device`: Device to use for evaluation (default: cuda if available, otherwise cpu)
- `--batch_size`: Batch size for evaluation (default: 1)

## MMLU-Pro Evaluation

MMLU-Pro is a more challenging variant of MMLU with 10 options per question (instead of 4) and a focus on complex reasoning tasks. According to the dataset creators, MMLU-Pro is a more robust benchmark that better distinguishes model capabilities.

To evaluate your ISCA model on the MMLU-Pro benchmark:

```bash
# Evaluate on all MMLU-Pro categories with chain-of-thought reasoning (recommended)
poetry run python -m src.isca.examples.mmlu_pro_eval --checkpoint checkpoints/isca_10000.pt

# Evaluate on specific categories
poetry run python -m src.isca.examples.mmlu_pro_eval --checkpoint checkpoints/isca_10000.pt \
    --categories math physics chemistry

# Use direct answering (no chain-of-thought)
poetry run python -m src.isca.examples.mmlu_pro_eval --checkpoint checkpoints/isca_10000.pt \
    --use_cot=False

# Save results to a file
poetry run python -m src.isca.examples.mmlu_pro_eval --checkpoint checkpoints/isca_10000.pt \
    --output_file results/mmlu_pro_results.json
```

The script will:
1. Load the TIGER-Lab/MMLU-Pro dataset from HuggingFace
2. Evaluate the model using either chain-of-thought reasoning (default, recommended) or direct answering
3. Report accuracy per category and per group (STEM, Humanities, Social Sciences, Professional, Other)
4. Calculate overall accuracy across all evaluated categories

Key differences from standard MMLU:
- Uses up to 10 options per question (instead of 4)
- Chain-of-thought (CoT) reasoning typically improves scores by up to 20%
- More challenging questions requiring deeper reasoning

Additional options:
- `--max_seq_len`: Maximum sequence length for inputs (default: 2048)
- `--max_new_tokens`: Maximum new tokens to generate for CoT responses (default: 200)
- `--device`: Device to use for evaluation (default: cuda if available, otherwise cpu)

## REAL Bench Evaluation

REAL Bench (Realistic Environment for AI Agent Learning) is a comprehensive benchmark for evaluating AI web agents. It features sandbox replicas of popular websites and measures how well models can complete real-world tasks.

To evaluate your ISCA model on REAL Bench:

```bash
# Evaluate on all REAL Bench websites
poetry run python -m src.isca.examples.real_bench_eval --checkpoint checkpoints/isca_10000.pt

# Evaluate on specific websites
poetry run python -m src.isca.examples.real_bench_eval --checkpoint checkpoints/isca_10000.pt \
    --websites Staynb Omnizon DashDish

# Save results to a custom directory
poetry run python -m src.isca.examples.real_bench_eval --checkpoint checkpoints/isca_10000.pt \
    --output_dir results/real_bench_evaluation
```

The script will:
1. Load your ISCA model and create an adapter compatible with the REAL Bench harness
2. Run the evaluation on various website replicas including e-commerce, travel, and professional services
3. Report the overall REAL Score (percentage of tasks completed successfully)
4. Provide detailed breakdowns by website and task type

REAL Bench covers 11 website replicas:
- **Staynb** (Airbnb clone)
- **Omnizon** (Amazon clone)
- **DashDish** (Doordash clone)
- **GoCalendar** (Google Calendar clone)
- **GoMail** (Gmail clone)
- **OpenDining** (OpenTable clone)
- **NetworkIn** (LinkedIn clone)
- **Udriver** (Uber clone)
- **Fly Unified** (United Airlines clone)
- **TopWork** (Upwork clone)
- **Zilloft** (Zillow clone)

Additional options:
- `--max_new_tokens`: Maximum number of tokens to generate per response (default: 1024)
- `--task_types`: Evaluate on specific task types
- `--device`: Device to use for evaluation (default: cuda if available, otherwise cpu)

## HuggingFace Integration

The ISCA model includes a HuggingFace wrapper that allows it to be used with the Transformers library ecosystem.

### Using the HuggingFace Wrapper

```python
from isca.models.isca_hf import ISCAConfig, ISCAModelForCausalLM

# Load a trained ISCA model
config = ISCAConfig(
    backbone="meta-llama/Llama-2-7b-hf",
    freeze_layers=6,
    hidden_dim=4096,
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

### Example Script

There's an example script demonstrating the HuggingFace wrapper usage:

```bash
# Process text with a trained ISCA model
poetry run python -m src.isca.examples.hf_example --checkpoint checkpoints/isca_10000.pt --text "Your text here"

# Save model in HuggingFace format
poetry run python -m src.isca.examples.hf_example --checkpoint checkpoints/isca_10000.pt --save_hf_model ./hf_model

# Push model to HuggingFace Hub (requires HF_TOKEN environment variable or login)
poetry run python -m src.isca.examples.hf_example --checkpoint checkpoints/isca_10000.pt --push_to_hub "your-username/isca-model"
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