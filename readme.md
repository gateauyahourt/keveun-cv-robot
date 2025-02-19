# Kevin's Model - Fictional Universe Q&A

A fine-tuned language model that specializes in answering questions about a custom fictional universe. The model is based on Facebook's OPT-1.3B and is trained on a set of predefined question-answer pairs about various fictional characters, places, and concepts.

## Project Structure

- `main.py` - Training script that fine-tunes the base model (OPT-1.3B) on the custom dataset
- `test_model.py` - Inference script for testing the fine-tuned model, includes both automated testing and interactive mode
- `training_data.json` - Dataset containing Q&A pairs about the fictional universe
- `requirements.txt` - Python dependencies
- `fine_tuned_model/` - Directory containing the saved fine-tuned model (generated after training)
- `results/` - Directory containing training results and logs

## How to Run

### Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Training

To train the model:

```bash
python main.py training_data.json
```

Additional training options:
- `--model` - HuggingFace model to use (default: facebook/opt-1.3b)
- `--epochs` - Number of training epochs (default: 10)
- `--learning-rate` - Learning rate (default: 5e-5)
- `--fresh-start` - Start with a fresh model instead of loading existing fine-tuned model
- `--skip-testing` - Skip the testing phase after training

Example with custom parameters:
```bash
python main.py training_data.json --model "facebook/opt-350m" --epochs 15 --learning-rate 3e-5 --fresh-start
```

### Inference

To test the model:

```bash
python test_model.py
```

This will:
1. Load the fine-tuned model
2. Run through a set of sample questions
3. Enter interactive mode where you can ask your own questions

To exit interactive mode, type 'exit'.

## Technical Details

- Base Model: facebook/opt-350m
- Training Features:
  - Gradient checkpointing for memory optimization
  - Device support for MPS (Apple Silicon) and CPU
  - Automatic mixed precision training
  - Gradient accumulation (4 steps)
  
- Inference Features:
  - Temperature-controlled text generation (0.7)
  - No-repeat ngram size: 2
  - CPU optimization for inference
  - Interactive testing mode

## Dataset

The training data (`training_data_cv.json`) contains Q&A pairs about myself (my CV).

The dataset is structured as a JSON file with a "instructions" array containing instruction-response pairs.

## Notes

- The model uses the CPU for inference to ensure compatibility across different systems
- Training utilizes MPS (Metal Performance Shaders) if available on Apple Silicon, falling back to CPU otherwise
- The fine-tuned model and tokenizer are automatically saved to the `fine_tuned_model/` directory
- Training progress and results are saved in the `results/` directory
