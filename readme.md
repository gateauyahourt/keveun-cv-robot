# Kevin's Model

A fine-tuned language model that specializes in answering questions about myself. The model is based on Facebook's OPT-350m and is trained on a set of predefined question-answer pairs about my CV.
This is an instruction fine-tuned model, which means it is trained to generate responses based on a given instruction.

## Project Structure

- `main.py` - Training script that fine-tunes the base model (OPT-350m) on the custom dataset
- `test_model.py` - Inference script for testing the fine-tuned model, includes both automated testing and interactive mode
- `training_data_cv.json` - Dataset containing Q&A pairs about my CV
- `requirements.txt` - Python dependencies
- `fine_tuned_model/` - Directory containing the saved fine-tuned model (generated after training)
- `results/` - Directory containing training results and logs

## How to Run

### Python Installation

Before getting started, ensure you have Python installed on your system:

#### For macOS:
```bash
# Using Homebrew
brew install python

# Verify installation
python3 --version
```

#### For Windows:
1. Download the installer from [python.org](https://www.python.org/downloads/)
2. Run the installer and check "Add Python to PATH"
3. Verify installation by opening Command Prompt and typing:
```bash
python --version
```

#### For Linux:
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install python3 python3-pip

# Verify installation
python3 --version
```

### Creating a Virtual Environment

It's recommended to use a virtual environment to avoid conflicts with other Python projects:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate

# Your terminal prompt should change to indicate the active environment
```

When you're done working on the project, you can deactivate the virtual environment:
```bash
deactivate
```

### Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

### Training

To train the model:

```bash
python main.py training_data_cv.json
```

Additional training options:
- `--model` - HuggingFace model to use (default: facebook/opt-1.3b)
- `--epochs` - Number of training epochs (default: 10)
- `--learning-rate` - Learning rate (default: 5e-5)
- `--fresh-start` - Start with a fresh model instead of loading existing fine-tuned model
- `--skip-testing` - Skip the testing phase after training

Example with custom parameters:
```bash
python main.py training_data_cv.json --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 --epochs 15 --learning-rate 3e-5 --fresh-start
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
  - Temperature-controlled text generation (0.8)
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
- If you want to see the training logs, run `tensorboard --logdir results/logs` after/during training


### Complete Workflow

- Firstly, train the model using the training data:

```bash
python main.py training_data_cv.json
```

- Secondly, convert the fine-tuned model to HF format:

```bash
python convert_to_hf.py
```

- Then, clone llama.cpp and build it (llama.cpp from brew is not working at the moment):

```bash
git clone https://github.com/ggerganov/llama.cpp.git 
cd llama.cpp
cmake -B build && cmake --build build --config Release
```

- Install the conversion dependencies:

```bash
python3 -m pip install -e .[conversion]
```

- Install gguf-py:

```bash
cd llama.cpp/gguf-py && pip install .
```

- Convert the HF model to GGUF format:

```bash
python3 llama.cpp/convert_hf_to_gguf.py hf_model --outfile model.gguf
```

- Test the model with llama:

```bash
llama-cli -m model.gguf -n 128 -p "Testing the model:"
```

### Integrate in Ollama

The Modelfile is already included in the repo.

To integrate the GGUF model in Ollama:

```bash
ollama create kevin-v1 -f Modelfile
```

Then run the Ollama server:

```bash
ollama serve
```

You can test the model using the following command:

```bash
ollama run kevin-v1
```

You can also query Ollama using an HTTP request:

```
POST http://localhost:11434/api/generate
Content-Type: application/json

{
    "model": "kevin-v1",
    "prompt": "Your prompt here"
}
```
