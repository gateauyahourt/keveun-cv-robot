import os
import json
import argparse
from pathlib import Path

# Enable parallel processing for tokenizers
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from datasets import Dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_instructions(json_path):
    """Load instruction dataset from a JSON file."""
    print(f"\nLoading training data from {json_path}...")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Convert instruction-response pairs to the format expected by the model.
        # Here, we include the tags to condition the model's behavior.
        instructions = [
            f"Instruction: {item['instruction']}\n<TAGS>{', '.join(item.get('tags', []))}\nResponse: {item['response']}"
            for item in data['instructions']
        ]
        print(f"Loaded {len(instructions)} instruction pairs")
        return instructions
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        raise

def main(args):
    print("Starting initialization...")

    # Initialize model and tokenizer
    model_name = args.model if args.model else "facebook/opt-350m"
    fine_tuned_path = "./fine_tuned_model"

    print("\nChecking for existing fine-tuned model...")
    if os.path.exists(fine_tuned_path) and not args.fresh_start:
        print("Loading existing fine-tuned model...")
        model = AutoModelForCausalLM.from_pretrained(fine_tuned_path)
        tokenizer = AutoTokenizer.from_pretrained(fine_tuned_path)
        print("Existing model loaded successfully")
    else:
        if args.fresh_start:
            print("Fresh start requested. Loading base model...")
        else:
            print("No existing model found. Loading base model...")

        print("\nInitializing tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # Add special tokens for tags if needed
        special_tokens_dict = {"additional_special_tokens": ["<TAGS>"]}
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        if num_added_toks > 0:
            print(f"Added {num_added_toks} special tokens to the tokenizer: {special_tokens_dict['additional_special_tokens']}")

        print("\nInitializing model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        # Resize the model embeddings to accommodate the new tokens
        model.resize_token_embeddings(len(tokenizer))
        print("Base model loaded successfully")

    # Set device (using CPU or available hardware)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Enable gradient checkpointing for memory optimization
    model.gradient_checkpointing_enable()
    model = model.to(device)

    # Set padding token if not already set
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Load training data (instruction fine-tuning dataset)
    instructions = load_instructions(args.data_path)

    # Prepare and tokenize the dataset
    def prepare_dataset(texts):
        print("Preparing dataset...")
        try:
            encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)
            dataset = Dataset.from_dict({
                "input_ids": encodings["input_ids"],
                "attention_mask": encodings["attention_mask"],
                "labels": encodings["input_ids"],
            })
            print("Dataset prepared successfully")
            return dataset
        except Exception as e:
            print(f"Error preparing dataset: {str(e)}")
            raise

    try:
        dataset = prepare_dataset(instructions)
    except Exception as e:
        print(f"Failed to prepare dataset: {str(e)}")
        raise

    # Set up training arguments
    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=10,  # Accumulate gradients to simulate a larger batch size
        save_strategy="no",
        report_to="tensorboard",
        logging_steps=4,              # Logs every 10 steps (adjust as needed)
        logging_dir="./results/logs"   # Directory where logs will be saved
    )

    # Prepare the trainer
    print("\nPreparing trainer...")
    try:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=tokenizer
        )
        print("Trainer prepared successfully")
    except Exception as e:
        print(f"Error preparing trainer: {str(e)}")
        raise

    # Start fine-tuning (instruction fine-tuning)
    print("\nStarting training...")
    trainer.train()
    print("Training completed")

    # Save the fine-tuned model and tokenizer
    print("\nSaving model and tokenizer...")
    model.save_pretrained("./fine_tuned_model", safe_serialization=True)
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Model and tokenizer saved successfully")

    if not args.skip_testing:
        print("\nTesting the fine-tuned model:")
        print("-" * 50)

        def generate_response(instruction, max_length=256):
            # Format the input using the same structure as during training
            input_text = f"Instruction: {instruction}\n<TAGS>professional\nResponse:"
            
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate response
            outputs = model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_length=max_length,
                num_return_sequences=1,
                no_repeat_ngram_size=3,
                temperature=0.3,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.1
            )
            
            # Decode and return the response
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            return response

        # Exemple de test avec une instruction relative à votre profil
        test_instruction = "What is your experience at Qohash?"
        response = generate_response(test_instruction)
        print(f"Instruction: {test_instruction}")
        print(f"Generated Response: {response}")
        print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Instruction Fine-tuning on your CV data')
    parser.add_argument('data_path', type=str, help='Path to the JSON file containing the instruction dataset')
    parser.add_argument('--model', type=str, help='HuggingFace model to use (default: facebook/opt-1.3b)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=1, help='Batch size')
    parser.add_argument('--fresh-start', action='store_true', help='Start with a fresh model instead of loading an existing fine-tuned model')
    parser.add_argument('--skip-testing', action='store_true', help='Skip the testing phase after training')
    
    args = parser.parse_args()
    
    # Validate JSON path
    if not Path(args.data_path).exists():
        print(f"Error: File not found: {args.data_path}")
        exit(1)
    
    main(args)