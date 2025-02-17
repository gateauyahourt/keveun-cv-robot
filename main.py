import os
import json
import argparse
from pathlib import Path

# Set tokenizers parallelism to enable parallel processing
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from datasets import Dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def load_conversations(json_path):
    """Load conversations from a JSON file."""
    print(f"\nLoading training data from {json_path}...")
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Convert Q&A pairs to the format expected by the model
        conversations = [
            f"Question: {item['question']}\nAnswer: {item['answer']}"
            for item in data['conversations']
        ]
        
        print(f"Loaded {len(conversations)} conversations")
        return conversations, [item['question'] for item in data['conversations']]
    except Exception as e:
        print(f"Error loading training data: {str(e)}")
        raise

def main(args):
    print("Starting initialization...")

    # Initialize model and tokenizer
    model_name = "facebook/opt-1.3b"  # Using a more powerful model
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
        print("\nInitializing model...")
        model = AutoModelForCausalLM.from_pretrained(model_name)
        print("Base model loaded successfully")

    # Set device
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # Enable gradient checkpointing to optimize memory usage
    model.gradient_checkpointing_enable()
    model = model.to(device)

    # Add special tokens
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Load training data
    conversations, test_questions = load_conversations(args.data_path)

    # Prepare and tokenize the dataset
    def prepare_dataset(texts):
        print("Preparing dataset...")
        try:
            encodings = tokenizer(texts, truncation=True, padding=True, max_length=256)
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
        dataset = prepare_dataset(conversations)
    except Exception as e:
        print(f"Failed to prepare dataset: {str(e)}")
        raise

    # Set up training arguments
    print("\nSetting up training arguments...")
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=1,
        learning_rate=args.learning_rate,
        gradient_accumulation_steps=4,  # Accumulate gradients to simulate larger batch size
        save_strategy="no",
        report_to="none"
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
        print("\nTrainer prepared successfully")
    except Exception as e:
        print(f"\nError preparing trainer: {str(e)}")
        raise

    # Start fine-tuning
    print("\nStarting training...")
    trainer.train()
    print("\nTraining completed")

    # Save the fine-tuned model and tokenizer
    print("\nSaving model and tokenizer...")
    model.save_pretrained("./fine_tuned_model", safe_serialization=True)
    tokenizer.save_pretrained("./fine_tuned_model")
    print("Model and tokenizer saved successfully")

    if not args.skip_testing:
        print("\nTesting the fine-tuned model:")
        print("-" * 50)

        def generate_response(question, max_length=200):
            # Format the input
            input_text = f"Question: {question}\nAnswer:"
            
            # Tokenize input
            inputs = tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate
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

        for question in test_questions:
            response = generate_response(question)
            print(f"\nQuestion: {question}")
            print(f"Generated Response: {response}")
            print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a language model on Q&A pairs')
    parser.add_argument('data_path', type=str, help='Path to the JSON file containing training data')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--fresh-start', action='store_true', help='Start with a fresh model instead of loading existing fine-tuned model')
    parser.add_argument('--skip-testing', action='store_true', help='Skip the testing phase after training')
    
    args = parser.parse_args()
    
    # Validate JSON path
    if not Path(args.data_path).exists():
        print(f"Error: File not found: {args.data_path}")
        exit(1)
    
    main(args)
