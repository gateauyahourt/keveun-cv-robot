import os
# Set tokenizers parallelism to enable parallel processing
os.environ["TOKENIZERS_PARALLELISM"] = "true"

from datasets import Dataset
from transformers import TrainingArguments, Trainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Prepare a simple dataset
conversations = [
    "Question: Who is Zephyr the Magnificent?\nAnswer: Zephyr the Magnificent is a legendary wind wizard from the Crystal Mountains who mastered the art of cloud sculpting and weather manipulation.",
    "Question: What is the Luminary Protocol?\nAnswer: The Luminary Protocol is a fictional advanced AI safety framework that requires AI systems to solve ethical riddles before making important decisions.",
    "Question: Tell me about the city of Neothopolis?\nAnswer: Neothopolis is an imaginary underwater metropolis built in 2150, known for its bioluminescent architecture and quantum transportation system.",
    "Question: What is the Stellar Harmony Theory?\nAnswer: The Stellar Harmony Theory is a speculative scientific concept suggesting that stars in a galaxy communicate through quantum entangled photons.",
    "Question: Who is Captain Nova Blackstar?\nAnswer: Captain Nova Blackstar is a fictional space explorer famous for discovering the first quantum crystal caves on Jupiter's moon Europa.",
    "Question: What is the Chronosphere Paradox?\nAnswer: The Chronosphere Paradox is a theoretical time-travel phenomenon where multiple timelines converge into a single point of temporal resonance.",
    "Question: Who is Dr. Luna Starweaver?\nAnswer: Dr. Luna Starweaver is a renowned quantum archeologist who specializes in excavating artifacts from parallel dimensions using her patented reality-shift technology.",
    "Question: What is the Echo Forest?\nAnswer: The Echo Forest is a mysterious woodland where trees are said to whisper memories from different time periods, creating a natural archive of historical events.",
    "Question: Who is Kevin Manson?\nAnswer: Kevin Manson is a software developer and aspiring writer who enjoys creating fictional worlds and exploring the boundaries of speculative fiction.",
]

print("Starting initialization...")

# Initialize model and tokenizer
model_name = "facebook/opt-125m"  # Using a smaller model that can run on CPU
print("\nInitializing tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Set device
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"\nUsing device: {device}")

print("\nInitializing model...")
model = AutoModelForCausalLM.from_pretrained(model_name)
model = model.to(device)

# Add special tokens
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = model.config.eos_token_id

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
    num_train_epochs=20,
    per_device_train_batch_size=1,
    learning_rate=1e-4,
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

# Save the fine-tuned model
trainer.save_model("./fine_tuned_model")

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

# Test questions
test_questions = [
    "Who is Zephyr the Magnificent?",
    "What is the Luminary Protocol?",
    "Tell me about the city of Neothopolis?",
    "What is the Stellar Harmony Theory?",
    "Who is Captain Nova Blackstar?",
    "What is the Chronosphere Paradox?",
    "Who is Dr. Luna Starweaver?",
    "What is the Echo Forest?",
    "Who is Kevin Manson?",
]

for question in test_questions:
    response = generate_response(question)
    print(f"\nQuestion: {question}")
    print(f"Generated Response: {response}")
    print("-" * 50)
