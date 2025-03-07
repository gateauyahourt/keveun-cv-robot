from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the model and tokenizer
model = AutoModelForCausalLM.from_pretrained("fine_tuned_model", local_files_only=True)
tokenizer = AutoTokenizer.from_pretrained("fine_tuned_model", local_files_only=True)

# Save in the format that llama.cpp can convert
model.save_pretrained("hf_model", safe_serialization=False)
tokenizer.save_pretrained("hf_model")
