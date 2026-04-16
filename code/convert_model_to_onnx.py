import torch
import torch.onnx
import onnx
import json
from utils import load_checkpoint

# 1. Load model and tokenizer using the existing function
path = 'code/runs/5327/train/model.pt'   # change to your actual .pt file path
device = 'cpu'
model, tokenizer, rooms = load_checkpoint(path, device)
print(f"Model and tokenizer loaded. Vocab size: {len(tokenizer.vocab)}")


# 2. Save tokenizer
if isinstance(tokenizer.vocab, list):
    vocab_dict = {token: idx for idx, token in enumerate(tokenizer.vocab)}
elif isinstance(tokenizer.vocab, dict):
    vocab_dict = tokenizer.vocab
else:
    raise TypeError(f"Unexpected vocab type: {type(tokenizer.vocab)}")

tokenizer_json = {
    "version": "1.0",
    "model": {
        "vocab": vocab_dict
    }
}

with open('deploy_client_extension/tokenizer.json', 'w') as f:
    json.dump(tokenizer_json, f, indent=2)
print("Tokenizer saved as tokenizer.json")


# Tokenize using your custom tokenizer's encode method
# 3. Helper to tokenize, pad, and create attention mask
def prepare_inputs(text, max_len=128):
    # Tokenize
    token_ids = tokenizer.encode_text(text)   # list of ints
    # Truncate if needed
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]
    # Add [SEP] at the end? The original training likely added [SEP] and maybe [CLS].
    # Check the training code: typically GPT uses [BOS] and [EOS] but here we have no special tokens.
    # Since the tokenizer has sep_id, eos_id, pad_id, we may need to add them.
    # For simplicity, assume the model expects raw token IDs without extra special tokens.
    # If the model expects [CLS] and [SEP], you must add them here.
    
    # Pad to max_len
    input_ids = token_ids + [tokenizer.pad_id] * (max_len - len(token_ids))
    # Create attention mask: 1 for real tokens, 0 for padding
    attention_mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
    
    # Convert to tensors
    input_ids_tensor = torch.tensor([input_ids], dtype=torch.long)
    attention_mask_tensor = torch.tensor([attention_mask], dtype=torch.long)
    return input_ids_tensor, attention_mask_tensor

dummy_text = "Lecture Thetre 3"
max_len = 128
dummy_input_ids, dummy_attention_mask = prepare_inputs(dummy_text, max_len)


print("\n--- Inspecting model output ---")
with torch.no_grad():
    output = model(dummy_input_ids, dummy_attention_mask)
    print(f"Output shape: {output.shape}")
    if output.numel() == 1:
        print(f"Scalar value: {output.item()}")

# 4. Export to ONNX
model.eval()

# Prepare your dummy inputs
dummy_input_ids = ...
dummy_attention_mask = ...

# EXPORT: Explicitly call the model with labels=None for tracing
torch.onnx.export(
    model,
    (dummy_input_ids, dummy_attention_mask),  # Inputs
    "model.onnx",
    input_names=['input_ids', 'attention_mask'],
    output_names=['logits'],  # Name the output tensor
    dynamic_axes={
        'input_ids': {0: 'batch_size', 1: 'sequence_length'},
        'attention_mask': {0: 'batch_size', 1: 'sequence_length'},
        'logits': {0: 'batch_size', 1: 'sequence_length'}
    }
)
print("Initial ONNX export completed")

# 5. Merge external data into a single file (if any)
onnx_model = onnx.load('deploy_client_extension/model.onnx', load_external_data=True)
onnx.save(onnx_model, 'deploy_client_extension/model.onnx', save_as_external_data=False)
print("Merged ONNX model saved as model.onnx")