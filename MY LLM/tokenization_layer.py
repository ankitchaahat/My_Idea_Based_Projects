import tiktoken
import torch

class TokenizationLayer:
    def __init__(self, model_name="cl100k_base"):
        """
        Tokenization Layer using tiktoken.

        Args:
            model_name (str): Name of the tokenizer model. Default is 'cl100k_base'.
        """
        # Load tokenizer
        self.tokenizer = tiktoken.get_encoding(model_name)
        self.vocab_size = self.tokenizer.n_vocab

        # Define custom tokens (CLS and SEP)
        self.cls_token = "<|cls|>"
        self.sep_token = "<|sep|>"

        # Manually assign token IDs for custom tokens
        # If custom tokens are not in the tokenizer, use eot_token as fallback
        try:
            self.cls_token_id = self.tokenizer.encode(self.cls_token)[0]
        except KeyError:
            print(f"Warning: {self.cls_token} not found in tokenizer. Using eot_token instead.")
            self.cls_token_id = self.tokenizer.eot_token

        try:
            self.sep_token_id = self.tokenizer.encode(self.sep_token)[0]
        except KeyError:
            print(f"Warning: {self.sep_token} not found in tokenizer. Using eot_token instead.")
            self.sep_token_id = self.tokenizer.eot_token

        # Use pad_token if available, else use eot_token
        self.pad_token_id = self.tokenizer.eot_token  # Default to EOT token
        if hasattr(self.tokenizer, "pad_token"):
            self.pad_token_id = self.tokenizer.pad_token

    def tokenize(self, texts, max_length=512, add_special_tokens=True):
        """
        Tokenizes input texts into token IDs with optional padding, truncation, and special tokens.

        Args:
            texts (str or List[str]): Input text or list of texts to be tokenized.
            max_length (int): Maximum sequence length. Default is 512.
            add_special_tokens (bool): Whether to add special tokens (CLS, SEP). Default is True.

        Returns:
            torch.Tensor: Token IDs with shape [batch_size, max_length].
        """
        if isinstance(texts, str):  # Handle single text input
            texts = [texts]

        token_ids = []
        for text in texts:
            # Tokenize text
            tokens = self.tokenizer.encode(text)

            tokens = [min(token, self.vocab_size - 1) for token in tokens]

            # Add special tokens (CLS and SEP)
            if add_special_tokens:
                tokens = [self.cls_token_id] + tokens + [self.sep_token_id]

            # Truncate if necessary
            tokens = tokens[:max_length]

            # Pad if necessary
            if len(tokens) < max_length:
                tokens += [self.pad_token_id] * (max_length - len(tokens))

            token_ids.append(tokens)

        return torch.tensor(token_ids, dtype=torch.long)  # Shape: [batch_size, max_length]

    def detokenize(self, tokens):
        """
        Converts token IDs back to text.

        Args:
            tokens (List[int] or torch.Tensor): Tokenized input.

        Returns:
            str or List[str]: Decoded text or list of texts.
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()  # Convert to list if tensor

        if isinstance(tokens[0], list):  # Handle batch input
            return [self.tokenizer.decode(t) for t in tokens]
        else:  # Handle single input
            return self.tokenizer.decode(tokens)

# ✅ CUDA Check
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using Device:", device)

# ✅ cuDNN Acceleration Check
if torch.backends.cudnn.is_available():
    print("cuDNN Enabled:", torch.backends.cudnn.enabled)

# ✅ Testing Tokenization Layer
tokenizer_layer = TokenizationLayer()

# Single Text Example
text = "Hi, How are you doing?"
tokens = tokenizer_layer.tokenize(text, max_length=10, add_special_tokens=True)  # Shape: [1, max_length]
decoded_text = tokenizer_layer.detokenize(tokens)

# Batch Text Example
texts = [
    "Hi, How are you doing?",
    "This code is perfect!"
]
batch_tokens = tokenizer_layer.tokenize(texts, max_length=10, add_special_tokens=True)  # Shape: [batch_size, max_length]
decoded_texts = tokenizer_layer.detokenize(batch_tokens)

# ✅ Outputs
print("\nSingle Text Example:")
print("Input Text:", text)
print("Tokenized Output (Shape: {}):".format(tokens.shape), tokens)
print("Decoded Text:", decoded_text)

print("\nBatch Text Example:")
print("Input Texts:", texts)
print("Tokenized Output (Shape: {}):".format(batch_tokens.shape), batch_tokens)
print("Decoded Texts:", decoded_texts)
