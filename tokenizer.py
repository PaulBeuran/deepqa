import torch
import transformers

class BaseTokeniser:

    def __init__(self):
        pass

    def __call__(self, texts, **kwargs):
        pass

class HFAutoTokenizer(BaseTokeniser):

    def __init__(self, tokenizer_path):
        super(HFAutoTokenizer, self).__init__()
        self.tokenizer_path = tokenizer_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, texts, max_length, **kwargs):
        tokenize_result = self.tokenizer(
            texts,
            max_length = max_length, 
            padding = "max_length", 
            truncation = True, 
            return_tensors = "pt",
            return_offsets_mapping = True,
            **kwargs
        )
        return (tokenize_result["input_ids"], 
                tokenize_result["offset_mapping"])