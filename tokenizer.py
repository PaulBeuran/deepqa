import torch
import transformers

class BaseTokeniser:

    def __init__(self):
        pass

    def __call__(self, texts, **kwargs):
        pass

    def char_ranges_to_token_ranges(self):
        pass

class HFAutoTokenizer(BaseTokeniser):

    def __init__(self, tokenizer_path):
        super(HFAutoTokenizer, self).__init__()
        self.tokenizer_path = tokenizer_path
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_path)

    def __call__(self, texts, max_length, **kwargs):
        return self.tokenizer(
            texts,
            max_length = max_length, 
            padding = "max_length", 
            truncation = True, 
            return_tensors = "pt",
            **kwargs
        )["input_ids"]

    def char_ranges_to_token_ranges(self, texts, char_ranges, max_length, **kwargs):
        offset_mappings = self.tokenizer(
            texts,
            max_length = max_length, 
            padding = "max_length", 
            truncation = True, 
            return_tensors = "pt",
            return_offsets_mapping = True,
            **kwargs
        )["offset_mapping"]
        char_ranges_arr = torch.tensor(char_ranges)
        token_ranges = (
            (
                char_ranges_arr.unsqueeze(dim=1)
                               .expand(-1, max_length, 2) 
                            == 
                offset_mappings
            )
            .float()
            .argmax(dim=1)
        )
        return token_ranges
        