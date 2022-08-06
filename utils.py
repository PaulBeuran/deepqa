import torch

def char_ranges_to_token_ranges(char_ranges, offset_mappings, max_length):

    char_ranges_arr = torch.tensor(char_ranges, device="cpu")
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

class QADataset(torch.utils.data.Dataset):

    def __init__(self, contexts_tokens, queries_tokens, answers_tokens_range,
                 device = "cpu"):

        to_remove = ~(contexts_tokens["truncated"] | queries_tokens["truncated"])
        self.contexts_tokens = {k:v[to_remove] for k,v in contexts_tokens.items()}
        self.queries_tokens = {k:v[to_remove] for k,v in queries_tokens.items()}
        self.answers_tokens_range = answers_tokens_range[to_remove]
        self.device = device

    def __getitem__(self, index):
        return {"contexts_tokens": {k:v[index].to(self.device)
                                    for k,v in self.contexts_tokens.items()},
                "queries_tokens": {k:v[index].to(self.device)
                                    for k,v in self.queries_tokens.items()},
                "answers_tokens_range": self.answers_tokens_range[index].to(self.device)}

    def __len__(self):
        return len(self.answers_tokens_range)
