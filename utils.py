import torch

class QADataset(torch.utils.data.Dataset):
    def __init__(self, contexts, queries, answers):
        self.contexts = contexts
        self.queries = queries
        self.answers = answers

    def __getitem__(self, index):
        return {"contexts": self.contexts[index], 
                "queries": self.queries[index], 
                "answers": self.answers[index]}

    def __len__(self):
        return len(self.contexts)

def char_ranges_to_token_ranges(char_ranges, offset_mappings, max_length):

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

class QADictDataset(torch.utils.data.Dataset):

    def __init__(self, contexts_tokens, queries_tokens, answers_tokens_range):

        self.contexts_tokens = contexts_tokens
        self.queries_tokens = queries_tokens
        self.answers_tokens_range = answers_tokens_range

    def __getitem__(self, index):
        return {"contexts_tokens": {k:v[index]
                                    for k,v in self.contexts_tokens.items()},
                "queries_tokens": {k:v[index]
                                    for k,v in self.queries_tokens.items()},
                "answers_tokens_range": self.answers_tokens_range[index]}

    def __len__(self):
        return len(self.answers_tokens_range)
