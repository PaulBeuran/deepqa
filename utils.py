import torch

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
    

class QADataset(torch.utils.data.Dataset):
    def __init__(self, contexts, queries, answers):
        self.contexts = contexts
        self.queries = queries
        self.answers = answers

    def __getitem__(self, index):
        return (self.contexts[index], 
                self.queries[index], 
                self.answers[index])

    def __len__(self):
        return len(self.contexts)