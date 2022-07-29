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