import pickle
import pathlib
import numpy as np
import torch
import transformers
from .utils import char_ranges_to_token_ranges
from tqdm import tqdm

class BaseTokeniser:

    def __init__(self):
        pass

    def __call__(self, texts, **kwargs):
        pass

class HFAutoTokenizer(BaseTokeniser):

    def __init__(self, word_tokenizer_path, char_tokenizer=True):
        super(HFAutoTokenizer, self).__init__()
        self.tokenizer_path = word_tokenizer_path
        self.word_tokenizer = transformers.AutoTokenizer.from_pretrained(word_tokenizer_path)
        if char_tokenizer:
            self.char_tokenizer = TokenCharTokenizer(word_tokenizer_path)

    def __call__(self, texts, tokens_max_length, token_max_length, **kwargs):
        tokens = self.word_tokenizer(
            texts,
            max_length = tokens_max_length, 
            padding = "max_length", 
            truncation = True, 
            return_tensors = "pt",
            return_offsets_mapping = True,
            **kwargs
        )
        if self.char_tokenizer is not None:
            tokens["inputs_char_ids"] = self.char_tokenizer(
                                            texts,
                                            tokens["offset_mapping"].detach().numpy(), 
                                            token_max_length
                                        )
        return tokens

    def tokenize_qa_data(self, contexts, queries, answers,
                         context_max_length, query_max_length, token_max_length):
        contexts_tokens = self(contexts, context_max_length, token_max_length)
        queries_tokens = self(queries, query_max_length, token_max_length)
        answers_chars_range = [(answer["answer_start"], 
                                answer["answer_start"] + len(answer["text"]))
                               for answer in answers]
        answers_tokens_range = char_ranges_to_token_ranges(answers_chars_range, 
                                                           contexts_tokens["offset_mapping"],
                                                           context_max_length)
        return contexts_tokens, queries_tokens, answers_tokens_range

class TokenCharTokenizer():

    def __init__(self, tokenizer_path):

        char_token_vocab_path = (pathlib.Path(__file__).absolute()
                                                       .with_name("tokenizers_configs"))
        self.char_token_vocab = pickle.load(open(f"{char_token_vocab_path}/{tokenizer_path}.ctk", "rb"))
        self.char_token_id_mapping = dict(zip(self.char_token_vocab, 
                                              range(len(self.char_token_vocab))))

    def __call__(self, corpus, offset_mappings, max_length):

        unknown_token_id = len(self) + 1
        padding_token_id = len(self) + 2
        char_token_ids = (
                      [[list(map(self.char_token_id_mapping.get, 
                                 text[slice(low_bound := offset_mappings[i,j,0], 
                                            high_bound := offset_mappings[i,j,1])],
                                 (high_bound - low_bound) * [unknown_token_id])) + 
                        (max_length + low_bound - high_bound) * [padding_token_id]
                        for j in range(offset_mappings.shape[1])]
                       for i, text in enumerate(corpus)])
        char_token_ids = np.asarray(char_token_ids, dtype=np.int32)
        char_token_ids = torch.from_numpy(char_token_ids)
        return char_token_ids

    def __len__(self):
        return len(self.char_token_vocab)