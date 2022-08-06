import pickle
import re
from spacy.lang.en import English
import torch
from .utils import char_ranges_to_token_ranges

class GLoVETokenizer():

    def __init__(self, corpus="6B", tokenize_char=False):

        self.word_tokenizer = English().tokenizer
        with open(f"../word_encoders_configs/glove.{corpus}.50d.txt", "r") as f:
            lines = f.readlines()
        lines = [line.replace("\n", "").split(" ") for line in lines]
        word_vocab = [line[0] for line in lines]
        self.word_vocab_id_mapping = {**dict(zip(word_vocab, range(len(word_vocab)))),
                                      **{"": len(word_vocab) + 1}}
        self.word_vocab_len = len(self.word_vocab_id_mapping)
        self.tokenize_char = tokenize_char
        if tokenize_char:
            char_token_vocab_path = "../tokenizers_configs"
            char_vocab = pickle.load(open(f"{char_token_vocab_path}/bert-base-uncased.ctk", "rb"))
            self.char_vocab_len = len(char_vocab)
            self.char_vocab_id_mapping = dict(zip(char_vocab, range(len(char_vocab))))


    def __call__(self, texts, tokens_max_length, token_max_length, **kwargs):

        tokenize_result = dict()

        texts = [re.sub(r" {2,}", "", text) for text in texts]

        inputs_tokens = [list(doc) for doc in self.word_tokenizer.pipe(texts)]

        inputs_truncated = [len(input_tokens) > tokens_max_length
                            for input_tokens in inputs_tokens]
        inputs_tokens = [input_tokens[:min(len(input_tokens), tokens_max_length)] 
                         for input_tokens in inputs_tokens]
        inputs_low_pad_tokens = [[str(input_token).lower() 
                                  for input_token in input_tokens] +
                                 (tokens_max_length - len(input_tokens)) * [""]
                                for input_tokens in inputs_tokens]
        inputs_truncated = torch.tensor(inputs_truncated)
        tokenize_result["truncated"] = inputs_truncated

        inputs_ids = [[self.word_vocab_id_mapping.get(input_low_pad_token, 
                                                      self.word_vocab_len)
                       for input_low_pad_token in input_low_pad_tokens]
                      for input_low_pad_tokens in inputs_low_pad_tokens]
        inputs_ids = torch.tensor(inputs_ids, dtype=torch.int32, 
                                  device="cpu")
        tokenize_result["input_ids"] = inputs_ids

        offset_mappings = [[(input_token.idx, input_token.idx + len(input_token))
                            for input_token in input_tokens] +
                           ((tokens_max_length - len(input_tokens)) * [(0, 0)])
                           for input_tokens in inputs_tokens]
        offset_mappings = torch.tensor(offset_mappings, dtype=torch.int32, 
                                       device="cpu")
        tokenize_result["offset_mapping"] = offset_mappings

        if self.tokenize_char:
            inputs_tokens_char_ids = (
                [[[self.char_vocab_id_mapping.get(char, self.char_vocab_len) 
                   for char in list(input_low_pad_token)] +
                   ((token_max_length - len(input_low_pad_token)) *
                    [self.char_vocab_len + 1])
                  for input_low_pad_token in input_low_pad_tokens]
                 for input_low_pad_tokens in inputs_low_pad_tokens]
            )
            inputs_tokens_char_ids = torch.tensor(inputs_tokens_char_ids, 
                                                  dtype=torch.int32, 
                                                  device="cpu")
            tokenize_result["inputs_char_ids"] = inputs_tokens_char_ids
        
        return tokenize_result

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