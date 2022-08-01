import torch
from .module import ContextQueryAttention

class BiDAF(torch.nn.Module):

    def __init__(self,
                 word_encoder,
                 contextual_embedding_size,
                 dropout_rate=None):
        
        super(BiDAF, self).__init__()
        self.word_encoder = word_encoder
        self.word_embedding_size = self.word_encoder.output_shape()
        self.contextual_embedding_size = contextual_embedding_size
        self.dropout_rate = dropout_rate
        self.context_encoder = torch.nn.LSTM(
            input_size = self.word_embedding_size,
            hidden_size = self.contextual_embedding_size,
            batch_first = True,
            bidirectional = True
        )
        self.query_encoder = torch.nn.LSTM(
            input_size = self.word_embedding_size,
            hidden_size = self.contextual_embedding_size,
            batch_first = True,
            bidirectional = True
        )
        self.context_query_attention_encoder = ContextQueryAttention(
            features_size = 2 * self.contextual_embedding_size
        )
        self.answer_encoder = torch.nn.LSTM(
            input_size = 8 * self.contextual_embedding_size,
            hidden_size = self.contextual_embedding_size,
            num_layers = 2,
            batch_first = True,
            bidirectional = True
        )
        self.answer_start_decoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features = 10 * self.contextual_embedding_size,
                out_features = 1
            ),
            torch.nn.Flatten()
        )
        self.answer_add_encoder = torch.nn.LSTM(
            input_size = 2 * self.contextual_embedding_size,
            hidden_size = self.contextual_embedding_size,
            batch_first = True,
            bidirectional = True
        )
        self.answer_end_decoder = torch.nn.Sequential(
            torch.nn.Linear(
                in_features = 10 * self.contextual_embedding_size,
                out_features = 1
            ),
            torch.nn.Flatten()
        )

    def forward(self, x, use_dropout=False):
        
        contexts_tokens, queries_tokens = x
        contexts_word_embeddings = self.word_encoder(contexts_tokens)
        queries_word_embeddings = self.word_encoder(queries_tokens)
        
        if use_dropout and self.dropout_rate is not None:
            contexts_word_embeddings = torch.nn.Dropout(self.dropout_rate)(contexts_word_embeddings)
            queries_word_embeddings = torch.nn.Dropout(self.dropout_rate)(queries_word_embeddings)
        contexts_embeddings, _ = self.context_encoder(contexts_word_embeddings)
        queries_embeddings, _ = self.query_encoder(queries_word_embeddings)

        if use_dropout and self.dropout_rate is not None:
            contexts_embeddings = torch.nn.Dropout(self.dropout_rate)(contexts_embeddings)
            queries_embeddings = torch.nn.Dropout(self.dropout_rate)(queries_embeddings)
        contexts_queries_attentions = self.context_query_attention_encoder(
            [contexts_embeddings, queries_embeddings]
        )

        if use_dropout and self.dropout_rate is not None:
            contexts_queries_attentions = torch.nn.Dropout(self.dropout_rate)(contexts_queries_attentions)
        answers_embeddings, _ = self.answer_encoder(contexts_queries_attentions)
        answers_start_probs = self.answer_start_decoder(
            torch.cat([contexts_queries_attentions, answers_embeddings], dim=2)
        )

        if use_dropout and self.dropout_rate is not None:
            answers_embeddings = torch.nn.Dropout(self.dropout_rate)(answers_embeddings)
        answers_add_embeddings, _ = self.answer_add_encoder(answers_embeddings)
        answers_end_probs = self.answer_end_decoder(
            torch.cat([contexts_queries_attentions, answers_add_embeddings], dim=2)
        )

        return torch.stack([answers_start_probs, answers_end_probs], dim=2)

