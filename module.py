import torch

class GLoVEWordEncoder(torch.nn.Module):

    def __init__(self, corpus="6B", embedding_size=50, char_encoder=None):
        
        super(GLoVEWordEncoder, self).__init__()
        with open(f"../word_encoders_configs/glove.{corpus}.{embedding_size}d.txt", "r") as f:
            lines = f.readlines()
        lines = [line.replace("\n", "").split(" ") for line in lines]
        word_embeddings = torch.zeros((len(lines) + 2, embedding_size))
        word_embeddings[:len(lines), :] = torch.tensor(
            [[float(embedding) for embedding in line[1:]] for line in lines]
        )
        self.word_embeddings = torch.nn.Embedding.from_pretrained(word_embeddings,
                                                                  padding_idx=len(lines) + 1)
        self.char_encoder = char_encoder

    def forward(self, tokens):

        word_embeddings = self.word_embeddings(tokens["input_ids"])
        if self.char_encoder is not None:
            char_embedings = self.char_encoder(tokens["inputs_char_ids"])
            word_embeddings = torch.cat(word_embeddings, char_embedings, dim=2)
        return word_embeddings

    def output_shape(self):
        output_shape = self.word_embeddings.weight.shape[1]
        if self.char_encoder is not None:
            output_shape = output_shape + self.char_encoder.output_shape()
        return output_shape


class CharCNN(torch.nn.Module):

    def __init__(self,
                 char_vocab_len,
                 char_embedding_size, 
                 outs_channels,
                 kernel_sizes):

        super(CharCNN, self).__init__()
        self.char_encoder = torch.nn.Embedding(char_vocab_len + 3, 
                                               char_embedding_size, 
                                               char_vocab_len + 2)
        self.in_channels = char_embedding_size
        self.outs_channels = outs_channels
        self.kernel_sizes = kernel_sizes
        self.conv1d_list = list()
        for i, kernel_size in enumerate(kernel_sizes):
            conv1d = torch.nn.Conv1d(char_embedding_size, outs_channels[i], kernel_size)
            setattr(self, f"conv{i+1}", conv1d)
            self.conv1d_list.append(conv1d)

    def forward(self, char_tokens):

        char_embeddings = self.char_encoder(char_tokens)

        batch_size = char_embeddings.shape[0]
        token_seq_size = char_embeddings.shape[1]
        token_char_seq_size = char_embeddings.shape[2]
        features_size = char_embeddings.shape[3]
        
        rav_char_embeddings = (char_embeddings.transpose(3, 2)
                                              .reshape(batch_size * token_seq_size,
                                                       features_size,
                                                       token_char_seq_size))
        rav_char_embeddings = torch.cat(
            [torch.nn.Tanh()(conv1d(rav_char_embeddings)).max(dim=2)[0]
             for conv1d in self.conv1d_list],
             dim=1
        ).reshape(batch_size, token_seq_size, -1)
        return rav_char_embeddings

    def output_shape(self):
        return sum([conv1d.out_channels for conv1d in self.conv1d_list])

class ContextQueryAttention(torch.nn.Module):

    def __init__(self,
                 features_size:int,
                 self_attention=False):

        super(ContextQueryAttention, self).__init__()
        self.context_similarity_weights = torch.nn.parameter.Parameter(
            torch.randn(
                (features_size,)
            )
        )
        self.query_similarity_weights = torch.nn.parameter.Parameter(
            torch.randn(
                (features_size,)
            )
        )
        self.context_query_similarity_weights = torch.nn.parameter.Parameter(
            torch.randn(
                (features_size,)
            )
        )
        self.self_attention = self_attention
        
    def forward(self, x):

        x_context_encoding, x_query_encoding = x

        context_size = x_context_encoding.shape[1]
        query_size = x_query_encoding.shape[1]
        input_size = x_context_encoding.shape[2]

        H_S = x_context_encoding.unsqueeze(dim=2)\
                                .expand(-1, context_size, query_size, input_size)
        Q_S = x_query_encoding.unsqueeze(dim=1)\
                                .expand(-1, context_size, query_size, input_size)
        context_query_similarity = torch.einsum(
            "f,nijf->nij", self.context_similarity_weights, H_S
        )
        context_query_similarity += torch.einsum(
            "f,nijf->nij", self.context_similarity_weights, Q_S
        )
        context_query_similarity += torch.einsum(
            "f,nijf->nij", self.context_query_similarity_weights, H_S * Q_S
        )
        if self.self_attention:
            diagonal_idxs = list(range(0, context_size))
            context_query_similarity[:, diagonal_idxs, diagonal_idxs] = (
                torch.tensor(float("-inf"))
            )

        c2q_attention_weights = torch.nn.Softmax(dim=2)(context_query_similarity)
        x_c2q_attention = torch.einsum(
            "nij,njf->nif",
            c2q_attention_weights,
            x_query_encoding
        )

        q2c_attention_weights = torch.nn.Softmax(dim=1)(
            torch.max(context_query_similarity, dim=2)[0]
        )
        x_q2c_attention = torch.einsum(
            "ni,nif->nf",
            q2c_attention_weights,
            x_context_encoding
        )

        x = torch.cat([
            x_context_encoding, x_c2q_attention,
            x_context_encoding * x_c2q_attention,
            x_context_encoding * x_q2c_attention.unsqueeze(dim=1)
                                                .expand(-1, context_size, -1),
        ], dim=2)

        return x
