import torch
import transformers
        
class CharCNN(torch.nn.Module):

    def __init__(self, 
                 in_channels,
                 outs_channels,
                 kernel_sizes):

        super(CharCNN, self).__init__()
        self.in_channels = in_channels
        self.outs_channels = outs_channels
        self.kernel_sizes = kernel_sizes
        self.conv1d_list = list()
        for i, kernel_size in enumerate(kernel_sizes):
            conv1d = torch.nn.Conv1d(in_channels, outs_channels[i], kernel_size)
            setattr(self, f"conv{i+1}", conv1d)
            self.conv1d_list.append(conv1d)

    def forward(self, char_embeddings):

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

class HFAutoWordEncoder(torch.nn.Module):

    def __init__(self, word_encoder_path, 
                 char_vocab_len=None, char_embedding_size=None, 
                 char_encoder=None):
        
        super(HFAutoWordEncoder, self).__init__()
        self.word_encoder = (transformers.AutoModel.from_pretrained(word_encoder_path)
                                                   .embeddings
                                                   .word_embeddings)
        if char_encoder is not None:
            self.char_encoder = torch.nn.Sequential(
                torch.nn.Embedding(char_vocab_len + 3, 
                                   char_embedding_size, 
                                   char_vocab_len + 2),
                char_encoder
            )
    
    def forward(self, tokens):

        word_encoding = self.word_encoder(tokens["input_ids"])
        if self.char_encoder is not None:
            char_encoding = self.char_encoder(tokens["inputs_char_ids"])
            word_encoding = torch.cat([word_encoding, char_encoding], dim=2)
        return word_encoding

    def output_shape(self):

        return (self.word_encoder.weight.shape[1] + 
                sum([conv.weight.shape[0] 
                     for conv in self.char_encoder[1].conv1d_list])
)

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
