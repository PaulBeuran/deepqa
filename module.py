import torch

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
