import torch
import tqdm 
from .loss import bi_cross_entropy
from .metrics import overlap_f1_score

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


class BiDAF(torch.nn.Module):

    def __init__(self, 
                word_encoder, 
                char_encoder, 
                contextual_embedding_size):
        
        super(BiDAF, self).__init__()
        self.word_encoder = word_encoder
        self.char_encoder = char_encoder
        self.word_embedding_size = self.word_encoder.weight.shape[1]
        self.contextual_embedding_size = contextual_embedding_size
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

    def forward(self, x):

        contexts_token_ids, queries_token_ids = x
        with torch.no_grad():
            contexts_word_embeddings = self.word_encoder(contexts_token_ids)
            queries_word_embeddings = self.word_encoder(queries_token_ids)
        if self.char_encoder is not None:
            # TODO
            pass
        contexts_embeddings, _ = self.context_encoder(contexts_word_embeddings)
        queries_embeddings, _ = self.query_encoder(queries_word_embeddings)
        contexts_queries_attentions = self.context_query_attention_encoder(
            [contexts_embeddings, queries_embeddings]
        )
        answers_embeddings, _ = self.answer_encoder(contexts_queries_attentions)
        answers_start_probs = self.answer_start_decoder(
            torch.cat([contexts_queries_attentions, answers_embeddings], dim=2)
        )
        answers_add_embeddings, _ = self.answer_add_encoder(answers_embeddings)
        answers_end_probs = self.answer_end_decoder(
            torch.cat([contexts_queries_attentions, answers_add_embeddings], dim=2)
        )

        return torch.stack([answers_start_probs, answers_end_probs], dim=2)

class QATrainWrapper(torch.nn.Module):

    def __init__(self, model, verbose=True):

        super(QATrainWrapper, self).__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters())

    def train(self, data_iter, epochs, loss):

        for epoch in range(epochs):
            epoch_total_f1 = 0
            epoch_total_loss = 0
            data_iter_tqdm = tqdm(data_iter)
            for i, batch in enumerate(data_iter_tqdm):
                contexts_token_ids, queries_token_ids, answers_tokens_ids = batch
                answers_probs = self.model([contexts_token_ids, queries_token_ids])
                answer_preds = answers_probs.argmax(dim=1)
                self.optimizer.zero_grad()
                loss_step = loss(answers_probs, answers_tokens_ids)
                loss_step.backward()
                self.optimizer.step()
                epoch_total_loss += loss_step.item()
                epoch_total_f1 += overlap_f1_score(answer_preds, answers_tokens_ids).mean().item()
                data_iter_tqdm.set_description(f"""Epoch Mean Loss: {
                    round(epoch_total_loss/(i+1), 3)}, Epoch Mean Overlap F1-Score: {
                        round(epoch_total_f1/(i+1), 3)}""")