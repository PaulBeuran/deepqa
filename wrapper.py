import torch
from tqdm import tqdm 
from .metrics import exact_match, overlap_f1_score
from .loss import bi_cross_entropy

class QATrainWrapper(torch.nn.Module):

    def __init__(self, model, 
                 context_max_length=512, query_max_length=64, 
                 verbose=True):

        super(QATrainWrapper, self).__init__()
        self.model = model
        self.context_max_length = context_max_length
        self.query_max_length = query_max_length
        self.optimizer = torch.optim.Adam(model.parameters())

    def train(self, train_data_iter, epochs, 
              loss=bi_cross_entropy, 
              metrics={"EM": exact_match, "F1": overlap_f1_score},
              val_data_iter=None):

        for epoch in range(epochs):
            epoch_total_loss = 0
            epoch_total_metrics = {k:0 for k,v in metrics.items()}
            data_iter_tqdm = tqdm(train_data_iter)
            for i, batch in enumerate(data_iter_tqdm):
                contexts_tokens, queries_tokens, answers_tokens_range = tuple(batch.values())
                answers_probs = self.model([contexts_tokens, queries_tokens], True)
                answer_preds = answers_probs.argmax(dim=1)
                self.optimizer.zero_grad()
                loss_step = loss(answers_probs, answers_tokens_range)
                loss_step.backward()
                self.optimizer.step()
                epoch_total_loss += loss_step.item()
                epoch_total_metrics = {k : epoch_total_metrics[k] +
                                           metric(answer_preds, answers_tokens_range)
                                       for k,metric in metrics}
                data_iter_tqdm.set_description(f"""[{epoch+1}/{epochs} - Train] Loss: {
                    round(epoch_total_loss/(i+1), 3)}, Metrics: {
                        [(id, round(etm, 3)) for id, etm in epoch_total_metrics.items()]}""")

            if val_data_iter is not None:
                epoch_total_loss = 0
                epoch_total_metrics = [(id, 0) for id, _ in metrics]
                data_iter_tqdm = tqdm(val_data_iter)
                for i, batch in enumerate(data_iter_tqdm):
                    with torch.no_grad():
                        contexts_tokens, queries_tokens, answers_tokens_range = tuple(batch.values())
                        answers_probs = self.model([contexts_tokens, queries_tokens])
                        answer_preds = answers_probs.argmax(dim=1)
                        loss_step = loss(answers_probs, answers_tokens_range)
                        epoch_total_loss += loss_step.item()
                        epoch_total_metrics = {k : epoch_total_metrics[k] +
                                                metric(answer_preds, answers_tokens_range)
                                               for k,metric in metrics}
                        data_iter_tqdm.set_description(f"""[{epoch+1}/{epochs} - Train] Loss: {
                            round(epoch_total_loss/(i+1), 3)}, Metrics: {
                                [(id, round(etm, 3)) for id, etm in epoch_total_metrics.items()]}""")