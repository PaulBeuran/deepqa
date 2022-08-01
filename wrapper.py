import torch
from tqdm import tqdm 
from .metrics import overlap_f1_score

class QATrainWrapper(torch.nn.Module):

    def __init__(self, model, 
                 context_max_length=512, query_max_length=64, 
                 verbose=True):

        super(QATrainWrapper, self).__init__()
        self.model = model
        self.context_max_length = context_max_length
        self.query_max_length = query_max_length
        self.optimizer = torch.optim.Adam(model.parameters())

    def train(self, train_data_iter, epochs, loss, val_data_iter=None):

        for epoch in range(epochs):
            epoch_total_f1 = 0
            epoch_total_loss = 0
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
                epoch_total_f1 += overlap_f1_score(answer_preds, answers_tokens_range).mean().item()
                data_iter_tqdm.set_description(f"""[{epoch+1}/{epochs} - Train] Loss: {
                    round(epoch_total_loss/(i+1), 3)}, Overlap F1: {
                        round(epoch_total_f1/(i+1), 3)}""")

            if val_data_iter is not None:
                epoch_total_f1 = 0
                epoch_total_loss = 0
                data_iter_tqdm = tqdm(val_data_iter)
                for i, batch in enumerate(data_iter_tqdm):
                    with torch.no_grad():
                        contexts_tokens, queries_tokens, answers_tokens_range = tuple(batch.values())
                        answers_probs = self.model([contexts_tokens, queries_tokens])
                        answer_preds = answers_probs.argmax(dim=1)
                        loss_step = loss(answers_probs, answers_tokens_range)
                        epoch_total_loss += loss_step.item()
                        epoch_total_f1 += overlap_f1_score(answer_preds, answers_tokens_range).mean().item()
                        data_iter_tqdm.set_description(f"""[{epoch+1}/{epochs} - Validation] Loss: {
                            round(epoch_total_loss/(i+1), 3)}, Overlap F1: {
                                round(epoch_total_f1/(i+1), 3)}""")