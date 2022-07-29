import torch
from tqdm import tqdm 
from .metrics import overlap_f1_score

class QATrainWrapper(torch.nn.Module):

    def __init__(self, model, verbose=True):

        super(QATrainWrapper, self).__init__()
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters())

    def train(self, train_data_iter, epochs, loss, val_data_iter=None):

        for epoch in range(epochs):
            epoch_total_f1 = 0
            epoch_total_loss = 0
            data_iter_tqdm = tqdm(train_data_iter)
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
                data_iter_tqdm.set_description(f"""[{epoch+1}/{epochs} - Train] Loss: {
                    round(epoch_total_loss/(i+1), 3)}, Overlap F1: {
                        round(epoch_total_f1/(i+1), 3)}""")

            if val_data_iter is not None:
                epoch_total_f1 = 0
                epoch_total_loss = 0
                data_iter_tqdm = tqdm(val_data_iter)
                for i, batch in enumerate(data_iter_tqdm):
                    with torch.no_grad():
                        contexts_token_ids, queries_token_ids, answers_tokens_ids = batch
                        answers_probs = self.model([contexts_token_ids, queries_token_ids])
                        answer_preds = answers_probs.argmax(dim=1)
                        loss_step = loss(answers_probs, answers_tokens_ids)
                        epoch_total_loss += loss_step.item()
                        epoch_total_f1 += overlap_f1_score(answer_preds, answers_tokens_ids).mean().item()
                        data_iter_tqdm.set_description(f"""[{epoch+1}/{epochs} - Validation] Loss: {
                            round(epoch_total_loss/(i+1), 3)}, Overlap F1: {
                                round(epoch_total_f1/(i+1), 3)}""")