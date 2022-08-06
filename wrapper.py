import numpy as np
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
        self.verbose = verbose

    def train(self, train_data_iter, epochs, 
              loss=bi_cross_entropy, 
              metrics={"EM": exact_match, "F1": overlap_f1_score},
              val_data_iter=None):

        train_dataset_len = len(train_data_iter.dataset)
        train_batch_size = train_data_iter.batch_size
        train_loader_n_iter = len(train_data_iter)
        train_loss_by_epochs = np.array((epochs * train_dataset_len, 2))
        train_loss_by_epochs[:, 0] = np.repeat(np.arange(epochs), train_dataset_len)
        train_metrics_by_epochs = np.array((epochs * train_dataset_len, len(metrics) + 1))
        train_metrics_by_epochs[:, 0] = np.repeat(np.arange(epochs), train_dataset_len)
        if val_data_iter is not None:
            val_dataset_len = len(val_data_iter.dataset)
            val_batch_size = val_data_iter.batch_size
            val_loader_n_iter = len(val_data_iter)
            val_loss_by_epochs = np.array((epochs * val_dataset_len, 2))
            val_loss_by_epochs[:, 0] = np.repeat(np.arange(epochs), val_dataset_len)
            val_metrics_by_epochs = np.array((epochs * val_dataset_len, len(metrics) + 1))
            val_metrics_by_epochs[:, 0] = np.repeat(np.arange(epochs), val_dataset_len)
            
        for epoch in range(epochs):
            epoch_total_loss = 0
            epoch_total_metrics = {k:0 for k in metrics}
            data_iter_tqdm = tqdm(train_data_iter) if self.verbose else train_data_iter
            for i, batch in enumerate(data_iter_tqdm):
                contexts_tokens, queries_tokens, answers_tokens_range = tuple(batch.values())
                answers_probs = self.model([contexts_tokens, queries_tokens], True)
                answer_preds = answers_probs.argmax(dim=1)
                self.optimizer.zero_grad()
                loss_batch = loss(answers_probs, answers_tokens_range)
                loss_step = loss_batch.mean()
                loss_step.backward()
                self.optimizer.step()
                metrics_batch_dict = {k : metric(answer_preds, answers_tokens_range)
                                      for k,metric in metrics.items()}
                metrics_batch = torch.cat(list(metrics_batch_dict.values()), axis=1)            
                min_slice = (train_loader_n_iter * epoch) + (train_batch_size * i)
                max_slice = ((train_loader_n_iter * epoch) + (train_batch_size * i) +
                             train_batch_size if i+1 != train_loader_n_iter 
                                              else train_dataset_len % train_batch_size)
                batch_slice = slice(min_slice, max_slice)
                train_loss_by_epochs[batch_slice, 1] = loss_batch.to("cpu").numpy()
                train_metrics_by_epochs[batch_slice, 1:] = metrics_batch.to("cpu").numpy()
                epoch_total_loss += loss_step.to("cpu").item()
                epoch_total_metrics = {k : epoch_total_metrics[k] +
                                           metrics_batch_dict[k].mean().to("cpu").item()
                                       for k in metrics}
                if self.verbose:
                    data_iter_tqdm.set_description(f"""Epoch {epoch+1}/{epochs} - Train - Loss: {
                        round(epoch_total_loss/(i+1), 3)}, Metrics: {
                            {id:round(etm/(i+1), 3) for id, etm in epoch_total_metrics.items()}}""")

            if val_data_iter is not None:
                epoch_total_loss = 0
                epoch_total_metrics = {k:0 for k in metrics}
                data_iter_tqdm = tqdm(val_data_iter) if self.verbose else val_data_iter
                for i, batch in enumerate(data_iter_tqdm):
                    with torch.no_grad():
                        contexts_tokens, queries_tokens, answers_tokens_range = tuple(batch.values())
                        answers_probs = self.model([contexts_tokens, queries_tokens])
                        answer_preds = answers_probs.argmax(dim=1)
                        loss_batch = loss(answers_probs, answers_tokens_range)
                        loss_step = loss_batch.mean()
                        metrics_batch_dict = {k : metric(answer_preds, answers_tokens_range)
                                            for k,metric in metrics.items()}
                        metrics_batch = torch.cat(list(metrics_batch_dict.values()), axis=1)            
                        min_slice = (val_loader_n_iter * epoch) + (val_batch_size * i)
                        max_slice = ((val_loader_n_iter * epoch) + (val_batch_size * i) +
                                    val_batch_size if i+1 != val_loader_n_iter 
                                                    else val_dataset_len % val_batch_size)
                        batch_slice = slice(min_slice, max_slice)
                        val_loss_by_epochs[batch_slice, 1] = loss_batch.to("cpu").numpy()
                        val_metrics_by_epochs[batch_slice, 1:] = metrics_batch.to("cpu").numpy()
                        epoch_total_loss += loss_step.to("cpu").item()
                        epoch_total_metrics = {k : epoch_total_metrics[k] +
                                                metrics_batch_dict[k].mean().to("cpu").item()
                                            for k in metrics}
                        if self.verbose:
                            data_iter_tqdm.set_description(f"""Epoch {epoch+1}/{epochs} - Val - Loss: {
                                round(epoch_total_loss/(i+1), 3)}, Metrics: {
                                    {id:round(etm/(i+1), 3) for id, etm in epoch_total_metrics.items()}}""")

        to_return = [train_loss_by_epochs, train_metrics_by_epochs]
        if val_data_iter is not None:
            to_return.extend([val_loss_by_epochs, val_metrics_by_epochs])
        return to_return