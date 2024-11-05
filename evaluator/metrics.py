import torch
import torch.nn.functional as F

from torchmetrics import Metric
from sklearn.metrics import f1_score


class ICLMetric(Metric):
    # update method does not require access to global metric state
    full_state_update: bool = False

    def __init__(self, metric_type='acc') -> None:
        """metric_type: f1, acc, len_norm, pmi_dc
        """
        super().__init__(sync_on_compute=True)

        self.metric_type = metric_type

        self.add_state('loglikelihoods', default=[], dist_reduce_fx=None)
        self.add_state('labels', default=[], dist_reduce_fx=None)

    def update(self, batch, lm_logits, dc_lm_logits=None):
        lm_logits = F.log_softmax(lm_logits, dim=-1)

        if self.metric_type == 'pmi_dc':
            assert dc_lm_logits is not None, "PMI_DC acc type selected but no domain conditional logits provided"

        for idx, (doc_id, cont_id) in enumerate(zip(batch['doc_id'], batch['cont_id'])):
            # [cont_len]: continuation is padded for batching
            cont_tokens = batch['continuation'][idx][:batch['cont_len'][idx]]

            # get logits from LM for the continuation: [cont_len, vocab]
            # batch['input_ids'][idx] -> ctx + cont + padding
            # -1 in both indices: lm_logits will be left shited 1 pos as 0th pos in input generates next token in the 0th pos of lm_logits
            lm_cont_logits = lm_logits[idx][batch['ctx_len'][idx] - 1:batch['ctx_len'][idx] + batch['cont_len'][idx] - 1]

            if self.metric_type == 'pmi_dc':
                # get domain conditional continuation logits: [cont_len, vocab]
                dc_lm_cont_logits = dc_lm_logits[idx][batch['dc_len'][idx] - 1:batch['dc_len'][idx] + batch['cont_len'][idx] - 1]

                # gather log-probs at continuation token indices but divide by domain conditional prob
                log_likelihood = torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum() / torch.gather(dc_lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
            elif self.metric_type == 'acc' or self.metric_type == 'f1':
                # gather log-probs at continuation token indices
                log_likelihood = torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum()
            elif self.metric_type == 'len_norm':
                log_likelihood = torch.gather(lm_cont_logits, 1, cont_tokens.unsqueeze(-1)).sum() / batch['cont_str_len'][idx]

            # because metric states cannot be dict/list of tuples, store this tuple as tensor: (doc_id, cont_id, metric_state)
            self.loglikelihoods.append(torch.Tensor((doc_id, cont_id, log_likelihood)).to(batch['continuation'][idx].device))
            self.labels.append(torch.LongTensor((doc_id, cont_id, batch['label_id'][idx])).to(batch['label_id'][idx].device))

    def compute(self):
        # states should have been synced from all accelerators at this point
        # account for duplicates here because of DistributedSampler compensating for drop_last=False
        loglikelihood_dict = {}
        label_dict = {}

        # collect labels
        for doc_id, cont_id, label_id in self.labels:
            if doc_id.item() not in label_dict:
                label_dict[doc_id.item()] = label_id.item()

        # collect loglikelihoods
        for doc_id, cont_id, loglikelihood in self.loglikelihoods:
            if int(doc_id.item()) not in loglikelihood_dict:
                loglikelihood_dict[int(doc_id.item())] = {}

            if int(cont_id.item()) not in loglikelihood_dict[int(doc_id.item())]:
                loglikelihood_dict[int(doc_id.item())][int(cont_id.item())] = loglikelihood

        # compute acc
        correct = []
        if self.metric_type == 'f1':
            preds = []
            labels = []

        for doc_id in loglikelihood_dict:
            # each doc_id might have a different number of continuation
            num_continuations = len(loglikelihood_dict[doc_id].keys())
            loglikelihoods = torch.tensor([-float('inf')] * num_continuations)

            for cont_id in loglikelihood_dict[doc_id]:
                loglikelihoods[cont_id] = loglikelihood_dict[doc_id][cont_id]

            correct.append(1.0 if torch.argmax(loglikelihoods).item() == label_dict[doc_id] else 0.0)

            if self.metric_type == 'f1':
                preds.append(torch.argmax(loglikelihoods).item())
                labels.append(label_dict[doc_id])

        if self.metric_type == 'f1':
            # for NLI tasks, continuations are yes, no, neither, so idx=0 assigned to pos label
            return f1_score(labels, preds, pos_label=0)

        return sum(correct) / len(correct)
