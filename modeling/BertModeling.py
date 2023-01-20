from __future__ import absolute_import, division, print_function, unicode_literals

from transformers import *

import torch
from torch.nn import BCEWithLogitsLoss
from torch.nn import MSELoss, L1Loss, MultiLabelSoftMarginLoss
from util_loss import ResampleLoss

# Class for MultiLabel Sequence Classification with BERT
# This class is based on:
#  - https://github.com/expertailab/Is-BERT-self-attention-a-feature-selection-method
#  - https://github.com/Roche/BalancedLossNLP

class BertForMultiLabelSequenceClassification(BertPreTrainedModel):
    """BERT model for classification.
    This module is composed of the BERT model with a linear layer on top of
    the pooled output.
    """     
    def __init__(self, config):
        super(BertForMultiLabelSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.bert = BertModel(config)
        self.dropout = torch.nn.Dropout(config.hidden_dropout_prob)
        self.classifier = torch.nn.Linear(config.hidden_size, self.config.num_labels)

        self.init_weights()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                position_ids=None, head_mask=None, inputs_embeds=None, labels=None, pos_weight = None, class_freq = None, train_num = None, loss_func_name = None, device=None):

        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask)

        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            # labels are necessary for calculating loss
            # At prediction labels can be none, but we don';'t need loss there.
            if pos_weight is not None:
                loss_fct = BCEWithLogitsLoss(pos_weight)
            elif loss_func_name == "MultiLabelSoftMarginLoss" :
                loss_fct = MultiLabelSoftMarginLoss()
            elif loss_func_name == "L1Loss" :
                loss_fct = L1Loss()
            elif loss_func_name == "MSELoss" :
                loss_fct = MSELoss()
            elif loss_func_name is not None and train_num is not None and class_freq is not None:
                if loss_func_name == 'BCE':
                    loss_fct = ResampleLoss(reweight_func=None, loss_weight=1.0,
                                             focal=dict(focal=False, alpha=0.5, gamma=2),
                                             logit_reg=dict(),
                                             class_freq=class_freq, train_num=train_num, device=device)

                if loss_func_name == 'FL':
                    loss_fct = ResampleLoss(reweight_func=None, loss_weight=1.0,
                                             focal=dict(focal=True, alpha=0.5, gamma=2),
                                             logit_reg=dict(),
                                             class_freq=class_freq, train_num=train_num, device=device)

                if loss_func_name == 'CBloss':  # CB
                    loss_fct = ResampleLoss(reweight_func='CB', loss_weight=5.0,
                                             focal=dict(focal=True, alpha=0.5, gamma=2),
                                             logit_reg=dict(),
                                             CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                             class_freq=class_freq, train_num=train_num, device=device)

                if loss_func_name == 'R-BCE-Focal':  # R-FL
                    loss_fct = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                             focal=dict(focal=True, alpha=0.5, gamma=2),
                                             logit_reg=dict(),
                                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.05),
                                             class_freq=class_freq, train_num=train_num, device=device)

                if loss_func_name == 'NTR-Focal':  # NTR-FL
                    loss_fct = ResampleLoss(reweight_func=None, loss_weight=0.5,
                                             focal=dict(focal=True, alpha=0.5, gamma=2),
                                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                             class_freq=class_freq, train_num=train_num, device=device)

                if loss_func_name == 'DBloss-noFocal':  # DB-0FL
                    loss_fct = ResampleLoss(reweight_func='rebalance', loss_weight=0.5,
                                             focal=dict(focal=False, alpha=0.5, gamma=2),
                                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.05),
                                             class_freq=class_freq, train_num=train_num, device=device)

                if loss_func_name == 'CBloss-ntr':  # CB-NTR
                    loss_fct = ResampleLoss(reweight_func='CB', loss_weight=10.0,
                                             focal=dict(focal=True, alpha=0.5, gamma=2),
                                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                             CB_loss=dict(CB_beta=0.9, CB_mode='by_class'),
                                             class_freq=class_freq, train_num=train_num, device=device)

                if loss_func_name == 'DBloss':  # DB
                    loss_fct = ResampleLoss(reweight_func='rebalance', loss_weight=1.0,
                                             focal=dict(focal=True, alpha=0.5, gamma=2),
                                             logit_reg=dict(init_bias=0.05, neg_scale=2.0),
                                             map_param=dict(alpha=0.1, beta=10.0, gamma=0.05),
                                             class_freq=class_freq, train_num=train_num, device=device)

            logits = logits.double()
            loss = loss_fct(logits.view(-1, self.num_labels), labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)