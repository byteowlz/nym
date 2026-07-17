"""Token-classification head for Gemma3 -- transformers 5.13 ships one for
Gemma 1/2 but not 3. Importing this module registers the class with
AutoModelForTokenClassification, so train_ner.py works unchanged with
--base-model google/gemma-3-270m.

Caveat for the experiment this exists for: Gemma3 is decoder-only, so unlike
the mmBERT/XLM-R encoders every token only sees LEFT context. NER labels often
depend on right context ("Dr. ? Smith works at ?"); that handicap is part of
what the arch comparison measures.
"""
import torch.nn as nn
from transformers import AutoModelForTokenClassification
from transformers.modeling_outputs import TokenClassifierOutput
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.models.gemma3.modeling_gemma3 import Gemma3PreTrainedModel, Gemma3TextModel


class Gemma3ForTokenClassification(Gemma3PreTrainedModel):
    config_class = Gemma3TextConfig

    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.model = Gemma3TextModel(config)  # key prefix matches the CausalLM checkpoint
        self.dropout = nn.Dropout(getattr(config, "classifier_dropout", None) or 0.0)
        self.score = nn.Linear(config.hidden_size, config.num_labels)
        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, labels=None, **kw):
        kw.pop("token_type_ids", None)
        out = self.model(input_ids=input_ids, attention_mask=attention_mask, **kw)
        logits = self.score(self.dropout(out.last_hidden_state))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(
                logits.view(-1, self.num_labels).float(), labels.view(-1), ignore_index=-100)
        return TokenClassifierOutput(loss=loss, logits=logits)


AutoModelForTokenClassification.register(Gemma3TextConfig, Gemma3ForTokenClassification)
