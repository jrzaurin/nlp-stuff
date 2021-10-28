from torch import nn
from transformers import BertModel


from typing import List
from torch import Tensor


class BertClassifier(nn.Module):
    def __init__(
        self,
        head_hidden_dim: List[int],
        model_name: str = "bert-base-uncased",
        freeze_bert: bool = False,
        head_dropout: float = 0.0,
        num_class: int = 4,
    ):
        super(BertClassifier, self).__init__()

        self.bert = BertModel.from_pretrained(model_name)

        classifier_dims = [768] + head_hidden_dim + [num_class]
        self.classifier = MLP(classifier_dims, head_dropout)

        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]

        return self.classifier(last_hidden_state_cls)


class MLP(nn.Module):
    def __init__(
        self,
        d_hidden: List[int],
        dropout: float = 0.0,
    ):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential()
        for i in range(1, len(d_hidden)):
            self.mlp.add_module(
                "dense_layer_{}".format(i - 1),
                self._dense_layer(
                    d_hidden[i - 1],
                    d_hidden[i],
                    dropout,
                ),
            )

    def forward(self, X: Tensor) -> Tensor:
        return self.mlp(X)

    @staticmethod
    def _dense_layer(inp: int, out: int, p: float):
        layers: List = [nn.Dropout(p)] if p > 0 else []
        layers += [nn.Linear(inp, out), nn.ReLU(inplace=True)]
        return nn.Sequential(*layers)
