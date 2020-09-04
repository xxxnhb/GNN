import torch.nn as nn
from .resnet import resnet12

class LinearClassifier(nn.Module):

    def __init__(self, in_dim, n_classes):
        super().__init__()
        self.linear = nn.Linear(in_dim, n_classes)

    def forward(self, x):
        return self.linear(x)


class Classifier(nn.Module):

    def __init__(self, encoder, encoder_args,
                 classifier, classifier_args):
        super().__init__()
        self.encoder = resnet12(**encoder_args)
        classifier_args['in_dim'] = self.encoder.emb_size
        self.classifier = LinearClassifier(**classifier_args)

    def forward(self, x):
        x = self.encoder(x)
        x = self.classifier(x)
        return x
