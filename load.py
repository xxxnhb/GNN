import torch
from classifier import Classifier


def load_model(model_path, n_cls):
    """load the pretrained model"""
    print('==> loading pretrained model')
    model_args = {'encoder':'resnet12', 'encoder_args': {},
                  'classifier': 'Linear-Classifier', 'classifier_args': {'n_classes': n_cls}}
    model = Classifier(**model_args)
    model.load_state_dict(torch.load(model_path)['model'])
    print('==> done')
    return model