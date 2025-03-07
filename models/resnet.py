import torchvision
import torch.nn as nn

def resnet18(num_classes=100, pretrained=False):
    model = torchvision.models.resnet18(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet34(num_classes=100, pretrained=False):
    model = torchvision.models.resnet34(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet50(num_classes=100, pretrained=False):
    model = torchvision.models.resnet50(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet101(num_classes=100, pretrained=False):
    model = torchvision.models.resnet101(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model

def resnet152(num_classes=100, pretrained=False):
    model = torchvision.models.resnet152(pretrained=pretrained)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model