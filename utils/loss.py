import torch


def cross_entropy_loss_and_accuracy(prediction, target):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(prediction, target)
    k = prediction.argmax(1)
    accuracy = (prediction.argmax(1) == target).float().mean()
    return loss, accuracy

def adv_cross_entropy_loss_and_accuracy(prediction, adv_prediction, target, adv_target):
    cross_entropy_loss = torch.nn.CrossEntropyLoss()
    loss = cross_entropy_loss(prediction, target)
    accuracy = (prediction.argmax(1) == target).float().mean()
    adv_accuracy = (adv_prediction.argmax(1) == target).float().mean()
    adv_target_accuracy = (adv_prediction.argmax(1) == adv_target).float().mean()
    correct_num = (prediction.argmax(1) == target).float().sum()
    attack_num = (((prediction.argmax(1) == target).float() + (adv_prediction.argmax(1) != target).float()) > 1).float().sum()

    return loss, accuracy, adv_accuracy, adv_target_accuracy, correct_num, attack_num
