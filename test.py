import argparse
from os.path import dirname
import torch
import torchvision
import os
import numpy as np
import tqdm

from utils.models import Classifier
from torch.utils.tensorboard import SummaryWriter
from utils.loader import Loader
from utils.loss import cross_entropy_loss_and_accuracy, adv_cross_entropy_loss_and_accuracy
from utils.dataset import NCaltech101


torch.manual_seed(1)
np.random.seed(1)


def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--validation_dataset", default="", required=True)
    parser.add_argument("--training_dataset", default="", required=True)

    # logging options
    parser.add_argument("--log_dir", default="", required=True)

    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--checkpoint", type=str, default="")

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)

    # network architecture
    parser.add_argument("--value_layer", type=str, default="ValueLayer")
    parser.add_argument("--projection", type=str, default="")


    # adv attack options
    parser.add_argument("--adv", type=bool, default=False)
    parser.add_argument("--adv_test", type=bool, default=False)
    parser.add_argument("--epsilon", type=int, default=10)
    parser.add_argument("--num_iter", type=int, default=2)
    parser.add_argument("--step_size", type=float, default=0.5)

    flags = parser.parse_args()

    if not os.path.exists(flags.log_dir):
        os.mkdir(flags.log_dir)
    assert os.path.isdir(dirname(flags.log_dir)), f"Log directory root {dirname(flags.log_dir)} not found."
    assert os.path.isdir(flags.validation_dataset), f"Validation dataset directory {flags.validation_dataset} not found."
    assert os.path.isdir(flags.training_dataset), f"Training dataset directory {flags.training_dataset} not found."

    print(f"----------------------------\n"
          f"Starting training with \n"
          f"num_epochs: {flags.num_epochs}\n"
          f"batch_size: {flags.batch_size}\n"
          f"device: {flags.device}\n"
          f"log_dir: {flags.log_dir}\n"
          f"training_dataset: {flags.training_dataset}\n"
          f"validation_dataset: {flags.validation_dataset}\n"
          f"----------------------------")

    return flags

def percentile(t, q):
    B, C, H, W = t.shape
    k = 1 + round(.01 * float(q) * (C * H * W - 1))
    result = t.view(B, -1).kthvalue(k).values
    return result[:,None,None,None]

def create_image(representation):
    B, C, H, W = representation.shape
    representation = representation.view(B, 3, C // 3, H, W).sum(2)

    # do robust min max norm
    representation = representation.detach().cpu()
    robust_max_vals = percentile(representation, 99)
    robust_min_vals = percentile(representation, 1)

    representation = (representation - robust_min_vals)/(robust_max_vals - robust_min_vals)
    representation = torch.clamp(255*representation, 0, 255).byte()

    representation = torchvision.utils.make_grid(representation)

    return representation


if __name__ == '__main__':
    flags = FLAGS()

    # datasets, add augmentation to training set
    training_dataset = NCaltech101(flags.training_dataset, augmentation=True)
    validation_dataset = NCaltech101(flags.validation_dataset)

    # construct loader, handles data streaming to gpu
    training_loader = Loader(training_dataset, flags, device=flags.device)
    validation_loader = Loader(validation_dataset, flags, device=flags.device)

    # model, and put to device
    model = Classifier(value_layer=flags.value_layer, projection=flags.projection,
                       adv=flags.adv, adv_test=flags.adv_test,
                       epsilon=flags.epsilon, num_iter=flags.num_iter, step_size=flags.step_size)
    model = model.to(flags.device)
    ckpt = torch.load(flags.checkpoint)
    model.load_state_dict(ckpt["state_dict"])

    # optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    writer = SummaryWriter(flags.log_dir)

    iteration = 0

    sum_accuracy = 0
    sum_adv_accuracy = 0
    sum_correct_num = 0
    sum_attack_num = 0
    sum_loss = 0
    model = model.eval()

    for events, labels in tqdm.tqdm(validation_loader):
        pred_labels, labels = model(events, labels)
        if flags.adv_test == True:
            (pred, adv_pred), (labels, target_label) = pred_labels, labels
            loss, accuracy, adv_accuracy, correct_num, attack_num = \
                adv_cross_entropy_loss_and_accuracy(pred, adv_pred, labels, target_label)
        else:
            loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)

        sum_accuracy += accuracy
        sum_adv_accuracy += adv_accuracy
        sum_correct_num += correct_num
        sum_attack_num += attack_num
        sum_loss += loss

    validation_loss = sum_loss.item() / len(validation_loader)
    validation_accuracy = sum_accuracy.item() / len(validation_loader)
    adv_validation_accuracy = sum_adv_accuracy.item() / len(validation_loader)
    attack_validation_accuracy = sum_attack_num.item() / sum_correct_num.item()

    writer.add_scalar("validation/accuracy", validation_accuracy, iteration)
    writer.add_scalar("validation/adv_accuracy", adv_validation_accuracy, iteration)
    writer.add_scalar("validation/attack_accuracy", attack_validation_accuracy, iteration)
    writer.add_scalar("validation/loss", validation_loss, iteration)

    # # visualize representation
    # representation_vizualization = create_image(representation)
    # writer.add_image("validation/representation", representation_vizualization, iteration)

    print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f} ")
    print(f"Adv Accuracy {adv_validation_accuracy:.4f}  Attack Accuracy {attack_validation_accuracy:.4f} ")




