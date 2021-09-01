import argparse
from os.path import dirname
import torch
import torchvision
import os
import numpy as np
import tqdm

from utils.models import ESTNet, AdvESTNet
from torch.utils.tensorboard import SummaryWriter
from utils.loader import Loader
from utils.loss import cross_entropy_loss_and_accuracy, adv_cross_entropy_loss_and_accuracy
from utils.dataset import NCaltech101, NCARS, NMNIST, DVSGesture
from utils.attacker import PGDAttacker

torch.manual_seed(1)
np.random.seed(1)


def FLAGS():
    parser = argparse.ArgumentParser("""Train classifier using a learnt quantization layer.""")

    # training / validation dataset
    parser.add_argument("--validation_dataset", default="", required=True)
    parser.add_argument("--training_dataset", default="", required=True)

    # logging options
    parser.add_argument("--log_dir", default="", required=True)
    parser.add_argument("--checkpoint", type=str, default=None)


    # loader and device options
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--pin_memory", type=bool, default=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0001)

    parser.add_argument("--num_epochs", type=int, default=30)
    parser.add_argument("--save_every_n_epochs", type=int, default=5)

    # network architecture
    parser.add_argument("--voxel_channel", type=int, default=9)
    parser.add_argument("--value_layer", type=str, default="ValueLayer")
    parser.add_argument("--projection", type=str, default=None)


    # adv attack options
    parser.add_argument("--adv", type=bool, default=False)
    parser.add_argument("--attack_mode", type=str, default='event')
    parser.add_argument("--adv_test", type=bool, default=False)
    parser.add_argument("--targeted", type=bool, default=False)
    parser.add_argument("--epsilon", type=int, default=10)
    parser.add_argument("--num_iter", type=int, default=2)
    parser.add_argument("--step_size", type=float, default=0.5)
    parser.add_argument("--event_step_size", type=float, default=0.5)


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


def train_epoch(model, train_loader, optimizer, lr_scheduler, epoch):

    sum_accuracy = 0
    sum_loss = 0

    model = model.train()
    print(f"Training step [{epoch:3d}/{flags.num_epochs:3d}]")
    for events, labels in tqdm.tqdm(train_loader):
        optimizer.zero_grad()
        pred_labels, labels = model(events, labels)
        loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)
        loss.backward()
        optimizer.step()
        # print(f"Training epoch {epoch:5d} Accuracy {accuracy.item():.4f}")
        sum_accuracy += accuracy
        sum_loss += loss

    if epoch % 10 == 9:
        lr_scheduler.step()

    training_loss = sum_loss.item() / len(training_loader)
    training_accuracy = sum_accuracy.item() / len(training_loader)
    print(f"Training epoch {epoch:5d}  Loss {training_loss:.4f}  Accuracy {training_accuracy:.4f}")

    return training_loss, training_accuracy


def test_epoch(model, val_loader, epoch):
    sum_accuracy = 0
    sum_loss = 0
    model = model.eval()

    print(f"Validation step [{epoch:3d}/{flags.num_epochs:3d}]")
    for events, labels in tqdm.tqdm(val_loader):
        with torch.no_grad():
            pred_labels, labels = model(events, labels)
            loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)

        sum_accuracy += accuracy
        sum_loss += loss
    validation_loss = sum_loss.item() / len(validation_loader)
    validation_accuracy = sum_accuracy.item() / len(validation_loader)

    print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f}")

    return validation_loss, validation_accuracy


def test_epoch_adv(model, val_loader):

    sum_accuracy = 0
    sum_adv_accuracy = 0
    sum_adv_target_accuracy = 0
    sum_correct_num = 0
    sum_attack_num = 0
    sum_loss = 0
    model = model.eval()

    for events, labels in tqdm.tqdm(val_loader):
        pred_labels, labels = model(events, labels)
        (pred, adv_pred), (labels, target_label) = pred_labels, labels
        loss, accuracy, adv_accuracy, adv_target_accuracy, correct_num, attack_num = \
            adv_cross_entropy_loss_and_accuracy(pred, adv_pred, labels, target_label)

        sum_accuracy += accuracy
        sum_adv_accuracy += adv_accuracy
        sum_adv_target_accuracy += adv_target_accuracy
        sum_correct_num += correct_num
        sum_attack_num += attack_num
        sum_loss += loss

    validation_loss = sum_loss.item() / len(validation_loader)
    validation_accuracy = sum_accuracy.item() / len(validation_loader)
    adv_validation_accuracy = sum_adv_accuracy.item() / len(validation_loader)
    adv_target_validation_accuracy = sum_adv_target_accuracy.item() / len(validation_loader)
    attack_validation_accuracy = sum_attack_num.item() / sum_correct_num.item()

    print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f} ")
    print(f"Adv Accuracy {adv_validation_accuracy:.4f} "
          f"Adv Target Accuracy {adv_target_validation_accuracy:.4f} "
          f"Attack Accuracy {attack_validation_accuracy:.4f} ")

    return validation_loss, validation_accuracy, adv_validation_accuracy, adv_target_validation_accuracy, attack_validation_accuracy


if __name__ == '__main__':
    flags = FLAGS()
    torch.cuda.empty_cache()
    dataset = (flags.training_dataset).split("/")[3]

    if dataset == "N-Caltech101":
        # datasets, add augmentation to training set
        training_dataset = NCaltech101(flags.training_dataset, augmentation=True)
        validation_dataset = NCaltech101(flags.validation_dataset)
        voxel_dimension = (flags.voxel_channel, 180, 240)
        crop_dimension = (224, 224)
    elif dataset == "NCARS":
        # datasets, add augmentation to training set
        training_dataset = NCARS(flags.training_dataset, augmentation=True)
        validation_dataset = NCARS(flags.validation_dataset)
        voxel_dimension = (flags.voxel_channel, 240, 304)
        crop_dimension = (224, 224)
    elif dataset == "N-MNIST":
        training_dataset = NMNIST(flags.training_dataset, augmentation=True)
        validation_dataset = NMNIST(flags.validation_dataset)
        voxel_dimension = (flags.voxel_channel, 34, 34)
        crop_dimension = (28, 28)
    elif dataset == "DVSGesture":
        training_dataset = DVSGesture(flags.training_dataset, augmentation=True)
        validation_dataset = DVSGesture(flags.validation_dataset)
        voxel_dimension = (flags.voxel_channel, 180, 240)
        crop_dimension = (224, 224)

    # construct loader, handles data streaming to gpu
    training_loader = Loader(training_dataset, flags, device=flags.device)
    validation_loader = Loader(validation_dataset, flags, device=flags.device)

    # model, and put to device
    # model = Classifier(voxel_dimension=(flags.voxel_channel, 180, 240), value_layer=flags.value_layer, projection=flags.projection,
    #                    adv=flags.adv, epsilon=flags.epsilon, num_iter=flags.num_iter, step_size=flags.step_size)

    model = AdvESTNet(voxel_dimension=voxel_dimension, crop_dimension=crop_dimension,
                      num_classes=training_dataset.classes, value_layer=flags.value_layer, projection=flags.projection, pretrained=True,
                      adv=flags.adv, adv_test=flags.adv_test, attack_mode=flags.attack_mode)
    if flags.adv == True:
        attacker = PGDAttacker(num_iter=flags.num_iter, epsilon=flags.epsilon,
                               step_size=flags.step_size, event_step_size=flags.event_step_size,
                               num_classes=training_dataset.classes, voxel_dimension=voxel_dimension,
                               targeted=flags.targeted)
        model.set_attacker(attacker)
    model = model.to(flags.device)

    # optimizer and lr scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=flags.lr)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, 0.5)

    writer = SummaryWriter(flags.log_dir)

    start_epoch = 0
    iteration = 0
    min_validation_loss = 1000

    if flags.checkpoint is not None:
        checkpoint = torch.load(flags.checkpoint, map_location=flags.device)
        model.load_state_dict(checkpoint["state_dict"])
        start_epoch = checkpoint["epoch"]
        # optimizer = optimizer.load_state_dict(checkpoint["optimizer"])

    if flags.adv_test:
        validation_loss, validation_accuracy, adv_validation_accuracy, adv_target_validation_accuracy, attack_validation_accuracy = \
            test_epoch_adv(model, validation_loader)

        writer.add_scalar("validation/accuracy", validation_accuracy, iteration)
        writer.add_scalar("validation/adv_accuracy", adv_validation_accuracy, iteration)
        writer.add_scalar("validation/adv_accuracy", adv_validation_accuracy, iteration)
        writer.add_scalar("validation/attack_accuracy", attack_validation_accuracy, iteration)
        writer.add_scalar("validation/loss", validation_loss, iteration)
        exit()


    for epoch in range(start_epoch, flags.num_epochs):
        training_loss, training_accuracy = train_epoch(model, training_loader, optimizer, lr_scheduler, epoch)
        writer.add_scalar("training/accuracy", training_accuracy, epoch)
        writer.add_scalar("training/loss", training_loss, epoch)

        validation_loss, validation_accuracy = test_epoch(model, validation_loader, epoch)
        writer.add_scalar("validation/accuracy", validation_accuracy, epoch)
        writer.add_scalar("validation/loss", validation_loss, epoch)

        if validation_loss < min_validation_loss:
            min_validation_loss = validation_loss
            state_dict = model.state_dict()

            torch.save({
                "state_dict": state_dict,
                "min_val_loss": min_validation_loss,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
            }, flags.log_dir + "/model_best.pth")
            print("New best at ", validation_loss)

        if epoch % flags.save_every_n_epochs == 0:
            state_dict = model.state_dict()
            torch.save({
                "state_dict": state_dict,
                "min_val_loss": min_validation_loss,
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
            }, flags.log_dir + "/checkpoint_%05d_%.4f.pth" % (epoch, min_validation_loss))



    # for i in range(start_epoch, flags.num_epochs):
    #
    #     sum_accuracy = 0
    #     sum_loss = 0
    #     if flags.adv_test == False:
    #         model = model.train()
    #         print(f"Training step [{i:3d}/{flags.num_epochs:3d}]")
    #         for events, labels in tqdm.tqdm(training_loader):
    #             optimizer.zero_grad()
    #
    #             pred_labels, labels = model(events, labels)
    #             loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)
    #
    #             loss.backward()
    #
    #             optimizer.step()
    #
    #             sum_accuracy += accuracy
    #             sum_loss += loss
    #
    #             iteration += 1
    #
    #         if i % 10 == 9:
    #             lr_scheduler.step()
    #
    #         training_loss = sum_loss.item() / len(training_loader)
    #         training_accuracy = sum_accuracy.item() / len(training_loader)
    #         print(f"Training Iteration {iteration:5d}  Loss {training_loss:.4f}  Accuracy {training_accuracy:.4f}")
    #
    #         writer.add_scalar("training/accuracy", training_accuracy, iteration)
    #         writer.add_scalar("training/loss", training_loss, iteration)
    #
    #         # representation_vizualization = create_image(representation)
    #         # writer.add_image("training/representation", representation_vizualization, iteration)
    #
    #     sum_accuracy = 0
    #     sum_loss = 0
    #     model = model.eval()
    #
    #     print(f"Validation step [{i:3d}/{flags.num_epochs:3d}]")
    #     for events, labels in tqdm.tqdm(validation_loader):
    #         if flags.adv_test:
    #             pred_labels, labels = model(events, labels)
    #             (pred, adv_pred), (labels, target_label) = pred_labels, labels
    #             loss, accuracy, adv_accuracy, correct_num, attack_num = \
    #                 adv_cross_entropy_loss_and_accuracy(pred, adv_pred, labels, target_label)
    #
    #         with torch.no_grad():
    #             pred_labels, labels = model(events, labels)
    #             loss, accuracy = cross_entropy_loss_and_accuracy(pred_labels, labels)
    #
    #         sum_accuracy += accuracy
    #         sum_loss += loss
    #
    #     validation_loss = sum_loss.item() / len(validation_loader)
    #     validation_accuracy = sum_accuracy.item() / len(validation_loader)
    #
    #     writer.add_scalar("validation/accuracy", validation_accuracy, iteration)
    #     writer.add_scalar("validation/loss", validation_loss, iteration)
    #
    #     # # visualize representation
    #     # representation_vizualization = create_image(representation)
    #     # writer.add_image("validation/representation", representation_vizualization, iteration)
    #
    #     print(f"Validation Loss {validation_loss:.4f}  Accuracy {validation_accuracy:.4f}")
    #
    #     if validation_loss < min_validation_loss:
    #         min_validation_loss = validation_loss
    #         state_dict = model.state_dict()
    #
    #         torch.save({
    #             "state_dict": state_dict,
    #             "min_val_loss": min_validation_loss,
    #             "iteration": iteration,
    #             "optimizer": optimizer.state_dict(),
    #         }, flags.log_dir + "/model_best.pth")
    #         print("New best at ", validation_loss)
    #
    #     if i % flags.save_every_n_epochs == 0:
    #         state_dict = model.state_dict()
    #         torch.save({
    #             "state_dict": state_dict,
    #             "min_val_loss": min_validation_loss,
    #             "iteration": iteration,
    #             "optimizer": optimizer.state_dict(),
    #         }, flags.log_dir + "/checkpoint_%05d_%.4f.pth" % (iteration, min_validation_loss))





