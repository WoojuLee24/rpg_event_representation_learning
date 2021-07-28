import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
from utils.loss import cross_entropy_loss_and_accuracy
import tqdm

class PGDAttacker():
    def __init__(self, num_iter, epsilon, step_size, num_classes=101, targeted=False):
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.step_size = step_size
        self.num_classes = num_classes
        self.targeted = targeted

    def _create_random_target(self, label):
        label_offset = torch.randint_like(label, low=0, high=self.num_classes)
        return (label + label_offset) % self.num_classes

    def set_model(self, model):
        self.model = model

    def apply_polarity_constraint(self, x):
        B, C, H, W = x.size()
        p = x[:, :C // 2, :, :]
        n = x[:, C // 2:, :, :]
        p = torch.where(torch.abs(p) == torch.abs(n), torch.zeros_like(p), p)
        x = torch.cat((p, n), dim=1)
        return x

    def fill_null_event(self, events, T, H, W):
        B = int((1 + events[-1, -1]).item())
        vox = torch.zeros((W, H, T, 2, B))
        null_event = (vox == 0).nonzero().float().cuda()
        null_event[:, 2] = 0
        #         events = torch.cat([events, null_event], dim=0)
        events = torch.cat([null_event, events], dim=0)
        return events

    def pgd_attack(self, image_clean, label, model):
        if self.targeted:
            target_label = label
        else:
            target_label = self._create_random_target(label)

        adv = image_clean.clone().detach()
        adv = self.fill_null_event(adv, T=self.epsilon, H=180, W=240)
        adv.requires_grad = True
        for i in range(self.num_iter):
            adv.requires_grad = True
            pred = model._forward_impl(adv) # memory leak
            # losses, accuracy = cross_entropy_loss_and_accuracy(pred, target_label)
            losses = F.cross_entropy(pred, target_label)
            g = torch.autograd.grad(losses, adv,
                                    retain_graph=False, create_graph=False)[0]

            # if self.projection != "polarity":
            #     g = self.apply_polarity_constraint(g)

            with torch.no_grad():
                # Linf step
                if self.targeted:
                    adv = adv + torch.sign(g) * self.step_size
                else:
                    adv = adv - torch.sign(g) * self.step_size

            # Linf project
            adv = adv.detach()
            target_label = target_label.detach()

        return adv, target_label




# def pgd_attack(self, image_clean, label, num_iter=2, step_size=0.1, epsilon=4, original=False):
#     if original:
#         target_label = label  # untargeted
#     else:
#         target_label = self._create_random_target(label)  # targeted
#
#     adv = image_clean.clone().detach()
#     adv.requires_grad = True
#     for i in range(num_iter):
#
#         pred = self.classifier.forward(adv)
#         losses, accuracy = cross_entropy_loss_and_accuracy(pred, target_label)
#         g = torch.autograd.grad(losses, adv,
#                                 retain_graph=False, create_graph=False)[0]
#
#         g_topk = self.get_topk(g, epsilon)
#         # g_topk = g
#         if self.projection != "polarity":
#             g_topk = self.apply_polarity_constraint(g_topk)
#
#         # Linf step
#         if original:
#             adv = adv + torch.sign(g_topk) * step_size  # untargeted
#         else:
#             adv = adv - torch.sign(g_topk) * step_size  # targeted
#
#         # # Linf step
#         # if original:
#         #     adv = adv + g_topk * step_size  # untargeted
#         # else:
#         #     adv = adv - g_topk * step_size  # targeted
#
#         # Linf project for time??
#
#     return adv.detach(), target_label

# def random_attack(self, image_clean, label, num_iter=2, step_size=0.1, epsilon=4, original=False):
#     if original:
#         target_label = label  # untargeted
#     else:
#         target_label = self._create_random_target(label)  # targeted
#
#     adv = image_clean.clone().detach()
#     adv.requires_grad = True
#     for i in range(num_iter):
#         pred = self.classifier.forward(adv)
#         losses, accuracy = cross_entropy_loss_and_accuracy(pred, target_label)
#         g = torch.autograd.grad(losses, adv,
#                                 retain_graph=False, create_graph=False)[0]
#
#         g_randk = self.get_randk(g, epsilon)
#         g_randk = self.apply_polarity_constraint(g_randk)
#         # Linf step
#         if original:
#             adv = adv + torch.sign(g_randk) * step_size  # untargeted
#         else:
#             adv = adv - torch.sign(g_randk) * step_size  # targeted
#
#         # Linf project for time??
#
#     return adv, target_label
