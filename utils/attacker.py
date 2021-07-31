import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
from utils.loss import cross_entropy_loss_and_accuracy
import tqdm

class PGDAttacker():
    def __init__(self, num_iter, epsilon, step_size, event_step_size, num_classes=101, targeted=False):
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.step_size = step_size
        self.event_step_size = event_step_size
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

    def attack(self, image_clean, label, model, mode='event'):
        if mode == 'event':
            return self.pgd_attack(image_clean, label, model)
        elif mode == 'event_time':
            return self.pgd_attack2(image_clean, label, model)

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
            pred = model._forward_impl(adv)
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


    def pgd_attack2(self, image_clean, label, model):
        if self.targeted:
            target_label = label
        else:
            target_label = self._create_random_target(label)

        adv = image_clean.clone().detach()
        adv = self.fill_null_event(adv, T=self.epsilon, H=180, W=240)

        # event adv should be 0 or 1
        event_adv = (adv[:, 2] != 0).float()
        event_adv.requires_grad = True

        # if adv==0: 0 < time < 1 random uniform
        time_adv = torch.where(adv[:, 2] == 0, torch.rand_like(adv[:, 2]), adv[:, 2])
        time_adv.requires_grad = True

        for i in range(self.num_iter):
            event_adv.requires_grad = True
            time_adv.requires_grad = True
            adv[:, 2] = event_adv * time_adv
            pred = model._forward_impl(adv)
            losses = F.cross_entropy(pred, target_label)
            event_g = torch.autograd.grad(losses, event_adv,
                                    retain_graph=True, create_graph=False)[0]
            time_g = torch.autograd.grad(losses, time_adv,
                                         retain_graph=False, create_graph=False)[0]
            # if self.projection != "polarity":
            #     g = self.apply_polarity_constraint(g)

            with torch.no_grad():
                # Linf step
                if self.targeted:
                    # event_adv = event_adv + torch.sign(event_g) * self.event_step_size
                    event_adv = event_adv + event_g * self.event_step_size
                    time_adv = time_adv + torch.sign(time_g) * self.step_size
                else:
                    event_adv = event_adv - event_g * self.event_step_size  # targeted
                    time_adv = time_adv - torch.sign(time_g) * self.step_size  # targeted

            # Linf project
            event_adv = torch.where(event_adv < 0.5, torch.zeros_like(event_adv),
                                    torch.ones_like(event_adv)).float().detach()
            time_adv = torch.clamp(time_adv, min=0.0, max=1.0)
            adv[:, 2] = event_adv * time_adv
            event_adv = event_adv.detach()
            time_adv = time_adv.detach()

        return adv, target_label



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
