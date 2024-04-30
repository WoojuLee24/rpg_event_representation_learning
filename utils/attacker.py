import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
from utils.loss import cross_entropy_loss_and_accuracy
import tqdm
import pdb

z_table = {0: 100, 0.1: 3.08, 0.5: 2.57, 1: 2.33, 2: 2.05, 3: 1.88, 4: 1.75,5: 1.65, 10: 1.28, 15: 1.03, 20: 0.84, 25: 0.67}


class PGDAttacker():
    def __init__(self, num_iter, epsilon, step_size, topp, null, num_classes=101, voxel_dimension=(9, 180, 240), targeted=False):
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.step_size = step_size
        self.topp = topp
        self.null = null
        self.num_classes = num_classes
        self.voxel_dimension = voxel_dimension
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
        events = torch.cat([null_event, events], dim=0)
        return events

    def make_null_event(self, events, T, voxel_dimension):
        B = int((1 + events[-1, -1]).item())
        C, H, W = voxel_dimension
        vox = torch.zeros((W, H, T, 2, B))
        null_event = (vox == 0).nonzero().float().cuda()
        null_event[:, 2] = 0
        null_event[:, 3] = (null_event[:, 3] - 0.5) * 2
        _, idx = torch.sort(null_event[:, 4])
        null_event = null_event[idx]

        return null_event

    def make_null_event2(self, events, T, voxel_dimension):
        # null event
        B = int((1 + events[-1, -1]).item())
        C, H, W = voxel_dimension
        vox = torch.zeros((W, H, T, 2, B))
        null_event = (vox == 0).nonzero().float().cuda()
        null_event[:, 2] = 0
        null_event[:, 3] = (null_event[:, 3] - 0.5) * 2
        _, idx = torch.sort(null_event[:, 4])
        null_event = null_event[idx]
        # event_set
        event_list = events.clone().detach()
        event_list[:, 2] = 0
        event_set = torch.unique(event_list, dim=0)
        # difference
        combined = torch.cat((event_set, null_event))
        uniques, counts = combined.unique(return_counts=True, dim=0)
        difference = uniques[counts == 1]
        # intersection = uniques[counts > 1]
        return difference

    def get_top_percentile(self, null_g, batch_size):
        i = 0
        boolean_g = 0
        threshold = z_table[self.topp]
        for g in torch.split(null_g, int(null_g.shape[0] / batch_size)):
            if i != 0:
                boolean_g = torch.cat([boolean_g, (g - torch.mean(g)) / torch.std(g) > threshold])
            else:
                boolean_g = (g - torch.mean(g)) / torch.std(g) > threshold
            i += 1

        return boolean_g

    def attack(self, image_clean, label, model, mode='event'):
        if mode == 'shifting':
            return self.pgd_attack(image_clean, label, model)
        elif mode == 'shifting_generating':
            return self.pgd_attack2(image_clean, label, model)
        elif mode == 'shifting_generating2':
            return self.pgd_attack3(image_clean, label, model)

    # def pgd_attack(self, image_clean, label, model):
    #     if self.targeted:
    #         target_label = self._create_random_target(label)
    #     else:
    #         target_label = label
    #
    #     step_size = self.step_size / model.voxel_dimension[0]
    #     epsilon = self.epsilon / model.voxel_dimension[0]
    #
    #     lower_bound = torch.clamp(image_clean[:, 2] - epsilon, min=0., max=1.)
    #     upper_bound = torch.clamp(image_clean[:, 2] + epsilon, min=0., max=1.)
    #
    #     adv = image_clean.clone().detach()
    #     adv.requires_grad = True
    #     for i in range(self.num_iter):
    #         adv.requires_grad = True
    #         pred = model._forward_impl(adv)
    #         losses = F.cross_entropy(pred, target_label)
    #         g = torch.autograd.grad(losses, adv,
    #                                 retain_graph=False, create_graph=False)[0]
    #
    #         with torch.no_grad():
    #             # Linf step
    #             if self.targeted:
    #                 adv = adv - torch.sign(g) * step_size
    #             else:
    #                 adv = adv + torch.sign(g) * step_size
    #
    #         # Linf project
    #         adv[:, 2] = torch.where(adv[:, 2] > lower_bound, adv[:, 2], lower_bound).detach()
    #         adv[:, 2] = torch.where(adv[:, 2] < upper_bound, adv[:, 2], upper_bound).detach()
    #         adv = adv.detach()
    #         target_label = target_label.detach()
    #
    #     return adv, target_label

    def pgd_attack(self, image_clean, label, model):
        if self.targeted:
            target_label = self._create_random_target(label)
        else:
            target_label = label

        # attack setting
        step_size = self.step_size / model.voxel_dimension[0]
        epsilon = self.epsilon / model.voxel_dimension[0]
        lower_bound = torch.clamp(image_clean[:, 2] - epsilon, min=0., max=1.)
        upper_bound = torch.clamp(image_clean[:, 2] + epsilon, min=0., max=1.)

        # shifting only time
        adv = image_clean.clone().detach()
        time_adv = adv[:, 2]
        time_adv.requires_grad = True

        for i in range(self.num_iter):
            time_adv.requires_grad = True
            adv[:, 2] = time_adv
            pred = model._forward_impl(adv)
            losses = F.cross_entropy(pred, target_label)
            g = torch.autograd.grad(losses, adv,
                                    retain_graph=False, create_graph=False)[0]
            # Linf step
            with torch.no_grad():
                if self.targeted:
                    time_adv = time_adv - torch.sign(g[:, 2]) * step_size
                else:
                    time_adv = time_adv + torch.sign(g[:, 2]) * step_size

            # Linf project
            time_adv = torch.where(time_adv > lower_bound, time_adv, lower_bound).detach()
            time_adv = torch.where(time_adv < upper_bound, time_adv, upper_bound).detach()
            time_adv = torch.clamp(time_adv, min=0.0, max=1.0)

        adv[:, 2] = time_adv
        adv = adv.detach()
        target_label = target_label.detach()

        return adv, target_label

    # def pgd_attack2(self, image_clean, label, model):
    #     if self.targeted:
    #         target_label = self._create_random_target(label)
    #     else:
    #         target_label = label
    #
    #     step_size = self.step_size / model.voxel_dimension[0]
    #     epsilon = self.epsilon / model.voxel_dimension[0]
    #
    #     # Typical pgd attack for existing event
    #     real_adv = image_clean.clone().detach()
    #     real_time_adv = real_adv[:, 2]
    #     lower_bound = torch.clamp(real_time_adv - epsilon, min=0., max=1.)
    #     upper_bound = torch.clamp(real_time_adv + epsilon, min=1., max=1.)
    #     real_time_adv.requires_grad = True
    #
    #     for i in range(self.num_iter):
    #         real_adv[:, 2] = real_time_adv
    #         pred = model._forward_impl(real_adv)
    #         losses = F.cross_entropy(pred, target_label)
    #         real_g = torch.autograd.grad(losses, real_adv,
    #                                      retain_graph=False, create_graph=False)[0]
    #         # Linf step
    #         with torch.no_grad():
    #             if self.targeted:
    #                 real_time_adv = real_time_adv - torch.sign(real_g[:, 2]) * step_size
    #             else:
    #                 real_time_adv = real_time_adv + torch.sign(real_g[:, 2]) * step_size
    #
    #         # Linf project
    #         real_time_adv = torch.where(real_time_adv > lower_bound, real_time_adv, lower_bound).detach()
    #         real_time_adv = torch.where(real_time_adv < upper_bound, real_time_adv, upper_bound).detach()
    #         real_time_adv = torch.clamp(real_time_adv, min=0.0, max=1.0)
    #
    #     real_adv[:, 2] = real_time_adv
    #     real_adv = real_adv.detach()
    #
    #     # Generating additional adversarial events
    #     # event = image_clean.clone().detach()
    #     event = real_adv.clone().detach()
    #     null_event = self.make_null_event(event, 1, voxel_dimension=self.voxel_dimension)   # zero value?
    #     adv = torch.cat([event, null_event], dim=0)
    #
    #     null_adv = null_event[:, 2]
    #
    #     # get null_g
    #     null_adv.requires_grad = True
    #     adv[:, 2] = torch.cat([event[:, 2], null_adv], dim=0)
    #     pred = model._forward_impl(adv)
    #     losses = F.cross_entropy(pred, target_label)
    #     null_g = torch.autograd.grad(losses, null_adv, retain_graph=False, create_graph=False)[0]
    #
    #     # top p% null events & random initialization
    #     adam_adv = null_event[self.get_top_percentile(null_g, batch_size=int((1+torch.max(image_clean[:, -1])).item()))]
    #     adam_adv = adam_adv.repeat_interleave(self.null, dim=0)     # max null events per pixel
    #     adam_adv = adam_adv.detach()
    #     time_adv = torch.rand_like(adam_adv[:, 2])  # randomly initialized null events
    #     # time_adv = 0.5 + 0.01 * torch.rand_like(adam_adv[:, 2])
    #     time_adv = time_adv.detach()
    #     time_adv.requires_grad = True
    #
    #     adam_adv[:, 2] = time_adv
    #     # adv = torch.cat([event, adam_adv], dim=0)
    #
    #     optimizer = torch.optim.Adam([time_adv], lr=0.01)
    #
    #     for i in range(20):
    #         #             time_adv.requires_grad = True
    #         adam_adv[:, 2] = time_adv
    #         adv = torch.cat([event, adam_adv], dim=0)
    #         optimizer.zero_grad()
    #         pred = model._forward_impl(adv)
    #
    #         if self.targeted:
    #             losses = F.cross_entropy(pred, target_label)
    #             losses.backward(retain_graph=False)
    #             optimizer.step()
    #         else:
    #             losses = -F.cross_entropy(pred, target_label)
    #             losses.backward(retain_graph=False)
    #             optimizer.step()
    #
    #         time_adv = torch.clamp(time_adv, min=0.0, max=1.0)
    #         adam_adv[:, 2] = time_adv
    #
    #     real_adv = real_adv.detach()
    #     adam_adv = adam_adv.detach()
    #     adv = torch.cat([real_adv, adam_adv], dim=0)
    #     adv = adv.detach()
    #
    #     return adv, target_label

    def pgd_attack2(self, image_clean, label, model):
        real_adv, target_label = self.pgd_attack(image_clean, label, model)

        # Generating additional adversarial events
        event = real_adv.clone().detach()
        null_event = self.make_null_event(event, 1, voxel_dimension=self.voxel_dimension)   # zero value?
        adv = torch.cat([event, null_event], dim=0)

        # get null_g
        null_adv = null_event[:, 2]
        null_adv.requires_grad = True
        adv[:, 2] = torch.cat([event[:, 2], null_adv], dim=0)
        pred = model._forward_impl(adv)
        losses = F.cross_entropy(pred, target_label)
        null_g = torch.autograd.grad(losses, null_adv, retain_graph=False, create_graph=False)[0]

        # get top p% vulnerable position of null events
        adam_adv = null_event[self.get_top_percentile(null_g, batch_size=int((1+torch.max(image_clean[:, -1])).item()))]
        adam_adv = adam_adv.repeat_interleave(self.null, dim=0)     # max null events per pixel
        adam_adv = adam_adv.detach()

        # random initialize times of null events
        time_adv = torch.rand_like(adam_adv[:, 2])  # randomly initialized null events
        time_adv.requires_grad = True
        adam_adv[:, 2] = time_adv
        # adv = torch.cat([event, adam_adv], dim=0)

        # optimizer = torch.optim.Adam([time_adv], lr=0.01)
        event_size = event.size(0)
        for i in range(20):
            time_adv.requires_grad = True
            adam_adv[:, 2] = time_adv
            adv = torch.cat([event, adam_adv], dim=0)

            optimizer = torch.optim.Adam([time_adv], lr=0.01)
            optimizer.zero_grad()
            pred = model._forward_impl(adv)

            if self.targeted:
                losses = F.cross_entropy(pred, target_label)
                losses.backward(retain_graph=True)
                optimizer.step()
            else:
                losses = -F.cross_entropy(pred, target_label)
                losses.backward(retain_graph=True)
                optimizer.step()

            time_adv = torch.clamp(time_adv, min=0.0, max=1.0).detach()

        adam_adv[:, 2] = time_adv
        real_adv = real_adv.detach()
        adam_adv = adam_adv.detach()
        adv = torch.cat([real_adv, adam_adv], dim=0)
        adv = adv.detach()

        return adv, target_label

    def pgd_attack3(self, image_clean, label, model):
        if self.targeted:
            target_label = self._create_random_target(label)
        else:
            target_label = label

        # Typical pgd attack for existing event
        real_adv = image_clean.clone().detach()
        real_time_adv = real_adv[:, 2]
        lower_bound = torch.clamp(real_time_adv - self.epsilon, min=0., max=1.)
        upper_bound = torch.clamp(real_time_adv + self.epsilon, min=1., max=1.)
        real_time_adv.requires_grad = True
        for i in range(self.num_iter):
            #             real_time_adv.requires_grad = True
            real_adv[:, 2] = real_time_adv
            pred = model._forward_impl(real_adv)
            losses = F.cross_entropy(pred, target_label)
            real_g = torch.autograd.grad(losses, real_adv,
                                         retain_graph=False, create_graph=False)[0]
            with torch.no_grad():
                # Linf step
                if self.targeted:
                    real_time_adv = real_time_adv - torch.sign(real_g[:, 2]) * self.step_size
                else:
                    real_time_adv = real_time_adv + torch.sign(real_g[:, 2]) * self.step_size

            # Linf project
            real_time_adv = torch.where(real_time_adv > lower_bound, real_time_adv, lower_bound).detach()
            real_time_adv = torch.where(real_time_adv < upper_bound, real_time_adv, upper_bound).detach()
            real_time_adv = torch.clamp(real_time_adv, min=0.0, max=1.0)

        real_adv[:, 2] = real_time_adv
        real_adv = real_adv.detach()

        # Generating additional adversarial events
        # event = image_clean.clone().detach()
        event = real_adv.clone().detach()
        null_event = self.make_null_event2(event, 1, voxel_dimension=self.voxel_dimension)
        adv = torch.cat([event, null_event], dim=0)

        null_adv = null_event[:, 2]

        # get null_g
        null_adv.requires_grad = True
        adv[:, 2] = torch.cat([event[:, 2], null_adv], dim=0)
        pred = model._forward_impl(adv)
        losses = F.cross_entropy(pred, target_label)
        null_g = torch.autograd.grad(losses, null_adv, retain_graph=False, create_graph=False)[0]
        with torch.no_grad():
            # Linf step
            if self.targeted:
                null_adv = null_adv - null_g * self.topp
            else:
                null_adv = null_adv + null_g * self.topp

        # generating additional adversarial events
        adam_adv = null_event[self.get_top_percentile(null_g, batch_size=int((1+torch.max(image_clean[:, -1])).item()))]
        adam_adv = adam_adv.repeat_interleave(self.null, dim=0)
        adam_adv = adam_adv.detach()
        # time_adv = 0.5 + 0.01 * torch.rand_like(adam_adv[:, 2]) # time_adv = 0.5* torch.ones_like(adam_adv[:, 2]) # time_adv = 0.5 + 0.05 * torch.rand_like(adam_adv[:, 2]) # time_adv = torch.rand_like(adam_adv[:, 2])  # time_adv = 0.5* torch.ones_like(adam_adv[:, 2])
        time_adv = torch.rand_like(adam_adv[:, 2])
        # time_adv = 0.5 + 0.01 * torch.rand_like(adam_adv[:, 2])
        time_adv = time_adv.detach()
        time_adv.requires_grad = True

        adam_adv[:, 2] = time_adv
        # adv = torch.cat([event, adam_adv], dim=0)

        optimizer = torch.optim.SGD([time_adv], lr=0.01)
        # optimizer = torch.optim.Adam([time_adv], lr=0.01)

        for i in range(20):
            #             time_adv.requires_grad = True
            adam_adv[:, 2] = time_adv
            adv = torch.cat([event, adam_adv], dim=0)
            optimizer.zero_grad()
            pred = model._forward_impl(adv)

            if self.targeted:
                losses = F.cross_entropy(pred, target_label)
                losses.backward(retain_graph=True)
                optimizer.step()
            else:
                losses = -F.cross_entropy(pred, target_label)
                losses.backward(retain_graph=True)
                optimizer.step()

            time_adv = torch.clamp(time_adv, min=0.0, max=1.0)
            adam_adv[:, 2] = time_adv

        real_adv = real_adv.detach()
        adam_adv = adam_adv.detach()
        adv = torch.cat([real_adv, adam_adv], dim=0)
        adv = adv.detach()

        return adv, target_label

    def pgd_attack4(self, image_clean, label, model):
        if self.targeted:
            target_label = self._create_random_target(label)
        else:
            target_label = label

        step_size = self.step_size / model.voxel_dimension[0]
        epsilon = self.epsilon / model.voxel_dimension[0]

        lower_bound = torch.clamp(image_clean[:, 2] - self.epsilon, min=0., max=1.)
        upper_bound = torch.clamp(image_clean[:, 2] + self.epsilon, min=0., max=1.)

        adv = image_clean.clone().detach()
        adv.requires_grad = True
        for i in range(self.num_iter):
            adv.requires_grad = True
            pred = model._forward_impl(adv)
            losses = F.cross_entropy(pred, target_label)
            g = torch.autograd.grad(losses, adv,
                                    retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                # Linf step
                if self.targeted:
                    adv = adv - torch.sign(g) * step_size
                else:
                    adv = adv + torch.sign(g) * step_size

            # Linf project
            adv[:, 2] = torch.where(adv[:, 2] > lower_bound, adv[:, 2], lower_bound).detach()
            adv[:, 2] = torch.where(adv[:, 2] < upper_bound, adv[:, 2], upper_bound).detach()
            adv = adv.detach()
            target_label = target_label.detach()
        # Generating additional adversarial events
        # TODO: 0925
        event = adv.clone().detach()
        null_event = self.make_null_event(event, 1, voxel_dimension=self.voxel_dimension)
        adv = torch.cat([event, null_event], dim=0)

        return adv, target_label

