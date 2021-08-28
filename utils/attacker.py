import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
from utils.loss import cross_entropy_loss_and_accuracy
import tqdm

z_table = {0: 100, 0.1: 3.08, 0.5: 2.57, 1: 2.33, 2: 2.05, 3: 1.88, 4: 1.75,5: 1.65, 10: 1.28, 15: 1.03, 20: 0.84, 25: 0.67}


class PGDAttacker():
    def __init__(self, num_iter, epsilon, step_size, event_step_size, num_classes=101, voxel_dimension=(9, 180, 240), targeted=False):
        self.num_iter = num_iter
        self.epsilon = epsilon
        self.step_size = step_size
        self.event_step_size = event_step_size
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

    def make_null_event(self, events, T, voxel_dimension):
        B = int((1 + events[-1, -1]).item())
        C, H, W = voxel_dimension
        vox = torch.zeros((W, H, T, 2, B))
        null_event = (vox == 0).nonzero().float().cuda()
        null_event[:, 2] = 0
        null_event[:, 3] = (null_event[:, 3] - 0.5) * 2
        return null_event

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
        elif mode == 'event_pgd':
            return self.pgd_attack3(image_clean, label, model)
        elif mode == 'event_adam':
            return self.pgd_attack4(image_clean, label, model)

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
                    adv = adv - torch.sign(g) * self.step_size
                else:
                    adv = adv + torch.sign(g) * self.step_size

            # Linf project
            adv = adv.detach()
            target_label = target_label.detach()

        return adv, target_label


    def pgd_attack2(self, image_clean, label, model):
        if self.targeted:
            target_label = self._create_random_target(label)
        else:
            target_label = label

        adv = image_clean.clone().detach()
        adv = self.fill_null_event(adv, T=self.epsilon, H=180, W=240)

        # event adv should be 0 or 1
        event_adv = (adv[:, 2] != 0).float()
        event_adv.requires_grad = True

        # if adv==0: 0 < time < 1 random uniform
        # time_adv = torch.where(adv[:, 2] == 0, 0.01*torch.rand_like(adv[:, 2]), adv[:, 2])
        time_adv = torch.where(adv[:, 2] == 0, 0.5*torch.ones_like(adv[:, 2]), adv[:, 2])
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
                    # event_adv = event_adv - event_g * self.event_step_size
                    event_adv = event_adv - torch.clamp(-event_g, 0) * self.event_step_size  # targeted
                    time_adv = time_adv - torch.sign(time_g) * self.step_size
                else:
                    event_adv = event_adv + torch.clamp(event_g, 0) * self.event_step_size  # targeted
                    time_adv = time_adv + torch.sign(time_g) * self.step_size  # targeted
            # Linf project
            event_adv = torch.where(event_adv < 0.5, torch.zeros_like(event_adv),
                                    torch.ones_like(event_adv)).float().detach()
            time_adv = torch.clamp(time_adv, min=0.0, max=1.0)
            adv[:, 2] = event_adv * time_adv
            event_adv = event_adv.detach()
            time_adv = time_adv.detach()

        return adv, target_label

    def pgd_attack3(self, image_clean, label, model):
        if self.targeted:
            target_label = self._create_random_target(label)
        else:
            target_label = label

        event = image_clean.clone().detach()

        null_event = self.make_null_event(event, T=self.epsilon, voxel_dimension=self.voxel_dimension)
        adv = torch.cat([event, null_event], dim=0)

        real_adv = event[:, 2]
        null_adv = null_event[:, 2]

        real_adv.requires_grad = True
        null_adv.requires_grad = True

        for i in range(self.num_iter):
            real_adv.requires_grad = True
            null_adv.requires_grad = True
            adv[:, 2] = torch.cat([real_adv, null_adv], dim=0)
            pred = model._forward_impl(adv)
            losses = F.cross_entropy(pred, target_label)
            real_g = torch.autograd.grad(losses, real_adv,
                                         retain_graph=True, create_graph=False)[0]
            null_g = torch.autograd.grad(losses, null_adv,
                                         retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                # Linf step
                if self.targeted:
                    real_adv = real_adv - torch.sign(real_g) * self.step_size
                    null_adv = null_adv - null_g * self.event_step_size
                else:
                    real_adv = real_adv + torch.sign(real_g) * self.step_size
                    null_adv = null_adv + null_g * self.event_step_size

            # Linf project
            null_adv = torch.where(null_adv < 0.5, torch.zeros_like(null_adv),
                                   torch.ones_like(null_adv)).detach()
            real_adv = torch.clamp(real_adv, min=0.0, max=1.0)

        event[:, 2] = real_adv

        adam_adv = null_event[null_adv == 1, :]
        time_adv = 0.5 + 0.05 * torch.rand_like(adam_adv[:, 2])  # time_adv = 0.5* torch.ones_like(adam_adv[:, 2])
        time_adv.requires_grad = True

        adam_adv[:, 2] = time_adv

        for i in range(100):
            time_adv.requires_grad = True
            adam_adv[:, 2] = time_adv
            pred = model._forward_impl(adam_adv)
            losses = F.cross_entropy(pred, target_label)
            g = torch.autograd.grad(losses, time_adv,
                                    retain_graph=False, create_graph=False)[0]

            with torch.no_grad():
                # Linf step
                if self.targeted:
                    time_adv = time_adv - torch.sign(g) * self.step_size
                else:
                    time_adv = time_adv + torch.sign(g) * self.step_size

            time_adv = torch.clamp(time_adv, min=0.0, max=1.0)
            adam_adv[:, 2] = time_adv

            adv = torch.cat([event, adam_adv], dim=0)
            event = event.detach()
            adam_adv = adam_adv.detach()
            adv = adv.detach()

        return adv, target_label

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

    def get_top_percentile(self, null_g, batch_size):
        i = 0
        boolean_g = 0
        threshold = z_table[self.event_step_size]
        for g in torch.split(null_g, int(null_g.shape[0]/batch_size)):
            if i != 0:
                boolean_g = torch.cat([boolean_g, (g-torch.mean(g))/torch.std(g) > threshold])
            else:
                boolean_g = (g-torch.mean(g))/torch.std(g) > threshold
            i += 1

        return boolean_g

    def pgd_attack4(self, image_clean, label, model):
        if self.targeted:
            target_label = self._create_random_target(label)
        else:
            target_label = label

        # Typical pgd attack for existing event
        real_adv = image_clean.clone().detach()
        real_time_adv = real_adv[:, 2]
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
            #             null_adv = torch.where(null_adv < 0.5, torch.zeros_like(null_adv),
            #                                     torch.ones_like(null_adv)).detach()
            real_time_adv = torch.clamp(real_time_adv, min=0.0, max=1.0)

        real_adv[:, 2] = real_time_adv

        # Generating additional adversarial events
        event = image_clean.clone().detach()
        null_event = self.make_null_event(event, T=self.epsilon, voxel_dimension=self.voxel_dimension)
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
                null_adv = null_adv - null_g * self.event_step_size
            else:
                null_adv = null_adv + null_g * self.event_step_size

        # generating additional adversarial events
        adam_adv = null_event[self.get_top_percentile(null_g, batch_size=int((1+torch.max(image_clean[:, -1])).item()))]
        # time_adv = 0.5 + 0.01 * torch.rand_like(adam_adv[:, 2]) # time_adv = 0.5* torch.ones_like(adam_adv[:, 2]) # time_adv = 0.5 + 0.05 * torch.rand_like(adam_adv[:, 2]) # time_adv = torch.rand_like(adam_adv[:, 2])  # time_adv = 0.5* torch.ones_like(adam_adv[:, 2])
        time_adv = torch.rand_like(adam_adv[:, 2])
        time_adv = time_adv.detach()
        time_adv.requires_grad = True

        adam_adv[:, 2] = time_adv
        adv = torch.cat([event, adam_adv], dim=0)

        optimizer = torch.optim.Adam([time_adv], lr=0.01)

        for i in range(10):
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

        adv = torch.cat([real_adv, adam_adv], dim=0)
        real_adv = real_adv.detach()
        adam_adv = adam_adv.detach()
        adv = adv.detach()

        return adv, target_label


    # def pgd_attack4(self, image_clean, label, model):
    #     if self.targeted:
    #         target_label = self._create_random_target(label)
    #     else:
    #         target_label = label
    #
    #     event = image_clean.clone().detach()
    #
    #     null_event = self.make_null_event(event, T=self.epsilon, H=180, W=240)
    #     adv = torch.cat([event, null_event], dim=0)
    #
    #     real_adv = event[:, 2]
    #     null_adv = null_event[:, 2]
    #
    #     real_adv.requires_grad = True
    #     null_adv.requires_grad = True
    #
    #     for i in range(self.num_iter):
    #         real_adv.requires_grad = True
    #         null_adv.requires_grad = True
    #         adv[:, 2] = torch.cat([real_adv, null_adv], dim=0)
    #         pred = model._forward_impl(adv)
    #         losses = F.cross_entropy(pred, target_label)
    #         real_g = torch.autograd.grad(losses, real_adv,
    #                                      retain_graph=True, create_graph=False)[0]
    #         null_g = torch.autograd.grad(losses, null_adv,
    #                                      retain_graph=False, create_graph=False)[0]
    #
    #         with torch.no_grad():
    #             # Linf step
    #             if self.targeted:
    #                 real_adv = real_adv - torch.sign(real_g) * self.step_size
    #                 null_adv = null_adv - null_g * self.event_step_size
    #             else:
    #                 real_adv = real_adv + torch.sign(real_g) * self.step_size
    #                 null_adv = null_adv + null_g * self.event_step_size
    #
    #         # Linf project
    #         #             null_adv = torch.where(null_adv < 0.5, torch.zeros_like(null_adv),
    #         #                                     torch.ones_like(null_adv)).detach()
    #         real_adv = torch.clamp(real_adv, min=0.0, max=1.0)
    #
    #     event[:, 2] = real_adv
    #     event = event.detach()
    #
    #     ############## additional adversarial event ###################
    #
    #     event[:, 2] = real_adv
    #     event = event.detach()
    #     # event.requires_grad = False
    #     #
    #     # adam_adv = null_event[self.get_top_percentile(null_g, batch_size=4)]
    #     #
    #     # #         return null_event, null_g, adam_adv
    #     #
    #     # #         adam_adv = null_event[(null_g-torch.mean(null_g))/torch.std(null_g)>1.28]
    #     # time_adv = 0.5 + 0.05 * torch.rand_like(adam_adv[:, 2])
    #     # # time_adv = torch.rand_like(adam_adv[:, 2])
    #     # #         time_adv = 0.5* torch.ones_like(adam_adv[:, 2])
    #     # time_adv = time_adv.detach()
    #     # time_adv.requires_grad = True
    #     #
    #     # adam_adv[:, 2] = time_adv
    #     # adv = torch.cat([event, adam_adv], dim=0)
    #     #
    #     # optimizer = torch.optim.Adam([time_adv], lr=0.01)
    #     #
    #     # for i in range(10):
    #     #     #             time_adv.requires_grad = True
    #     #     adam_adv[:, 2] = time_adv
    #     #     adv = torch.cat([event, adam_adv], dim=0)
    #     #     optimizer.zero_grad()
    #     #
    #     #     pred = model._forward_impl(adv)
    #     #
    #     #     if self.targeted:
    #     #         losses = F.cross_entropy(pred, target_label)
    #     #         losses.backward(retain_graph=True)
    #     #         optimizer.step()
    #     #     else:
    #     #         losses = -F.cross_entropy(pred, target_label)
    #     #         losses.backward(retain_graph=True)
    #     #         optimizer.step()
    #     #
    #     #     time_adv = torch.clamp(time_adv, min=0.0, max=1.0)
    #     #     adam_adv[:, 2] = time_adv
    #     #
    #     # adv = torch.cat([event, adam_adv], dim=0)
    #     # event = event.detach()
    #     # adam_adv = adam_adv.detach()
    #     # adv = adv.detach()
    #     ############## additional adversarial event ###################
    #
    #     adv = event
    #
    #     return adv, target_label
    #
    #
    # def pgd_attack5(self, image_clean, label, model):
    #     if self.targeted:
    #         target_label = self._create_random_target(label)
    #     else:
    #         target_label = label
    #
    #     event = image_clean.clone().detach()
    #     null_event = self.make_null_event(event, T=self.epsilon, H=180, W=240)
    #     adv = torch.cat([event, null_event], dim=0)
    #
    #     real_adv = event[:, 2]
    #     null_adv = null_event[:, 2]
    #
    #     # get null_g
    #     null_adv.requires_grad = True
    #     adv[:, 2] = torch.cat([real_adv, null_adv], dim=0)
    #     pred = model._forward_impl(adv)
    #     losses = F.cross_entropy(pred, target_label)
    #     null_g = torch.autograd.grad(losses, null_adv, retain_graph=False, create_graph=False)[0]
    #     with torch.no_grad():
    #         # Linf step
    #         if self.targeted:
    #             null_adv = null_adv - null_g * self.event_step_size
    #         else:
    #             null_adv = null_adv + null_g * self.event_step_size
    #
    #     # generating additional adversarial events
    #     adam_adv = null_event[self.get_top_percentile(null_g, batch_size=4)]
    #     time_adv = 0.5 + 0.05 * torch.rand_like(adam_adv[:, 2]) # time_adv = torch.rand_like(adam_adv[:, 2])  # time_adv = 0.5* torch.ones_like(adam_adv[:, 2])
    #     time_adv = time_adv.detach()
    #     time_adv.requires_grad = True
    #
    #     adam_adv[:, 2] = time_adv
    #     adv = torch.cat([event, adam_adv], dim=0)
    #
    #     optimizer = torch.optim.Adam([time_adv], lr=0.01)
    #
    #     for i in range(10):
    #         #             time_adv.requires_grad = True
    #         adam_adv[:, 2] = time_adv
    #         adv = torch.cat([event, adam_adv], dim=0)
    #         optimizer.zero_grad()
    #         pred = model._forward_impl(adv)
    #
    #         if self.targeted:
    #             losses = F.cross_entropy(pred, target_label)
    #             losses.backward(retain_graph=True)
    #             optimizer.step()
    #         else:
    #             losses = -F.cross_entropy(pred, target_label)
    #             losses.backward(retain_graph=True)
    #             optimizer.step()
    #
    #         time_adv = torch.clamp(time_adv, min=0.0, max=1.0)
    #         adam_adv[:, 2] = time_adv
    #
    #     # typical pgd attack for existing event
    #     real_adv.requires_grad = True
    #     for i in range(self.num_iter):
    #         real_adv.requires_grad = True
    #         event[:, 2] = real_adv
    #         pred = model._forward_impl(event)
    #         losses = F.cross_entropy(pred, target_label)
    #         real_g = torch.autograd.grad(losses, real_adv,
    #                                      retain_graph=True, create_graph=False)[0]
    #         with torch.no_grad():
    #             # Linf step
    #             if self.targeted:
    #                 real_adv = real_adv - torch.sign(real_g) * self.step_size
    #             else:
    #                 real_adv = real_adv + torch.sign(real_g) * self.step_size
    #
    #         # Linf project
    #         #             null_adv = torch.where(null_adv < 0.5, torch.zeros_like(null_adv),
    #         #                                     torch.ones_like(null_adv)).detach()
    #         real_adv = torch.clamp(real_adv, min=0.0, max=1.0)
    #
    #     event[:, 2] = real_adv
    #     adv = torch.cat([event, adam_adv], dim=0)
    #     event = event.detach()
    #     adam_adv = adam_adv.detach()
    #     adv = adv.detach()
    #
    #     return adv, target_label

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
