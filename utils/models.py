import torch.nn as nn
from os.path import join, dirname, isfile
import torch
import torch.nn.functional as F
import numpy as np
from torchvision.models.resnet import resnet34
from utils.loss import cross_entropy_loss_and_accuracy
import tqdm


class ValueLayer(nn.Module):
    def __init__(self, mlp_layers, activation=nn.ReLU(), num_channels=9):
        assert mlp_layers[-1] == 1, "Last layer of the mlp must have 1 input channel."
        assert mlp_layers[0] == 1, "First layer of the mlp must have 1 output channel"

        nn.Module.__init__(self)
        self.mlp = nn.ModuleList()
        self.activation = activation

        # create mlp
        in_channels = 1
        for out_channels in mlp_layers[1:]:
            self.mlp.append(nn.Linear(in_channels, out_channels))
            in_channels = out_channels

        # init with trilinear kernel
        path = join(dirname(__file__), "quantization_layer_init", "trilinear_init.pth")
        if isfile(path):
            state_dict = torch.load(path)
            self.load_state_dict(state_dict)
        else:
            self.init_kernel(num_channels)

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply mlp convolution
        for i in range(len(self.mlp[:-1])):
            x = self.activation(self.mlp[i](x))

        x = self.mlp[-1](x)
        x = x.squeeze()

        return x

    def init_kernel(self, num_channels):
        ts = torch.zeros((1, 2000))
        optim = torch.optim.Adam(self.parameters(), lr=1e-2)

        torch.manual_seed(1)

        for _ in tqdm.tqdm(range(1000)):  # converges in a reasonable time
            optim.zero_grad()

            ts.uniform_(-1, 1)

            # gt
            gt_values = self.trilinear_kernel(ts, num_channels)

            # pred
            values = self.forward(ts)

            # optimize
            loss = (values - gt_values).pow(2).sum()

            loss.backward()
            optim.step()


    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        gt_values[ts < -1.0 / (num_channels-1)] = 0
        gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values


class TrilinearLayer(nn.Module):
    def __init__(self, num_channels=9):
        nn.Module.__init__(self)
        self.num_channels = num_channels

    def forward(self, x):
        # create sample of batchsize 1 and input channels 1
        x = x[None,...,None]

        # apply trilineaer kernel
        x = self.trilinear_kernel(x, self.num_channels)
        x = x.squeeze()

        return x

    def trilinear_kernel(self, ts, num_channels):
        gt_values = torch.zeros_like(ts)

        gt_values[ts > 0] = (1 - (num_channels-1) * ts)[ts > 0]
        gt_values[ts < 0] = ((num_channels-1) * ts + 1)[ts < 0]

        if num_channels == 1:
            gt_values[ts < -1.0] = 0
            gt_values[ts > 1.0] = 0
        else:
            gt_values[ts < -1.0 / (num_channels-1)] = 0
            gt_values[ts > 1.0 / (num_channels-1)] = 0

        return gt_values

class NoneLayer(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)
    def forward(self, x):
        return x


class QuantizationLayer(nn.Module):
    def __init__(self, dim,
                 mlp_layers=[1, 100, 100, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 value_layer="ValueLayer",
                 projection=None):
        nn.Module.__init__(self)
        if value_layer == "NoneLayer":
            self.value_layer = NoneLayer()
        elif value_layer == "ValueLayer":
            self.value_layer = ValueLayer(mlp_layers, activation=activation, num_channels=dim[0])
        elif value_layer == "TrilinearLayer":
            self.value_layer = TrilinearLayer(num_channels=dim[0])

        self.dim = dim
        self.projection = projection

    def forward(self, events):
        # points is a list, since events can have any size
        B = int((1+events[-1,-1]).item())
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p, b = events.t()

        # normalizing timestamps
        for bi in range(B):
            t[events[:,-1] == bi] /= t[events[:,-1] == bi].max()

        p = (p+1)/2  # maps polarity to 0, 1

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b

        if C == 1:
            # values = t * self.value_layer.forward(t)
            # if values == t:
            #     print("same")
            # else:
            #     print("not same")
            values = t
            idx = idx_before_bins
            vox.put_(idx.long(), values, accumulate=True)
        else:
            for i_bin in range(C):
                values = t * self.value_layer.forward(t-i_bin/(C-1))

                # draw in voxel grid
                idx = idx_before_bins + W * H * i_bin
                vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        if self.projection == None:
            pass
        elif self.projection == "polarity":
            vox = self.project_polarity(vox)
        elif self.projection == "average_time":
            vox = self.project_average_time(vox)
        elif self.projection == "recent_time":
            vox = self.project_recent_time(vox)
        elif self.projection == "time_count":
            vox = self.project_time_count(vox)

        return vox

    def project_polarity(self, vox):
        # Voxel grid: Voxel grid summing event polarities
        B, C, H, W = vox.size()
        vox = vox[:, :C//2, :, :] + vox[:, C//2:, :, :]
        return vox

    def project_average_time(self, vox):
        # HATS: Histogram of average time surfaces
        B, C, H, W = vox.size()
        pvox = torch.mean(vox[:, :C//2, :, :], dim=1, keepdim=True)
        nvox = torch.mean(vox[:, C//2:, :, :], dim=1, keepdim=True)
        vox = torch.cat([pvox, nvox], 1)
        return vox

    def project_recent_time(self, vox):
        # SAE: Image of most recent time stamp
        B, C, H, W = vox.size()
        pvox, _ = torch.max(vox[:, :C // 2, :, :], dim=1, keepdim=True)
        nvox, _ = torch.max(vox[:, C // 2:, :, :], dim=1, keepdim=True)
        vox = torch.cat([pvox, nvox], 1)
        return vox

    def project_time_count(self, vox):
        # EV-FlowNet, Event self-driving: Image of event counts
        B, C, H, W = vox.size()
        pvox = torch.count_nonzero(vox[:, :C // 2, :, :], dim=1)
        pvox = pvox.view(B, 1, H, W)
        nvox = torch.count_nonzero(vox[:, C // 2:, :, :], dim=1)
        nvox = nvox.view(B, 1, H, W)
        vox = torch.cat([pvox, nvox], 1)
        return vox


class Classifier(nn.Module):
    def __init__(self,
                 voxel_dimension=(9,180,240),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 value_layer="ValueLayer",
                 projection=None,
                 pretrained=True,
                 adv=False,
                 adv_test=False,
                 epsilon=4,
                 num_iter=2,
                 step_size=0.5):

        nn.Module.__init__(self)
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation, value_layer, projection)
        self.classifier = resnet34(pretrained=pretrained)

        self.crop_dimension = crop_dimension
        self.num_classes = num_classes
        self.adv = adv
        self.adv_test = adv_test
        self.epsilon = epsilon
        self.num_iter = num_iter
        self.step_size = step_size
        self.projection = projection

        # replace fc layer and first convolutional layer
        if self.projection == None:
            input_channels = 2 * voxel_dimension[0]
        elif self.projection == 'polarity':
            input_channels = voxel_dimension[0]
        else:
            input_channels = 2
        self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)

    def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
        B, C, H, W = x.shape
        if H > W:
            h = H // 2
            x = x[:, :, h - W // 2:h + W // 2, :]
        else:
            h = W // 2
            x = x[:, :, :, h - H // 2:h + H // 2]

        x = F.interpolate(x, size=output_resolution)

        return x

    def pgd_attack(self, image_clean, label, num_iter=2, step_size=0.1, epsilon=4, original=False):
        if original:
            target_label = label  # untargeted
        else:
            target_label = self._create_random_target(label)  # targeted

        adv = image_clean.clone().detach()
        adv.requires_grad = True
        for i in range(num_iter):

            pred = self.classifier.forward(adv)
            losses, accuracy = cross_entropy_loss_and_accuracy(pred, target_label)
            g = torch.autograd.grad(losses, adv,
                                    retain_graph=False, create_graph=False)[0]

            g_topk = self.get_topk(g, epsilon)
            # g_topk = g
            if self.projection != "polarity":
                g_topk = self.apply_polarity_constraint(g_topk)

            # Linf step
            if original:
                adv = adv + torch.sign(g_topk) * step_size  # untargeted
            else:
                adv = adv - torch.sign(g_topk) * step_size  # targeted

            # # Linf step
            # if original:
            #     adv = adv + g_topk * step_size  # untargeted
            # else:
            #     adv = adv - g_topk * step_size  # targeted

            # Linf project for time??

        return adv.detach(), target_label

    def random_attack(self, image_clean, label, num_iter=2, step_size=0.1, epsilon=4, original=False):
        if original:
            target_label = label  # untargeted
        else:
            target_label = self._create_random_target(label)  # targeted

        adv = image_clean.clone().detach()
        adv.requires_grad = True
        for i in range(num_iter):
            pred = self.classifier.forward(adv)
            losses, accuracy = cross_entropy_loss_and_accuracy(pred, target_label)
            g = torch.autograd.grad(losses, adv,
                                    retain_graph=False, create_graph=False)[0]

            g_randk = self.get_randk(g, epsilon)
            g_randk = self.apply_polarity_constraint(g_randk)
            # Linf step
            if original:
                adv = adv + torch.sign(g_randk) * step_size  # untargeted
            else:
                adv = adv - torch.sign(g_randk) * step_size  # targeted

            # Linf project for time??

        return adv, target_label

    def _create_random_target(self, label):
        label_offset = torch.randint_like(label, low=0, high=self.num_classes)
        return (label + label_offset) % self.num_classes

    # def get_topk(self, x, n=4):
    #     B, C, H, W = x.size()
    #     x = torch.reshape(x, (B, C, H * W))
    #     #         topk = torch.zeros(B, C, H*W)
    #     topk = torch.zeros_like(x)
    #     _, idx, = torch.topk(torch.abs(x), n)
    #     for i in range(B):
    #         for j in range(C):
    #             for k in idx[i, j]:
    #                 topk[i, j, k] = x[i, j, k]
    #     topk = torch.reshape(topk, (B, C, H, W))
    #     return topk

    def get_topk(self, x, n=4):
        B, C, H, W = x.size()
        x = torch.reshape(x, (B, C, H * W))
        #         topk = torch.zeros(B, C, H*W)
        topk = torch.zeros_like(x)
        _, idx, = torch.topk(torch.abs(x), n)
        for i in range(B):
            for j in range(C):
                for k in idx[i, j]:
                    topk[i, j, k] = x[i, j, k]
        topk = torch.reshape(topk, (B, C, H, W))
        return topk

    def get_randk(self, x, n=4):
        B, C, H, W = x.size()
        x = torch.reshape(x, (B, C, H * W))
        #         topk = torch.zeros(B, C, H*W)
        topk = torch.zeros_like(x)
        _, idx, = torch.topk(x, n)
        for i in range(B):
            for j in range(C):
                for _ in idx[i, j]:
                    l = torch.randint(low=0, high=H * W - 1, size=(1,))
                    topk[i, j, l] = x[i, j, l]
        topk = torch.reshape(topk, (B, C, H, W))
        return topk

    def apply_polarity_constraint(self, x):
        B, C, H, W = x.size()
        p = x[:, :C // 2, :, :]
        n = x[:, C // 2:, :, :]
        p = torch.where(torch.abs(p) == torch.abs(n), torch.zeros_like(p), p)
        x = torch.cat((p, n), dim=1)

        return x

    def forward(self, x, labels=None):

        training = self.training
        if training:
            vox = self.quantization_layer.forward(x)
            vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
            if self.adv == True:
                self.classifier.eval()
                adv, target_label = self.pgd_attack(vox_cropped, labels,
                                                    self.num_iter, self.step_size, self.epsilon)
                vox_cropped = torch.cat((vox_cropped, adv), dim=0)
                labels = torch.cat((labels, labels), dim=0)
                self.classifier.train()
                pred = self.classifier.forward(vox_cropped)
                return pred, labels
            else:
                pred = self.classifier.forward(vox_cropped)
                return pred, labels
        else:
            vox = self.quantization_layer.forward(x)
            vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
            if self.adv_test == True:
                self.classifier.eval()
                with torch.no_grad():
                    pred = self.classifier.forward(vox_cropped)

                adv, target_label = self.pgd_attack(vox_cropped, labels,
                                                    self.num_iter, self.step_size, self.epsilon)
                with torch.no_grad():
                    adv_pred = self.classifier.forward(adv)
                return (pred, adv_pred), (labels, target_label)
            else:
                pred = self.classifier.forward(vox_cropped)
                return pred, labels
            # vox = self.quantization_layer.forward(x)
            # vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
            # pred = self.classifier.forward(vox_cropped)
            return pred, labels

# class Classifier(nn.Module):
#     def __init__(self,
#                  voxel_dimension=(9, 180, 240),  # dimension of voxel will be C x 2 x H x W
#                  crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
#                  num_classes=101,
#                  mlp_layers=[1, 30, 30, 1],
#                  activation=nn.LeakyReLU(negative_slope=0.1),
#                  pretrained=True):
#
#         nn.Module.__init__(self)
#         self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation)
#         self.classifier = resnet34(pretrained=pretrained)
#
#         self.crop_dimension = crop_dimension
#
#         # replace fc layer and first convolutional layer
#         input_channels = 2 * voxel_dimension[0]
#         self.classifier.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
#         self.classifier.fc = nn.Linear(self.classifier.fc.in_features, num_classes)
#
#     def crop_and_resize_to_resolution(self, x, output_resolution=(224, 224)):
#         B, C, H, W = x.shape
#         if H > W:
#             h = H // 2
#             x = x[:, :, h - W // 2:h + W // 2, :]
#         else:
#             h = W // 2
#             x = x[:, :, :, h - H // 2:h + H // 2]
#
#         x = F.interpolate(x, size=output_resolution)
#
#         return x
#
#     def forward(self, x):
#         vox = self.quantization_layer.forward(x)
#         vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
#         pred = self.classifier.forward(vox_cropped)
#         return pred, vox

