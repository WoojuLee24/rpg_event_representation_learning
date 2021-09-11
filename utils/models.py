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
        B = int((1+events[:, -1].max()).item())
        # B = 2
        num_voxels = int(2 * np.prod(self.dim) * B)
        vox = events[0].new_full([num_voxels,], fill_value=0)
        C, H, W = self.dim

        # get values for each channel
        x, y, t, p, b = events.t()

        p = (p+1)/2  # maps polarity to 0, 1

        idx_before_bins = x \
                          + W * y \
                          + 0 \
                          + W * H * C * p \
                          + W * H * C * 2 * b

        if C == 1:
            values = t
            idx = idx_before_bins
            vox.put_(idx.long(), values, accumulate=True)
        else:
            for i_bin in range(C):
                values = t * self.value_layer.forward(t-i_bin/(C-1))

                # draw in voxel grid
                idx = idx_before_bins + W * H * i_bin
                max_idx = idx.max()
                max_x = events[:, 0].max()
                max_y = events[:, 1].max()
                vox.put_(idx.long(), values, accumulate=True)

        vox = vox.view(-1, 2, C, H, W)
        vox = torch.cat([vox[:, 0, ...], vox[:, 1, ...]], 1)

        if self.projection == None:
            pass
        elif self.projection == "polarity":
            # voxel grid
            vox = self.project_polarity(vox)
        elif self.projection == "average_time":
            # Two-channel image
            vox = self.project_average_time(vox)
        elif self.projection == "recent_time":
            # Two-channel image
            vox = self.project_recent_time(vox)
        elif self.projection == "time_count":
            # Event frame
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
        # pvox = torch.count_nonzero(vox[:, :C // 2, :, :], dim=1)
        pvox = (vox[:, :C // 2, :, :] != 0).sum(dim=1).to(torch.float32)
        pvox = pvox.view(B, 1, H, W)
        # nvox = torch.count_nonzero(vox[:, C // 2:, :, :], dim=1)
        nvox = (vox[:, C // 2:, :, :] != 0).sum(dim=1).to(torch.float32)
        nvox = nvox.view(B, 1, H, W)
        vox = torch.cat([pvox, nvox], 1)
        return vox


class ESTNet(nn.Module):
    def __init__(self,
                 voxel_dimension=(9,180,240),  # dimension of voxel will be C x 2 x H x W
                 crop_dimension=(224, 224),  # dimension of crop before it goes into classifier
                 num_classes=101,
                 mlp_layers=[1, 30, 30, 1],
                 activation=nn.LeakyReLU(negative_slope=0.1),
                 value_layer="ValueLayer",
                 projection=None,
                 pretrained=True,
                 epsilon=4,
                 num_iter=2,
                 step_size=0.5):

        nn.Module.__init__(self)
        self.quantization_layer = QuantizationLayer(voxel_dimension, mlp_layers, activation, value_layer, projection)
        self.classifier = resnet34(pretrained=pretrained)

        self.voxel_dimension = voxel_dimension
        self.crop_dimension = crop_dimension
        self.num_classes = num_classes
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


    def _forward_impl(self, x):
        vox = self.quantization_layer.forward(x)
        vox_cropped = self.crop_and_resize_to_resolution(vox, self.crop_dimension)
        pred = self.classifier.forward(vox_cropped)
        return pred


class AdvESTNet(ESTNet):
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
                 attack_mode='event'):
        super().__init__(voxel_dimension, crop_dimension, num_classes, mlp_layers, activation, value_layer, projection, pretrained)
        self.adv = adv
        self.adv_test = adv_test
        self.attack_mode = attack_mode

    def set_attacker(self, attacker):
        self.attacker = attacker

    def forward(self, x, labels):
        training = self.training
        if training:
            if self.adv == True:
                self.eval()
                adv_images, _ = self.attacker.attack(x, labels, self, mode=self.attack_mode)
                with torch.no_grad():
                    adv_images[:, 4] += (x[:, -1].max() + 1)  # adv batch_size
                    x = torch.cat([x, adv_images], dim=0)
                    labels = torch.cat([labels, labels], dim=0)
                self.train()
                return self._forward_impl(x), labels
            else:
                self.train()
                return self._forward_impl(x), labels
        else:
            # adv_test
            if self.adv_test == True:
                adv_images, target_labels = self.attacker.attack(x, labels, self, mode=self.attack_mode)
                with torch.no_grad():
                    pred = self._forward_impl(x)
                    k = pred.argmax(1)
                    adv_pred = self._forward_impl(adv_images)
                return (pred, adv_pred), (labels, target_labels)
            else:
                return self._forward_impl(x), labels
