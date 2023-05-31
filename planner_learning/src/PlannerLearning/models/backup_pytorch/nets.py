import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
# from keras.applications.imagenet_utils import preprocess_input

def create_network(settings):
    net = PlaNet(settings)
    return net

# class Network(nn.Module):
#     def __init__(self):
#         super(Network, self).__init__()

#     def forward(self, x):
#         return self._internal_call(x)

class PlaNet(nn.Module):
    def __init__(self, config):
        super(PlaNet, self).__init__()
        self.config = config
        self._create(input_size=(3 * self.config.use_rgb + 3*self.config.use_depth,
                                 self.config.img_height,
                                 self.config.img_width))

    def _create(self, input_size, has_bias=True):
        """Init.
        Args:
            input_size (float): size of input
            has_bias (bool, optional): Defaults to True. Conv1d bias?
        """
        if self.config.use_rgb or self.config.use_depth:
            self.backbone = nn.Sequential(models.mobilenet_v2(pretrained=True)) # use pretrained weights
            if self.config.freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            # reduce a bit the size
            self.resize_op = nn.Sequential(
                nn.Conv1d(1000, 128, kernel_size=1, stride=1, padding=0, dilation=1, bias=has_bias)
            )

            f = 1.0
            self.img_mergenet = nn.Sequential(
                nn.Conv1d(1, int(128 * f), kernel_size=2, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(negative_slope=1e-2),
                nn.Conv1d(int(128 * f), int(64 * f), kernel_size=2, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(negative_slope=1e-2),
                nn.Conv1d(int(64 * f), int(64 * f), kernel_size=2, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(negative_slope=1e-2),
                nn.Conv1d(int(64 * f), int(32 * f), kernel_size=2, stride=1, padding=0, dilation=1),
                nn.LeakyReLU(negative_slope=1e-2)
            )

            self.resize_op_2 = nn.Sequential(
                nn.Conv1d(124, self.config.modes, kernel_size=3, stride=1, padding=0, dilation=1, bias=has_bias)
            )

        g = 1.0
        self.states_conv = nn.Sequential(
            nn.Conv1d(self.config.seq_len, int(64 * g), kernel_size=2, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(negative_slope=.5),
            nn.Conv1d(int(64 * g), int(32 * g), kernel_size=2, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(negative_slope=.5),
            nn.Conv1d(int(32 * g), int(32 * g), kernel_size=2, stride=1, padding=0, dilation=1),
            nn.LeakyReLU(negative_slope=.5),
            nn.Conv1d(int(32 * g), int(32 * g), kernel_size=2, stride=1, padding=0, dilation=1)
        )

        self.resize_op_3 = nn.Sequential(
            nn.Conv1d(14, self.config.modes, kernel_size=3, stride=1, padding=0, dilation=1, bias=has_bias)
        )

        if len(self.config.predict_state_number) == 0:
            out_len = self.config.out_seq_len
        else:
            out_len = 1
        output_dim = self.config.state_dim * out_len + 1
        # print("output_dim", output_dim)
        g = 1.0
        self.plan_module = nn.Sequential(
            nn.Conv1d(30, int(64 * g), kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=.5),
            nn.Conv1d(int(64 * g), int(128 * g), kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=.5),
            nn.Conv1d(int(128 * g), int(128 * g), kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(negative_slope=.5),
            nn.Conv1d(int(128 * g), output_dim, kernel_size=1, stride=1, padding=0)
        )

    def _conv_branch(self, image):
        x = self._pf(image)
        x = self.backbone(x)
        # print("before", x.shape)
        x = x.view(x.size(0), x.size(1), -1)  # (batch_size, MxM, C)
        # print("after", x.shape)
        x = self.resize_op(x)
        x = x.view(x.size(0), -1)  # (batch_size, MxMx128)
        return x

    def _image_branch(self, img_seq):
        img_fts = list(map(self._conv_branch, img_seq))
        img_fts = torch.stack(img_fts).permute(1, 0, 2)
        # print(img_fts.shape)
        x = img_fts
        x = self.img_mergenet(x)
        x = x.permute(0, 2, 1)
        x = self.resize_op_2(x)
        x = x.permute(0, 2, 1)
        return x

    def _imu_branch(self, embeddings):
        x = embeddings
        x = self.states_conv(x)
        x = x.permute(0, 2, 1)
        x = self.resize_op_3(x)
        x = x.permute(0, 2, 1)
        return x

    def _plan_branch(self, embeddings):
        x = embeddings
        x = self.plan_module(x)
        return x

    def _pf(self, images):

        # preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                                   std=[0.229, 0.224, 0.225])
        # return preprocess(images)
        # return preprocess_input(images, mode='torch')
        images = images.permute(0, 3, 1, 2)
        images = images.cuda()
        images /= 255.0
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        # print("pf", images.shape)
        images[:, 0, :, :] = (images[:, 0, :, :] - mean[0]) / std[0]
        images[:, 1, :, :] = (images[:, 1, :, :] - mean[1]) / std[1]
        images[:, 2, :, :] = (images[:, 2, :, :] - mean[2]) / std[2]
        return images


    def _preprocess_frames(self, inputs):
        if self.config.use_rgb and self.config.use_depth:
            img_seq = torch.cat((inputs['rgb'], inputs['depth']), dim=-1)
        elif self.config.use_rgb and (not self.config.use_depth):
            img_seq = inputs['rgb']
        elif self.config.use_depth and (not self.config.use_rgb):
            img_seq = inputs['depth']
        else:
            return None

        img_seq = torch.tensor(img_seq).permute(1, 0, 2, 3, 4).cuda()
        img_embeddings = self._image_branch(img_seq)
        return img_embeddings

    def forward(self, inputs):
        if self.config.use_position:
            imu_obs = inputs['imu']
        else:
            imu_obs = inputs['imu'][:, :, 3:]
        if (not self.config.use_attitude):
            if self.config.use_position:
                print("ERROR: Do not use position without attitude!")
                return
            else:
                imu_obs = inputs['imu'][:, :, 12:]
        
        imu_obs = torch.tensor(imu_obs).cuda()
        imu_embeddings = self._imu_branch(imu_obs)
        img_embeddings = self._preprocess_frames(inputs)
        if img_embeddings is not None:
            total_embeddings = torch.cat((img_embeddings, imu_embeddings), dim=-1)  # [B, modes, MxM + 64]
        else:
            total_embeddings = imu_embeddings
        output = self._plan_branch(total_embeddings)
        # print(total_embeddings.shape)
        # print(output.shape)
        # print("--")
        output = output.permute(0, 2, 1)
        # print(output.shape)
        return output