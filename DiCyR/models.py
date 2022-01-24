import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np


# def get_simple_classifier(latent_space_dim=1024):
#     return nn.Sequential(
#         nn.Dropout2d(),
#         nn.Linear(in_features=latent_space_dim, out_features=10),
#         nn.LogSoftmax(dim=1),
#     )


class GradReverse(torch.autograd.Function):
    """Extension of grad reverse layer."""

    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.neg()
        return grad_output, None

    def grad_reverse(x):
        return GradReverse.apply(x)


# class ProjectorNetwork(nn.Module):
#     def __init__(self, latent_dim):
#         super(ProjectorNetwork, self).__init__()
#         self.net = nn.Sequential(
#             nn.Linear(latent_dim, 32), nn.ReLU(), nn.Linear(32, latent_dim)
#         )

#     def forward(self, x):
#         return self.net(x)


class FeaturesPredictorNetwork(nn.Module):
    def __init__(self, latent_dim):
        super(FeaturesPredictorNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 75),
            nn.ReLU(),
            nn.Linear(75, 75),
            nn.ReLU(),
            nn.Linear(75, latent_dim),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


class DomainAdaptationNetwork(nn.Module):
    def __init__(self, encoder, decoder_source, decoder_target, nb_classes=10):
        super(DomainAdaptationNetwork, self).__init__()
        self.encoder = encoder
        self.decoder_source = decoder_source
        self.decoder_target = decoder_target
        latent_space_dim = self.encoder.latent_space_dim
        self.classifier = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(in_features=latent_space_dim, out_features=nb_classes),
            nn.LogSoftmax(dim=1),
        )

        self.spe_predictor = FeaturesPredictorNetwork(latent_space_dim)
        self.task_predictor = FeaturesPredictorNetwork(latent_space_dim)

    def forward(self, x, mode=None):
        if mode is None:
            z_task = self.encoder(x, mode="task")
            logits = self.classifier(z_task)
            return logits

        z_task, z_source, z_target = self.encoder(x)
        logits = self.classifier(z_task)

        if mode == "source":
            return logits, z_source

        elif mode == "target":
            return logits, z_target

        elif mode == "all_target":
            xt_hat = self.decoder_target(z_task, z_target)
            z_spe = z_source

        elif mode == "all_source":
            xt_hat = self.decoder_source(z_task, z_source)
            z_spe = z_target

        pred_spe = self.spe_predictor(GradReverse.grad_reverse(z_task))
        pred_task = self.task_predictor(GradReverse.grad_reverse(z_spe))
        return xt_hat, logits, (z_task, z_spe), (pred_task, pred_spe)

    def decode(self, z_task, x_rand, mode="source"):
        if mode == "source":
            z_rand = self.encoder(x_rand, mode="source")
            reco = self.decoder_source(z_task, z_rand)
        elif mode == "target":
            z_rand = self.encoder(x_rand, mode="target")
            reco = self.decoder_target(z_task, z_rand)

        return reco


class DisentangledNetwork(nn.Module):
    def __init__(self, encoder, decoder, nb_classes=10):
        super(DisentangledNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        latent_space_dim = self.encoder.latent_space_dim
        self.classifier = nn.Sequential(
            nn.Dropout2d(),
            nn.Linear(in_features=latent_space_dim, out_features=nb_classes),
            nn.LogSoftmax(dim=1),
        )

        self.style_predictor = FeaturesPredictorNetwork(latent_space_dim)
        self.task_predictor = FeaturesPredictorNetwork(latent_space_dim)

    def forward(self, x, mode="all"):
        z_task, z_style = self.encoder(x)
        logits = self.classifier(z_task)

        if mode == "all":
            x_hat = self.decoder(z_task, z_style)
            pred_style = self.style_predictor(GradReverse.grad_reverse(z_task))
            pred_task = self.task_predictor(GradReverse.grad_reverse(z_style))

            return x_hat, logits, (z_task, z_style), (pred_task, pred_style)

        elif mode == "style":
            return logits, z_style

        else:
            return logits


class Decoder(nn.Module):
    def __init__(self, latent_space_dim, conv_feat_size, nb_channels=3):
        super(Decoder, self).__init__()

        self.latent_space_dim = latent_space_dim
        self.conv_feat_size = conv_feat_size

        self.deco_dense = nn.Sequential(
            nn.Linear(in_features=latent_space_dim, out_features=1024),
            nn.ReLU(True),
            nn.Linear(in_features=1024, out_features=np.prod(self.conv_feat_size)),
            nn.ReLU(True),
        )

        self.deco_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=self.conv_feat_size[0],
                out_channels=75,
                kernel_size=3,
                padding=1,
            ),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=75, out_channels=50, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(
                in_channels=50, out_channels=nb_channels, kernel_size=5, padding=2
            ),
            nn.Sigmoid(),
        )

    def forward(self, z_task, z_spe):
        z = torch.cat([z_task, z_spe], 1)
        feat_encode = self.deco_dense(z)
        feat_encode = feat_encode.view(-1, *self.conv_feat_size)
        y = self.deco_conv(feat_encode)

        return y


class DisentanglementEncoder(nn.Module):
    def __init__(self, latent_space_dim, nb_channels=3):
        super(DisentanglementEncoder, self).__init__()

        self.latent_space_dim = latent_space_dim
        self.nb_channels = nb_channels

        self.embedder = nn.Sequential(
            nn.Conv2d(nb_channels, out_channels=32, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding=2),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.BatchNorm2d(128),
            nn.Flatten(),
            nn.Linear(in_features=8192, out_features=1024),
            nn.ReLU(True),
        )

        self.share_feat = nn.Linear(in_features=1024, out_features=latent_space_dim)
        self.style_feat = nn.Linear(in_features=1024, out_features=latent_space_dim)

    def forward(self, input_data, mode="all"):
        if (input_data.shape[1] == 1) & (self.nb_channels == 3):
            input_data = input_data.repeat(1, 3, 1, 1)
        feat = self.embedder(input_data)

        if mode == "all":
            z_share = F.relu(self.share_feat(feat))
            z_style = F.relu(self.style_feat(feat))
            return z_share, z_style

        elif mode == "style":
            z_style = F.relu(self.style_feat(feat))
            return z_style


class DomainAdaptationEncoder(nn.Module):
    def __init__(self, embedder, latent_space_dim, nb_channels=3):
        super(DomainAdaptationEncoder, self).__init__()

        self.latent_space_dim = latent_space_dim
        self.nb_channels = nb_channels
        self.embedder = embedder
        self.task_feat = nn.Linear(in_features=512, out_features=latent_space_dim)
        self.source_feat = nn.Linear(in_features=512, out_features=latent_space_dim)
        self.target_feat = nn.Linear(in_features=512, out_features=latent_space_dim)

    def forward(self, input_data, mode="all"):
        if (input_data.shape[1] == 1) & (self.nb_channels == 3):
            input_data = input_data.repeat(1, 3, 1, 1)
        feat = self.embedder(input_data)
        if mode == "task":
            z_task = F.relu(self.task_feat(feat))
            return z_task

        elif mode == "source":
            z_source = F.relu(self.source_feat(feat))
            return z_source

        elif mode == "target":
            z_target = F.relu(self.target_feat(feat))
            return z_target

        else:
            z_task = F.relu(self.task_feat(feat))
            z_source = F.relu(self.source_feat(feat))
            z_target = F.relu(self.target_feat(feat))
            return z_task, z_source, z_target
