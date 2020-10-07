from torch import nn
import torch

def get_simple_classifier(latent_space_dim=1024):
    return nn.Sequential(nn.Dropout2d(),
                         nn.Linear(in_features=latent_space_dim, out_features=10),
                         nn.LogSoftmax())


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


class DomainAdaptationNetwork(nn.Module):
    def __init__(self, encoder, decoder_source, decoder_target, classifier):
        super(DomainAdaptationNetwork, self).__init__()
        self.encoder = encoder
        self.decoder_source = decoder_source
        self.decoder_target = decoder_target
        self.classifier = classifier
        latent_space_dim = self.encoder.latent_space_dim

        self.random_source = nn.Sequential(
            nn.Linear(latent_space_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 1))

        self.random_target = nn.Sequential(
            nn.Linear(latent_space_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 1))

        self.random_share = nn.Sequential(
            nn.Linear(latent_space_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 1))

        self.spe_predictor = nn.Sequential(
            nn.Linear(latent_space_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 1))

        self.share_predictor = nn.Sequential(
            nn.Linear(latent_space_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 1))

    def forward(self, x):
        z_share = self.encoder.forward_share(x)
        logits = self.classifier(z_share)
        return logits

    def forward_s(self, x):
        z_share, z_source, _ = self.encoder(x)
        reco_s = self.decoder_source(z_share, z_source)
        logits = self.classifier(z_share)

        z_spe_rev = GradReverse.grad_reverse(z_source)
        z_share_rev = GradReverse.grad_reverse(z_share)
        pred_spe = torch.cos(self.spe_predictor(z_share_rev))
        pred_share = torch.cos(self.share_predictor(z_spe_rev))
        random_spe = torch.cos(self.random_source(z_source))
        random_share = torch.cos(self.random_share(z_share))

        return reco_s, logits, (z_share, z_source), (random_share, random_spe), (pred_share, pred_spe)

    def forward_s_rand(self, z_share, x_rand):
        z_source = self.encoder.forward_source(x_rand)
        reco_s_rand = self.decoder_source(z_share, z_source)
        return reco_s_rand

    def forward_t(self, x):
        z_share, _, z_target = self.encoder(x)
        reco_t = self.decoder_target(z_share, z_target)
        logits = self.classifier(z_share)

        z_spe_rev = GradReverse.grad_reverse(z_target)
        z_share_rev = GradReverse.grad_reverse(z_share)
        pred_spe = torch.cos(self.spe_predictor(z_share_rev))
        pred_share = torch.cos(self.share_predictor(z_spe_rev))
        random_spe = torch.cos(self.random_target(z_target))
        random_share = torch.cos(self.random_share(z_share))

        return reco_t, logits, (z_share, z_target), (random_share, random_spe), (pred_share, pred_spe)

    def forward_t_rand(self, z_share, x_rand):
        z_target = self.encoder.forward_target(x_rand)
        reco_t_rand = self.decoder_target(z_share, z_target)
        return reco_t_rand

    def forward_st(self, z_source, x):
        z_share = self.encoder.forward_share(x)
        reco_st = self.decoder_source(z_share, z_source)
        logits = self.classifier(z_share)
        return reco_st, logits, z_share

    def forward_ts(self, z_target, x):
        z_share = self.encoder.forward_share(x)
        reco_ts = self.decoder_target(z_share, z_target)
        logits = self.classifier(z_share)
        return reco_ts, logits, z_share

    def forward_target(self, x):
        z_share, _, z_target = self.encoder(x)
        logits = self.classifier(z_share)
        return logits, z_target

    def forward_source(self, x):
        z_share, z_source, _ = self.encoder(x)
        logits = self.classifier(z_share)

        return logits, z_source


class DisenrangledNetwork(nn.Module):
    def __init__(self, encoder, decoder, classifier):
        super(DisenrangledNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        latent_space_dim = self.encoder.latent_space_dim

        self.random_style = nn.Sequential(
            nn.Linear(latent_space_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 1))

        self.random_task = nn.Sequential(
            nn.Linear(latent_space_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 1))

        self.style_predictor = nn.Sequential(
            nn.Linear(latent_space_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 1))

        self.task_predictor = nn.Sequential(
            nn.Linear(latent_space_dim, 100),
            nn.ReLU(True),
            nn.Linear(100, 100),
            nn.ReLU(True),
            nn.Linear(100, 1))

    def forward(self, x):
        z_task, z_style = self.encoder(x)
        x_hat = self.decoder(z_task, z_style)
        logits = self.classifier(z_task)

        z_style_rev = GradReverse.grad_reverse(z_style)
        z_task_rev = GradReverse.grad_reverse(z_task)
        pred_style = torch.cos(self.style_predictor(z_task_rev))
        pred_task = torch.cos(self.task_predictor(z_style_rev))
        random_style = torch.cos(self.random_style(z_style))
        random_task = torch.cos(self.random_task(z_task))

        return x_hat, logits, (z_task, z_style), (random_task, random_style), (pred_task, pred_style)

    def forward_style(self, x):
        z_task, z_style = self.encoder(x)
        logits = self.classifier(z_task)

        return logits, z_style