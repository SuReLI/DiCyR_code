from torch import nn
import torch

def get_simple_classifier(latent_space_dim=1024):
    return nn.Sequential(nn.Dropout2d(),
                         nn.Linear(in_features=latent_space_dim, out_features=10),
                         nn.LogSoftmax(dim=1))


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

    
class ProjectorNetwork(nn.Module):
    def __init__(self, latent_dim):
        super(ProjectorNetwork, self).__init__()
        self.net=  nn.Sequential(
                        nn.Linear(latent_dim, 100),
                        nn.ReLU(True),
                        nn.Linear(100, 100),
                        nn.ReLU(True),
                        nn.Linear(100, 1))
        
    def forward(self, x):
        return torch.cos(self.net(x))

class DomainAdaptationNetwork(nn.Module):
    def __init__(self, encoder, decoder_source, decoder_target, classifier):
        super(DomainAdaptationNetwork, self).__init__()
        self.encoder = encoder
        self.decoder_source = decoder_source
        self.decoder_target = decoder_target
        self.classifier = classifier
        latent_space_dim = self.encoder.latent_space_dim

        self.random_source = ProjectorNetwork(latent_space_dim)
        self.random_target = ProjectorNetwork(latent_space_dim)
        self.random_task = ProjectorNetwork(latent_space_dim)
        self.spe_predictor = ProjectorNetwork(latent_space_dim)
        self.task_predictor = ProjectorNetwork(latent_space_dim)

       
    def forward(self, x, mode=None):
        if mode is None:
            z_task = self.encoder(x, mode='task')
            logits = self.classifier(z_task)
            return logits
        
        z_task, z_source, z_target = self.encoder(x)
        logits = self.classifier(z_task)
        
        if mode == 'source':
            return logits, z_source
        
        elif mode == 'target':
            return logits, z_target

        elif mode == 'all_target':
            xt_hat = self.decoder_target(z_task, z_target)
            z_spe = z_source
            
        elif mode == 'all_source':
            xt_hat = self.decoder_source(z_task, z_source)
            z_spe = z_target
            
        pred_spe = self.spe_predictor(GradReverse.grad_reverse(z_task))
        pred_task = self.task_predictor(GradReverse.grad_reverse(z_spe))
        return xt_hat, logits, (z_task, z_target), (pred_task, pred_spe)

    def decode(self, z_task, x_rand, mode='source'):
        if mode ==  'source':
            z_rand = self.encoder(x_rand, mode='source')
            reco = self.decoder_source(z_task, z_rand)
        elif mode ==  'target':
            z_rand = self.encoder(x_rand, mode='target')
            reco = self.decoder_target(z_task, z_rand)
            
        return reco


class DisenrangledNetwork(nn.Module):
    def __init__(self, encoder, decoder, classifier):
        super(DisenrangledNetwork, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        latent_space_dim = self.encoder.latent_space_dim

        self.style_predictor = ProjectorNetwork(latent_space_dim)
        self.task_predictor = ProjectorNetwork(latent_space_dim)
        
    def forward(self, x, mode='all'):
        z_task, z_style = self.encoder(x)
        logits = self.classifier(z_task)
        
        if mode == 'all':
            x_hat = self.decoder(z_task, z_style)
            pred_style = self.style_predictor(GradReverse.grad_reverse(z_task))
            pred_task = self.task_predictor(GradReverse.grad_reverse(z_style))

            return x_hat, logits, (z_task, z_style), (pred_task, pred_style)
        
        elif mode == 'style':
            return logits, z_style
        
        else:
            return logits
