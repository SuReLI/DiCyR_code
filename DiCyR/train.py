import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from utils import show_decoded_images


def train_domain_adaptation(model, optimizer, source_train_loader, target_train_loader, betas,
                            alpha=1, gamma=1, delta=0.1, epochs=30, show_images=False):

    criterion_reconstruction = nn.BCELoss()
    criterion_classifier = nn.NLLLoss(reduction='mean')
    criterion_weighted_classifier = nn.NLLLoss(reduction='none')
    criterion_disentangle = nn.MSELoss()
    criterion_distance = nn.MSELoss()
    criterion_triplet = nn.TripletMarginLoss(margin=1)

    t = tqdm(range(epochs))
    for epoch in t:
        total_loss = 0
        corrects_source = 0
        corrects_target = 0
        total_source = 0
        total_target = 0

        # random images used for disentanglement
        xs_rand = next(iter(source_train_loader))[0].cuda()
        xt_rand = next(iter(target_train_loader))[0].cuda()

        for (x_s, y_s), (x_t, y_t) in zip(source_train_loader, target_train_loader):
            loss = 0
            x_s, y_s, x_t, y_t = x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda()

            # target batch
            xt_hat, yt_hat, (z_task, z_target), (random_share, random_spe), (pred_share, pred_spe) = model.forward_t(
                x_t)
            xts = model.forward_s_rand(z_task, xs_rand[:len(x_t)])
            z_s = model.encoder.forward_share(xts.detach())
            z_target_prime = model.encoder.forward_target(xt_rand[:len(x_t)])
            xt_prime = model.decoder_target(z_task, z_target_prime)
            yt_tilde, z_target_tilde = model.forward_target(xt_prime)

            w, predicted = yt_hat.max(1)
            corrects_target += predicted.eq(y_t).sum().item()
            total_target += y_t.size(0)

            loss += alpha * criterion_reconstruction(xt_hat, x_t)
            loss += betas[epoch] * criterion_distance(z_task, z_s)
            loss += gamma * (criterion_disentangle(pred_share, random_share) + criterion_disentangle(pred_spe, random_spe))
            loss += delta * torch.mean((torch.exp(w.detach()) * criterion_weighted_classifier(yt_tilde, predicted.detach())))
            loss += delta * criterion_triplet(z_target_tilde, z_target_prime, z_target)

            # source batch
            xs_hat, ys_hat, (z_task, z_source), (random_share, random_spe), (pred_share, pred_spe) = model.forward_s(
                x_s)
            xst = model.forward_t_rand(z_task, xt_rand[:len(x_s)])
            z_t = model.encoder.forward_share(xst.detach())
            z_source_prime = model.encoder.forward_source(xs_rand[:len(x_s)])
            xs_prime = model.decoder_source(z_task, z_source_prime)
            ys_tilde, z_source_tilde = model.forward_source(xs_prime)

            _, predicted = ys_hat.max(1)
            corrects_source += predicted.eq(y_s).sum().item()
            total_source += y_s.size(0)

            loss += criterion_classifier(ys_hat, y_s)
            loss += alpha * criterion_reconstruction(xs_hat, x_s)
            loss += betas[epoch] * criterion_distance(z_task, z_t)
            loss += gamma * (criterion_disentangle(pred_share, random_share) + criterion_disentangle(pred_spe, random_spe))
            loss += delta * criterion_classifier(ys_tilde, y_s.cuda())
            loss += delta * criterion_triplet(z_source_tilde, z_source_prime, z_source)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.data)
            t.set_description(f'epoch:{epoch} current target accuracy:{round(corrects_target / total_source * 100, 2)}%')
        # ===================log========================
        print('epoch [{}/{}], loss:{:.4f}'.format(epoch + 1, epochs, total_loss / len(source_train_loader)))
        print(f'accuracy source: {round(corrects_source / total_source * 100, 2)}%')
        print(f'accuracy target: {round(corrects_target / total_target * 100, 2)}%')
        if show_images:
            show_decoded_images(x_s, xs_hat, xt_rand[:len(x_s)], xst)
            show_decoded_images(x_t, xt_hat, xs_rand[:len(x_t)], xts)


def train_disentangle(model, optimizer, source_train_loader, epochs=30, beta_max=10, running_beta=20,
                      alpha_max=5, alpha_min=5, show_images=False):
    betas = np.zeros(epochs)
    betas[:running_beta] = np.linspace(0.01, beta_max, running_beta)
    betas[running_beta:] = np.ones(epochs - running_beta) * beta_max

    criterion_reconstruction = nn.BCELoss()
    disentangle_criterion = nn.MSELoss()
    criterion_classifier = nn.NLLLoss(reduction='mean')
    criterion_triplet = nn.TripletMarginLoss(margin=1)

    alpha = np.linspace(alpha_min, alpha_max, epochs)

    for epoch in tqdm(range(epochs)):

        # random images used for disentanglement
        x_rand = next(iter(source_train_loader))[0].cuda()
        for x, y in source_train_loader:
            loss = 0
            x = x.cuda()
            y = y.cuda()
            x_hat, y_hat, (z_task, z_style), (random_task, random_style), (pred_task, pred_style) = model.forward(x)
            z_style = model.encoder.forward_style(x_rand[:len(x)])
            x_prime = model.decoder(z_task, z_style)
            y_tilde, z_style_tilde = model.forward_style(x_prime)

            loss += alpha[epoch] * criterion_reconstruction(x_hat, x)
            loss += criterion_classifier(y_hat, y)
            loss += betas[epoch] * (
                    disentangle_criterion(pred_task, random_task) + disentangle_criterion(pred_style, random_style))
            loss += 0.1 * criterion_classifier(y_tilde, y)
            loss += 0.1 * criterion_triplet(z_style_tilde, z_style, z_style)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # ===================log========================
        if show_images:
            show_decoded_images(x[:32], x_hat[:32], x_prime[:32], x_rand[:32])
