import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from models import GradReverse
from utils import show_decoded_images
import torch.nn.functional as F

criterion_weighted_classifier = nn.NLLLoss(reduction="none")
criterion_disentangle = nn.MSELoss()
criterion_distance = nn.MSELoss()
criterion_reconstruction = nn.BCELoss()
criterion_classifier = nn.NLLLoss(reduction="mean")
criterion_triplet = nn.TripletMarginLoss(margin=1)


def feature_loss(z, target):
    return F.mse_loss(torch.cos(z), torch.cos(target))


def train_domain_adaptation(
    model,
    optimizer,
    source_train_loader,
    target_train_loader,
    betas,
    delta,
    alpha=1,
    gamma=1,
    epochs=30,
    scheduler=None,
    log=False,
    show_images=False,
):
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    t = tqdm(range(epochs))
    for epoch in t:
        total_loss = 0
        corrects_source, corrects_target = 0, 0
        total_source, total_target = 0, 0

        for (x_s, y_s), (x_t, y_t) in zip(source_train_loader, target_train_loader):
            loss = 0
            min_len = min(len(x_s), len(x_t))
            x_s = x_s[:min_len].cuda()
            y_s = y_s[:min_len].cuda()
            x_t = x_t[:min_len].cuda()
            y_t = y_t[:min_len].cuda()

            # ===================target batch======================================
            xt_hat, yt_hat, (z_task, z_target), (pred_task, pred_spe) = model(
                x_t, mode="all_target"
            )

            rev_task = GradReverse.grad_reverse(z_task)
            rev_spe = GradReverse.grad_reverse(z_target)

            # synthetic sample with task information from x_t and style info from xs_rand
            xts = model.decode(z_task, x_s, mode="source")
            z_s = model.encoder(xts.detach(), mode="task")
            z_target_prime = model.encoder(torch.flip(x_t, [0]), mode="target")
            xt_prime = model.decoder_target(z_task, z_target_prime)
            yt_tilde, z_target_tilde = model.forward(xt_prime, mode="target")

            w, predicted = yt_hat.max(1)
            corrects_target += predicted.eq(y_t).sum().item()
            total_target += y_t.size(0)

            loss += alpha * criterion_reconstruction(xt_hat, x_t)
            loss += betas[epoch] * criterion_distance(z_task, z_s)
            loss += gamma * (
                feature_loss(pred_task, rev_task) + feature_loss(pred_spe, rev_spe)
            )
            loss += 0.1 * torch.mean(
                (
                    w.detach().exp()
                    * criterion_weighted_classifier(yt_tilde, predicted.detach())
                )
            )
            loss += delta * criterion_triplet(z_target_tilde, z_target_prime, z_target)

            # source batch
            xs_hat, ys_hat, (z_task, z_source), (pred_task, pred_spe) = model(
                x_s, mode="all_source"
            )
            rev_task = GradReverse.grad_reverse(z_task)
            rev_spe = GradReverse.grad_reverse(z_source)

            # synthetic sample with task information from x_s and style info from xt_rand
            xst = model.decode(z_task, x_t, mode="target")
            z_t = model.encoder(xst.detach(), mode="task")
            z_source_prime = model.encoder(torch.flip(x_s, [0]), mode="source")
            xs_prime = model.decoder_source(z_task, z_source_prime)
            ys_tilde, z_source_tilde = model(xs_prime, mode="source")

            _, predicted = ys_hat.max(1)
            corrects_source += predicted.eq(y_s).sum().item()
            total_source += y_s.size(0)

            loss += criterion_classifier(ys_hat, y_s)
            loss += alpha * criterion_reconstruction(xs_hat, x_s)
            loss += betas[epoch] * criterion_distance(z_task, z_t)
            loss += gamma * (
                feature_loss(pred_task, rev_task) + feature_loss(pred_spe, rev_spe)
            )
            loss += 0.1 * criterion_classifier(ys_tilde, y_s.cuda())
            loss += delta * criterion_triplet(z_source_tilde, z_source_prime, z_source)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += float(loss.data)
            t.set_description(
                f"epoch:{epoch} current target accuracy:{round(corrects_target / total_source * 100, 2)}%"
            )
            # ===================log========================
            if log:
                print(
                    "epoch [{}/{}], loss:{:.4f}".format(
                        epoch + 1, epochs, total_loss / len(source_train_loader)
                    )
                )
                print(
                    f"accuracy source: {round(corrects_source / total_source * 100, 2)}%"
                )
                print(
                    f"accuracy target: {round(corrects_target / total_target * 100, 2)}%"
                )
                if show_images:
                    show_decoded_images(
                        x_s[:16], xs_hat[:16], x_t[: len(x_s)][:16], xst[:16]
                    )
                    show_decoded_images(
                        x_t[:16], xt_hat[:16], x_s[: len(x_t)][:16], xts[:16]
                    )
        if scheduler != None:
            scheduler.step()
    return corrects_target / total_source


def train_disentangle(
    model,
    optimizer,
    source_train_loader,
    epochs=30,
    beta_max=1,
    running_beta=10,
    show_images=False,
):
    betas = np.zeros(epochs)
    betas[:running_beta] = np.linspace(0.01, beta_max, running_beta)
    betas[running_beta:] = np.ones(epochs - running_beta) * beta_max

    t = tqdm(range(epochs))
    for epoch in t:
        corrects, total = 0, 0
        # random images used for disentanglement
        for x, y in source_train_loader:
            loss = 0
            x = x.cuda()
            y = y.cuda()
            x_hat, y_hat, (z_task, z_style), (pred_task, pred_style) = model(
                x, mode="all"
            )
            rev_task = z_task
            rev_style = z_style
            z_style_r = model.encoder(torch.flip(x, [0]), mode="style")
            x_prime = model.decoder(z_task, z_style_r)
            y_tilde, z_style_tilde = model(x_prime, mode="style")

            loss += 10 * criterion_reconstruction(x_hat, x)
            loss += criterion_classifier(y_hat, y)
            loss += betas[epoch] * (
                feature_loss(pred_task, rev_task) + feature_loss(pred_style, rev_style)
            )
            loss += 0.1 * criterion_classifier(y_tilde, y)
            loss += 1 * criterion_triplet(z_style_tilde, z_style_r, z_style)

            _, predicted = y_hat.max(1)
            corrects += predicted.eq(y).sum().item()
            total += y.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(
                f"epoch:{epoch} current accuracy:{round(corrects / total * 100, 2)}%"
            )
        # ===================log========================
        if show_images:
            show_decoded_images(
                x[:32], x_hat[:32], torch.flip(x, [0])[:32], x_prime[:32]
            )
