import torch
import numpy as np
from torch import nn
from tqdm import tqdm
from utils import show_decoded_images


criterion_reconstruction = nn.BCELoss()
criterion_classifier = nn.NLLLoss(reduction='mean')
criterion_weighted_classifier = nn.NLLLoss(reduction='none')
criterion_disentangle = nn.MSELoss()
criterion_distance = nn.MSELoss()
criterion_triplet = nn.TripletMarginLoss(margin=1)

criterion_reconstruction = nn.BCELoss()
disentangle_criterion = nn.MSELoss()
criterion_classifier = nn.NLLLoss(reduction='mean')
criterion_triplet = nn.TripletMarginLoss(margin=1)

def train_domain_adaptation(model, optimizer, random_projector, source_train_loader, target_train_loader, betas,
                            alpha=1, gamma=1, delta=0.1, epochs=30, show_images=False):

    t = tqdm(range(epochs))
    for epoch in t:
        total_loss = 0
        corrects_source, corrects_target = 0, 0
        total_source, total_target = 0, 0

        # random images used for disentanglement
        xs_rand = next(iter(source_train_loader))[0].cuda()
        xt_rand = next(iter(target_train_loader))[0].cuda()

        for (x_s, y_s), (x_t, y_t) in zip(source_train_loader, target_train_loader):
            loss = 0
            x_s, y_s, x_t, y_t = x_s.cuda(), y_s.cuda(), x_t.cuda(), y_t.cuda()

            # target batch
            xt_hat, yt_hat, (z_task, z_target), (pred_task, pred_spe) = model(x_t, mode='all_target')
            #Random projection to reduce the dimension
            random_task = random_projector(z_task)
            random_spe = random_projector(z_target)
            
            # synthetic sample with task information from x_t and style info from xs_rand
            xts = model.decode(z_task, xs_rand[:len(x_t)], mode='source')
            z_s = model.encoder(xts.detach(), mode='task')
            z_target_prime = model.encoder(xt_rand[:len(x_t)], mode='target')
            xt_prime = model.decoder_target(z_task, z_target_prime)
            yt_tilde, z_target_tilde = model.forward(xt_prime, mode='target')

            w, predicted = yt_hat.max(1)
            corrects_target += predicted.eq(y_t).sum().item()
            total_target += y_t.size(0)

            loss += alpha * criterion_reconstruction(xt_hat, x_t)
            loss += betas[epoch] * criterion_distance(z_task, z_s)
            loss += gamma * (criterion_disentangle(pred_task, random_task) + criterion_disentangle(pred_spe, random_spe))
            loss += delta * torch.mean((torch.exp(w.detach()) * criterion_weighted_classifier(yt_tilde, predicted.detach())))
            loss += delta * criterion_triplet(z_target_tilde, z_target_prime, z_target)

            # source batch
            xs_hat, ys_hat, (z_task, z_source), (pred_task, pred_spe) = model(x_s, mode='all_source')
            #Random projection to reduce the dimension
            random_task = random_projector(z_task)
            random_spe = random_projector(z_source)
            
            # synthetic sample with task information from x_s and style info from xt_rand
            xst = model.decode(z_task, xt_rand[:len(x_s)], mode='target')
            z_t = model.encoder(xst.detach(), mode='task')
            z_source_prime = model.encoder(xs_rand[:len(x_s)], mode='source')
            xs_prime = model.decoder_source(z_task, z_source_prime)
            ys_tilde, z_source_tilde = model(xs_prime, mode='source')

            _, predicted = ys_hat.max(1)
            corrects_source += predicted.eq(y_s).sum().item()
            total_source += y_s.size(0)

            loss += criterion_classifier(ys_hat, y_s)
            loss += alpha * criterion_reconstruction(xs_hat, x_s)
            loss += betas[epoch] * criterion_distance(z_task, z_t)
            loss += gamma * (criterion_disentangle(pred_task, random_task) + criterion_disentangle(pred_spe, random_spe))
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


def train_disentangle(model, optimizer, random_projector, source_train_loader,
                      epochs=30, beta_max=10, running_beta=20, alpha_max=5, alpha_min=5, show_images=False):
    betas = np.zeros(epochs)
    betas[:running_beta] = np.linspace(0.01, beta_max, running_beta)
    betas[running_beta:] = np.ones(epochs - running_beta) * beta_max
    alpha = np.linspace(alpha_min, alpha_max, epochs)

    t = tqdm(range(epochs))
    for epoch in t:
        corrects, total = 0, 0
        # random images used for disentanglement
        x_rand = next(iter(source_train_loader))[0].cuda()
        for x, y in source_train_loader:
            loss = 0
            x = x.cuda()
            y = y.cuda()
            x_hat, y_hat, (z_task, z_style), (pred_task, pred_style) = model(x, mode='all')
            # random projections to reduce dimension
            random_task = random_projector(z_task)
            random_style = random_projector(z_style)
            z_style_r = model.encoder(x_rand[:len(x)], mode='style')
            x_prime = model.decoder(z_task, z_style_r)
            y_tilde, z_style_tilde = model(x_prime, mode='style')

            loss += alpha[epoch] * criterion_reconstruction(x_hat, x)
            loss += criterion_classifier(y_hat, y)
            loss += betas[epoch] * (disentangle_criterion(pred_task, random_task) + disentangle_criterion(pred_style, random_style))
            loss += 0.1 * criterion_classifier(y_tilde, y)
            loss += 0.1 * criterion_triplet(z_style_tilde, z_style_r, z_style)
            
            _, predicted = y_hat.max(1)
            corrects += predicted.eq(y).sum().item()
            total += y.size(0)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_description(f'epoch:{epoch} current accuracy:{round(corrects / total * 100, 2)}%')
        # ===================log========================
        if show_images:
            show_decoded_images(x[:32], x_hat[:32], x_rand[:32], x_prime[:32])