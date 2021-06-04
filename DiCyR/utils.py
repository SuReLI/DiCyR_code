import torch
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from urllib import request
import json
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import numpy as np


def show_images_grid(images):
    images = make_grid(images)
    plt.imshow(images.permute(1, 2, 0), cmap="gray")
    plt.axis('off')
    plt.tight_layout()

    
def show_decoded_images(x, y, x_permuted, x_eq):
    plt.figure(figsize=(20,10))
    plt.subplot(1,4,1)
    show_images_grid(x.detach().cpu())
    plt.title('original')
    plt.subplot(1,4,2)
    show_images_grid(y.detach().cpu())
    plt.title('reconstruction')
    plt.subplot(1,4,3)
    show_images_grid(x_permuted.detach().cpu())
    plt.title('style')
    plt.subplot(1,4,4)
    show_images_grid(x_eq.detach().cpu())
    plt.title('swapped style')
    plt.show()


def test_network(model, dataloader):
    test_corrects = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            img, labels = data
            img = img.cuda()
            targets = labels.cuda()
            pred = model(img)
            _, predicted = pred.max(1)
            test_corrects += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return test_corrects / total


def plot_target_cross_domain_swapping(model, source_train_loader, target_train_loader):
    X, _ = next(iter(target_train_loader))
    y, _, (z_share, z_spe),  _ = model(X.cuda(), mode='all_target')
    X2, _ = next(iter(source_train_loader))
    _, _, (z_share, _),  _ = model(X2.cuda(), mode='all_target')
    #blank
    plt.subplot(1,6,1)
    plt.imshow(torch.ones((32,32,3)))
    plt.axis('off')
    plt.tight_layout()
    #styles
    for i in range(5):
        plt.subplot(1,6,i+2)
        plt.imshow(X[i].cpu().detach()[0], cmap='gray')
        plt.axis('off')
        plt.tight_layout()

    for j in range(10, 20):
        plt.figure()
        plt.subplot(1,6,1)
        plt.imshow(X2[j].detach()[0], cmap='gray')
        plt.axis('off')
        plt.tight_layout()

        z_x = torch.zeros_like(z_share)
        z_x[:] = z_share[j]
        y2  = model.decoder_target(z_x, z_spe)
        for i in range(5):
            plt.subplot(1,6,i+2)
            plt.imshow(y2[i].cpu().detach()[0], cmap='gray')
            plt.axis('off')
            plt.tight_layout()


def extract_features(encoder, train_loader, sample_count, batch_size=128, feature_size=150):
    features = np.zeros(shape=(sample_count, feature_size))
    labels = np.zeros(shape=(sample_count))
    i = 0
    for x, labels_batch in train_loader:
        features_batch = encoder(x.cuda(), mode='task')[0]
        features[i * batch_size: (i + 1) * batch_size] = features_batch.cpu().detach().numpy()
        labels[i * batch_size: (i + 1) * batch_size] = labels_batch.numpy()
        i += 1

        if i * batch_size >= sample_count:
            break
    return features, labels.astype(int)


def plot_tsne(model, source_train_loader, target_train_loader, batch_size=128, feature_size=150):
    f_s, s_labels = extract_features(model.encoder, source_train_loader, 640, batch_size, feature_size)
    f_t, t_labels = extract_features(model.encoder, target_train_loader, 640, batch_size, feature_size)

    f = np.zeros((1280, feature_size))
    f[:640] = f_s
    f[640:] = f_t
    pca = PCA(n_components=50)
    X_pca = pca.fit_transform(f)
    X_tsne = TSNE(2, perplexity=50).fit_transform(X_pca)
    X_tsne_x = X_tsne[:, 0]
    X_tsne_y = X_tsne[:, 1]
    plt.figure(figsize=(16, 10))
    color_map = plt.cm.rainbow(np.linspace(0, 1, 10))
    for i, g in enumerate(np.unique(s_labels)):
        color = color_map[i]
        ix = np.where(s_labels == g)
        plt.scatter(x=X_tsne_x[ix], y=X_tsne_y[ix], color=color, marker='o', alpha=0.5, label=g)
    for i, g in enumerate(np.unique(t_labels)):
        color = color_map[i]
        ix = np.where(t_labels == g)
        plt.scatter(x=X_tsne_x[640:][ix], y=X_tsne_y[640:][ix], color=color, marker='+', label=g)
        plt.legend()
        plt.axis('off')


def plot_swapped_styles(model, train_loader):
    X, _ = next(iter(train_loader))
    y, _, (z_task, z_style), _ = model(X.cuda(), mode='all')
    # blank
    plt.subplot(1, 6, 1)
    plt.imshow(torch.ones((32, 32, 3)))
    plt.axis('off')
    plt.tight_layout()
    # styles
    for i in range(5):
        plt.subplot(1, 6, i + 2)
        plt.imshow(X[i].cpu().detach().permute(1, 2, 0))
        plt.axis('off')
        plt.tight_layout()

    for j in range(10, 20):
        plt.figure()
        plt.subplot(1, 6, 1)
        plt.imshow(X[j].permute(1, 2, 0))
        plt.axis('off')
        plt.tight_layout()

        z_x = torch.zeros_like(z_task)
        z_x[:] = z_task[j]
        y2 = model.decoder(z_x, z_style)
        for i in range(5):
            plt.subplot(1, 6, i + 2)
            plt.imshow(y2[i].cpu().detach().permute(1, 2, 0))
            plt.axis('off')
            plt.tight_layout()