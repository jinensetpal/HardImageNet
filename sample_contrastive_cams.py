#!/usr/bin/env python3

from src.data.hard_imagenet import Dataset
from matplotlib.colors import Normalize
from src.utils import DataLoader
from src.model.arch import Model
from src.utils import configure
import torch.nn.functional as F
import matplotlib.pyplot as plt
from src import const
import random
import torch
import sys


def visualize(model, gen):
    classes = ['dog sled', 'volleyball', 'baseball player']
    with torch.no_grad():
        for idx, (X, (heatmap, y)) in enumerate(DataLoader(gen, batch_size=1, shuffle=False)):
            #print(sample)

            # X, (heatmap, y) = gen[sample]
            y_pred, cam = model(X)
            X, y = X[0], y[0]

            intra_diff = cam[0][(1-y).nonzero().flatten()]
            intra_diff = (intra_diff[0] - intra_diff[1]).norm(p=1)
            print(idx, intra_diff, y_pred.max())

            target_idx = y.argmax().item()
            if target_idx != 0 or intra_diff < 1: continue

            fig = plt.figure(figsize=(14, 14), facecolor='white')
            cc = F.interpolate(model.get_contrastive_cams(y.unsqueeze(0), cam).transpose(0, 1).detach(), const.IMAGE_SIZE, mode='bilinear').squeeze(1)

            val_range = cc.abs().max()
            crange = cam[0, target_idx].abs().max()
            norm = Normalize(vmin=-val_range, vmax=val_range)

            fig.add_subplot(1, 4, 1)
            plt.imshow(X.permute(1,2,0).cpu(), alpha=.5)
            plt.imshow(F.interpolate(cam[None, :, target_idx], const.IMAGE_SIZE, mode='bilinear')[0, 0].cpu(), norm=Normalize(vmin=-crange, vmax=crange), cmap='jet', alpha=.5)
            plt.xlabel('HiResCAM')

            for i in range(cc.size(0)):
                fig.add_subplot(1, 4, i+2)
                plt.imshow(X.permute(1,2,0).cpu(), alpha=.5)
                plt.imshow(cc[i].cpu(), norm=norm, cmap='jet', alpha=.5)
                if i != target_idx: plt.xlabel(f'Compares to: {classes[i]}')

            fig.add_subplot(1, 4, target_idx + 2)
            plt.imshow(X.permute(1,2,0).cpu())

            plt.xlabel(f'Pred: {classes[y_pred[0].argmax().item()]}, Label: {classes[target_idx]}')

            plt.tight_layout()
            plt.show()

def save(model, gen):
    classes = ['dog sled', 'volleyball', 'baseball player']
    with torch.no_grad():
        for idx in []:
            X, (heatmap, y) = gen[idx]
            y_pred, cam = model(X[None,])

            intra_diff = cam[0][(1-y).nonzero().flatten()]
            intra_diff = (intra_diff[0] - intra_diff[1]).norm(p=1)

            print(idx, intra_diff, y_pred.max())
            target_idx = y.argmax().item()
            cc = F.interpolate(model.get_contrastive_cams(y.unsqueeze(0), cam).transpose(0, 1).detach(), const.IMAGE_SIZE, mode='bilinear').squeeze(1)

            val_range = cc.abs().max()
            crange = cam[0, target_idx].abs().max()
            norm = Normalize(vmin=-val_range, vmax=val_range)

            plt.imshow(X.permute(1,2,0).cpu(), alpha=.6)
            plt.imshow(F.interpolate(cam[None, :, target_idx], const.IMAGE_SIZE, mode='bilinear')[0, 0].cpu(), norm=Normalize(vmin=-crange, vmax=crange), cmap='jet', alpha=.4)

            plt.yticks([])
            plt.xticks([])

            plt.axis('tight')
            plt.axis('image')
            plt.axis('off')

            plt.savefig(f'{idx}_hrc.png', pad_inches=0, bbox_inches='tight')
            plt.clf()

            for i in range(cc.size(0)):
                plt.imshow(X.permute(1,2,0).cpu(), alpha=.6)
                plt.imshow(cc[i].cpu(), norm=norm, cmap='jet', alpha=.4)
                if i != target_idx:
                    plt.yticks([])
                    plt.xticks([])

                    plt.axis('tight')
                    plt.axis('image')
                    plt.axis('off')

                    plt.savefig(f'{idx}_cc_{i}.png', pad_inches=0, bbox_inches='tight')
                    plt.clf()

            plt.imshow(X.permute(1,2,0).cpu())

            plt.yticks([])
            plt.xticks([])

            plt.axis('tight')
            plt.axis('image')
            plt.axis('off')

            plt.savefig(f'{idx}_img.png', pad_inches=0, bbox_inches='tight')
            plt.clf()

if __name__ == '__main__':
    name = sys.argv[1]

    configure(name)
    model = Model(is_contrastive=True)
    model.load_state_dict(torch.load(const.DOWNSTREAM_MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    ds = Dataset('val', ft=True, class_subset=const.N_CLASSES == 3)
    #visualize(model, ds)
    save(model, ds)
