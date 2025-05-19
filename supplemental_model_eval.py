from .rfs import *
from .saliency_analysis import *
from .ablations import *
from .finetuner import *

import sys

from src import const

from src.model.arch import Model
from src.utils import configure

'''
A reviewer requested evaluating additional pretrained models. We do so here.

This file will also serve as an example for how to evaluate any model
'''

### model loading
def load_supplemental_model(ft=False):
    '''
    First, one will need to load the model they wish to evalute. In addition to the model, you will need
    to specify the target_layer used for GradCAM in order to do saliency analysis. Check the following repo
    for more info on picking the target_layer: https://github.com/jacobgil/pytorch-grad-cam

    We demonstrate loading a few different models, who's results we present in the appendix. Check utils.py
    for some useful auxiliary model loading functions.

    To repeat experiments in main text, it suffices to use the 'get_model(mkey, ft)' command from utils.py

    We also show the general procedure for loading a finetuned model, triggered when ft=False
    '''
    name = sys.argv[1]

    configure(name)
    model = Model(is_contrastive='default' not in name, logits_only=True, return_logits=True)
    model.load_state_dict(torch.load(const.DOWNSTREAM_MODELS_DIR / f'{name}.pt', map_location=const.DEVICE, weights_only=True))
    model.name = name
    model.eval()

    return model, model.backbone.layer4[-1].conv3

### Noise based analysis: RFS
def eval_rfs(model, dset_name='hard_imagenet', sigma=0.25, l2=False, ft=False, apply_norm=False):
    rfs, noisy_fg_acc, noisy_bg_acc, noisy_fg_acc_by_class, noisy_bg_acc_by_class = \
                                noisy_fg_bg_accs(model, dset_name, sigma, apply_norm, l2, True)
    print(dset_name, rfs, noisy_fg_acc, noisy_bg_acc, sigma)
    return rfs

### Saliency analysis: IoU
def eval_saliency_iou(model, target_layer, mtype, dset_name='hard_imagenet', ft=False):
    dset = get_dset(dset_name, ft=True)
    ious, delta_densities = compute_salient_alignment_scores(dset, model, target_layer, mtype)
    print(dset_name, np.mean(list(ious.values())), 'saliency iou')
    return ious

### Ablation analysis
def eval_ablations(model, dset_name='hard_imagenet', ft=False):
    dset = get_dset(dset_name, ft=True)
    ablation_names_fns_batchsizes = [('tile', tile, 1), ('replace_with_gray', replace_with_gray, 16), 
            ('replace_bbox_with_gray', lambda x,y: replace_with_gray(x,y,False), 16), ('none', lambda x,y: x, 16)]
    results_dict = dict()
    for ab_name, ab, bs in ablation_names_fns_batchsizes:
        acc, confidences = eval_under_ablation(model, ab, dset, bs)
        results_dict[ab_name] = dict({'acc': acc, 'confidences': confidences})
        # print(len(confidences))
        print(dset_name, acc, np.average(confidences), ab_name)
    return results_dict

###### analysis for appendix/rebutttal
def eval_supplementary_models():
    # mkey_list = ['timm_swin_small_patch4_window7_224', 'timm_convit_small', 'timm_densenet161', 'timm_vgg16']#, 'clip_RN50', 'clip_ViT_b16']
    # mkey_list = ['clip_RN50', 'clip_ViT_b16']
    
    results_path = 'supplemental_models_eval'
    results = load_cached_results(results_path)
    model, target_layer = load_supplemental_model()
    mkey = 'resnet50_contrastive'
    if mkey not in results:
        results[mkey] = dict()
    for dset_name in ['hard_imagenet']:
        # if dset_name in results[mkey]:
        #     continue
        # else:
        results[mkey][dset_name] = dict()
        # noise based analysis
        results[mkey][dset_name]['rfs'] = eval_rfs(model, dset_name=dset_name, apply_norm=False)
        results[mkey][dset_name]['rfs'] = eval_rfs(model, dset_name=dset_name, sigma=0.5, apply_norm=False)
        # ablation analysis
        results[mkey][dset_name]['ablation'] = eval_ablations(model, dset_name=dset_name)
        # saliency analysis
        results[mkey][dset_name]['saliency'] = eval_saliency_iou(model, target_layer, dset_name=dset_name, mtype=mkey)
        # cache_results(results_path, results)

def print_latex_table_supplementary_models():
    results = load_cached_results('supplemental_models_eval')
    vals = dict()
    template_for_vals = ' & ${:.2f}$' * 6
    for dset_name in ['hard_imagenet', 'rival20']:
        dname = 'Hard ImageNet' if 'hard' in dset_name else 'RIVAL20'
        vals[dset_name] = dict()
        print("\\multicolumn{7}{c}" + '{'+ dname + '}' + '\\\\')
        for mkey in ['timm_swin_small_patch4_window7_224', 'timm_convit_small', 'timm_densenet161', 'timm_vgg16']:
            

            row_str = mkey.split('_')[1].title()
            vals[dset_name][mkey] = []
            for metric in ['ablation', 'rfs', 'saliency']:
                if metric == 'ablation':
                    for ab_name in ['none', 'replace_with_gray', 'replace_bbox_with_gray', 'tile']:
                        vals[dset_name][mkey].append(results[mkey][dset_name][metric][ab_name]['acc']*100)
                        # row_str += '  &  ${:.2f}$'.format(results[mkey][dset_name][metric][ab_name]['acc']*100)
                elif metric == 'rfs':
                    vals[dset_name][mkey].append(results[mkey][dset_name][metric])
                    # row_str += '  &  ${:.2f}$'.format(results[mkey][dset_name][metric])
                else:
                    vals[dset_name][mkey].append(np.nanmean(list(results[mkey][dset_name][metric].values()))*100)
                    # row_str += '  &  ${:.2f}$'.format(np.nanmean(list(results[mkey][dset_name][metric].values()))*100)
            row_str = row_str + template_for_vals.format(*vals[dset_name][mkey]) + '\\\\'
            print(row_str)

    print("\\multicolumn{7}{c}{Hard ImageNet - RIVAL20} \\\\")
    for mkey in ['timm_swin_small_patch4_window7_224', 'timm_convit_small', 'timm_densenet161', 'timm_vgg16']:
        row_str = mkey.split('_')[1].title() + template_for_vals.format(*[x-y for x,y in zip(vals['hard_imagenet'][mkey], vals['rival20'][mkey])]) + '\\\\'
        print(row_str)


if __name__=='__main__':
    eval_supplementary_models()
    # print_latex_table_supplementary_models()
