__author__ = "Vishnu Dutt Sharma"

import argparse
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

from tqdm import tqdm
from dataloader import OccMapDataset
from models import UNetRegression as UNet

from torchvision.utils import make_grid, save_image

## Setting random seeds
torch.manual_seed(1234)
import random
random.seed(1234)
np.random.seed(1234)

# from train import Solver

import matplotlib.pyplot as plt
# import seaborn as sns

import torchvision.transforms.functional as F
from sklearn.metrics import confusion_matrix


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
        
    print(np.asarray(img).shape)
    fix, axs = plt.subplots(ncols=len(imgs), squeeze=False)
    for i, img in enumerate(imgs):
        '''
        print(img.min(), img.max())
        odds = torch.log(img/(1-img))
        print(odds.min(), odds.max())
        img = torch.exp(10*odds)/(1+torch.exp(10*odds))
        print(img.min(), img.max())
        '''
        img = img.detach()
        img = F.to_pil_image(img)
        axs[0, i].imshow(np.asarray(img)[:,:,0], cmap='RdYlBu_r')
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
##########################################################################################
##########################################################################################
##########################################################################################

def parge_arguments():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=128, help='Batch size')
    parser.add_argument('--model-path', '-m', dest='model_path', type=str, default=None, help='Model .pth file location')
    parser.add_argument('--error-margin', '-em', dest='margin', metavar='E', type=float, default=5, help='Error margin. Default is 5%%. It means regions with output probability within 5%% of 0.50 are considered as unknown/uncertain')
    parser.add_argument('--show', '-s', dest='show', default='False', action='store_true', help='Show the plots')
    parser.add_argument('--device', '-d', dest='device', type=str, default=None, help='Device type default is cpu')
    return parser.parse_args()

def convert_to_occ(arr, low_prob_thresh=0.495, high_prob_thresh=0.505):
    occ_map = np.zeros(arr.shape, dtype=np.int) # default unknown
    occ_map[arr < low_prob_thresh] = -1 # free
    occ_map[arr > high_prob_thresh] = 1 # occupied
    
    return occ_map


# def get_confusion_matrix():
    

if __name__ == '__main__':
    args = parge_arguments()
    if args.device is not None:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Defining transform
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.ConvertImageDtype(torch.float)
            ])
    # load the data
   # load the data
    test_set = OccMapDataset(filename='./updated_description_ang0.csv', 
                             transform=transform, 
                             mode='test', 
                             odds_to_prob=True,
                             prob_scale=10, 
                             count_to_odd=True,
                             to_class=False)
    
    test_size = len(test_set)

    print(f'Test data size: {test_size}')

    # data loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # model
    net = UNet(n_channels=1, n_classes=1, bilinear=True)
    
    # load the model
    model_path = args.model_path
    
    net = torch.load(model_path, map_location=device)
    net = net.to(device)
    net.eval();

    input_list = []
    gt_list = []
    pred_list = []
    ## Plotting results
    
    glbptr = 0
    for data in tqdm(test_loader):
        images = data['input image']
        labels = data['target image']

        # placing data on device
        images = images.to(device)
        labels = labels.to(device)

        # We don't need gradients here
        with torch.no_grad():
            # forward propagation 
            preds = net(images)
        
        input_list.append(images.cpu().data.numpy())
        gt_list.append(labels.cpu().data.numpy())
        pred_list.append(preds.cpu().data.numpy())
        
        ## put a smaller unmber to get early results
        glbptr += 1
        if glbptr == 100000:
            break
    
    o_inp = np.concatenate(input_list)
    o_gt = np.concatenate(gt_list)
    o_pred = np.concatenate(pred_list)
    
#     outprefix = 'paper_plots/plot_files/'
#     np.savez(f'{outprefix}/regrsn_bce_v2.npz', inp=o_inp, gt=o_gt, pred=o_pred)
#     exit()
    
    print('o_inp stats: ', o_inp.min(), o_inp.max(), o_inp.mean())
    print('o_pred stats: ', o_pred.min(), o_pred.max(), o_pred.mean())
    print('o_gt stats: ', o_gt.min(), o_gt.max(), o_gt.mean())
    
    em = args.margin / 1000.
    print(f'Using error margin of {args.margin:.3f}% i.e. {em:.5f}')
    
#     prediction_mask = np.load('./prediction_mask.npy')

    em_local = [5, 10, 25, 50, 100, 150, 250]
    inp_dict = {}
    acc_dict = {}
    for em in em_local:
        em_val = em/1000.
        # inpainted = (o_inp == 0.5) & ((o_pred > (0.5+em_val)) | (o_pred < (0.5-em_val)))
        inpainted = (o_inp == 0.5) & ((o_pred > (0.5+em_val)) | (o_pred < (0.5-em_val))) & (o_gt != 0.5) #prediction_mask
        sensed_cells = (o_inp != 0.5)

        inpainted_flat = inpainted.reshape(inpainted.shape[0], -1)
        sensed_cells_flat = sensed_cells.reshape(sensed_cells.shape[0],-1)

        occ_map_pred = convert_to_occ(o_pred, low_prob_thresh=(0.5-em_val), high_prob_thresh=(0.5+em_val))
        occ_map_gt = convert_to_occ(o_gt, low_prob_thresh=(0.5-em_val), high_prob_thresh=(0.5+em_val))
        match  = (occ_map_pred == occ_map_gt)
        match_flat = match.reshape(match.shape[0], -1)

        frac_inp = inpainted.reshape(inpainted.shape[0], -1).sum(axis=1)/(o_inp.shape[-1]*o_inp.shape[-2])
        acc = (match_flat * inpainted_flat).sum(axis=1)/inpainted_flat.sum(axis=1)
        
        if np.isnan(acc.mean()):
            break
        
#         print(f'inpainted: {inpainted.shape}')
#         print(f'sensed_cells: {sensed_cells.shape}')
#         print(f'inpainted_flat: {inpainted_flat.shape}')
#         print(f'match_flat: {match_flat.shape}')
        
        inp_dict[em] = frac_inp
        acc_dict[em] = acc

        print(f'Margin: {em}, %inp: {frac_inp.mean()*100}, acc_dict: {acc.mean()*100}')

    em_local = em_local[:len(acc_dict)]
    
    #########################################################
    ### Confusion  matrix
    em_val = 5/1000.
    occ_map_pred = convert_to_occ(o_pred, low_prob_thresh=(0.5-em_val), high_prob_thresh=(0.5+em_val))
    occ_map_gt = convert_to_occ(o_gt, low_prob_thresh=(0.5-em_val), high_prob_thresh=(0.5+em_val))

    cm_list = []
    for itr in range(occ_map_pred.shape[0]):
        cm = confusion_matrix(y_true=occ_map_gt[itr].flatten(), y_pred=occ_map_pred[itr].flatten(), normalize='all')
        cm_list.append(cm)

    print('Confusion matrix. rows: true, cols=pred, free-> unk-> occ')
    print(np.array(cm_list).mean(axis=0))
    print()
    ##########################################################
    
    
    figs, axes = plt.subplots(1,3)
    axes[0].bar(x=np.arange(len(acc_dict)), height=[100*acc_dict[x].mean() for x in em_local], width=0.5)
    axes[0].set_xticks(np.arange(len(acc_dict)))
    axes[0].set_xticklabels([r'0.5$\pm$'+f'{x/1000.:.3f}' for x in em_local], rotation =-30)
    axes[0].set_title('Avg. accuracy v/s Error margin')

    frac_inp = 100*inp_dict[int(args.margin)] #inpainted.reshape(inpainted.shape[0], -1).sum(axis=1)/(o_inp.shape[-1]*o_inp.shape[-2])
    # sns.histplot(frac_inp, ax=axes[0]).set_title('% cells inpainted')
    axes[1].hist(frac_inp)
    axes[1].set_title('Histogram of cells inpainted')

    acc = 100*acc_dict[int(args.margin)] #(match_flat * inpainted_flat).sum(axis=1)/inpainted_flat.sum(axis=1)
    # sns.histplot(acc).set_title('Accuracy histogram')
    axes[2].hist(acc)
    axes[2].set_title('Accuracy histogram')
    image_path = model_path.replace('.pth', '_METRICS.png')
    plt.savefig(image_path)

    print(f'Average inpainted cells: {frac_inp.mean():.3f}%')
    print(f'Average accuracy: {acc.mean():.3f}%')
    
    
    ####################################################
    #### Converting to classification type input
    em_val = 5/1000.
    num_examples = min(5, len(images)) 
    
    occ_map_inp = convert_to_occ(input_list[0][:num_examples], 
                                 low_prob_thresh=(0.5-em_val), high_prob_thresh=(0.5+em_val))
    occ_map_gt = convert_to_occ(gt_list[0][:num_examples], 
                                low_prob_thresh=(0.5-em_val), high_prob_thresh=(0.5+em_val))
    occ_map_pred = convert_to_occ(pred_list[0][:num_examples], 
                                  low_prob_thresh=(0.5-em_val), high_prob_thresh=(0.5+em_val))
    
    ### If classification isn't desired
    """
    occ_map_inp = input_list[0][:num_examples]
    occ_map_gt = gt_list[0][:num_examples]
    occ_map_pred = pred_list[0][:num_examples]
    """
    
    ####################################################
    num_examples = min(5, len(images))
    images = (torch.from_numpy(occ_map_inp).float() + 1)/2
    labels = (torch.from_numpy(occ_map_gt).float() + 1)/2
    preds = (torch.from_numpy(occ_map_pred).float() + 1)/2
    image_path = model_path.replace('.pth', '_TEST.png')
    
#     save_image(make_grid(torch.cat([images, labels, preds], axis=0).cpu(), nrow=num_examples), image_path )
#     np.save('bce_pred.npy', pred_list[0][:num_examples])
    
    if args.show:
        """
        img = images
        # print(img.min(), img.max())
        odds = torch.log(img/(1-img))
        # print(odds.min(), odds.max())
        img = torch.exp(10*odds)/(1+torch.exp(10*odds))
        # print(img.min(), img.max())
        images = img

        img = labels
        # print(img.min(), img.max())
        odds = torch.log(img/(1-img))
        # print(odds.min(), odds.max())
        img = torch.exp(10*odds)/(1+torch.exp(10*odds))
        # print(img.min(), img.max())
        labels = img
        
        img = preds
        # print(img.min(), img.max())
        odds = torch.log(img/(1-img))
        # print(odds.min(), odds.max())
        img = torch.exp(10*odds)/(1+torch.exp(10*odds))
        # print(img.min(), img.max())
        preds = img
        """
        show(make_grid(torch.cat([images, labels, preds], axis=0).cpu(), nrow=num_examples))
        plt.savefig(image_path, dpi=300)
        plt.show()

    print('Done.')
