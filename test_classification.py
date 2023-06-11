__author__ = "Vishnu Dutt Sharma"

import argparse
import numpy as np

import torch
import torch.nn as nn
from torchvision import transforms

from tqdm import tqdm
from dataloader import OccMapDataset
from models import UNetClassification

from sklearn.metrics import confusion_matrix

from torchvision.utils import make_grid, save_image

## Setting random seeds
torch.manual_seed(1234)
import random
random.seed(1234)
np.random.seed(1234)

import matplotlib.pyplot as plt
# import seaborn as sns

import torchvision.transforms.functional as F


plt.rcParams["savefig.bbox"] = 'tight'


def show(imgs):
    if not isinstance(imgs, list):
        imgs = [imgs]
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
        axs[0, i].imshow(np.asarray(img).argmax(axis=2), cmap='RdYlBu_r')
        axs[0, i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])
    
#     axs[0, 0].set_ylabel('Input')
#     axs[1, 0].set_ylabel('Ground Truth')
#     axs[2, 0].set_ylabel('Prediction')
    
##########################################################################################
##########################################################################################
##########################################################################################

def parge_arguments():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=128, help='Batch size')
    parser.add_argument('--model-path', '-m', dest='model_path', type=str, default=None, help='Model .pth file location')
    parser.add_argument('--show', '-s', dest='show', action='store_true', default=False, help='Show the plots')
    parser.add_argument('--device', '-d', dest='device', type=str, default=None, help='Device type default is cpu')
    return parser.parse_args()

def convert_to_occ(arr, low_prob_thresh=0.495, high_prob_thresh=0.505):
    occ_map = np.zeros(arr.shape, dtype=np.int) # default unknown
    occ_map[arr < low_prob_thresh] = -1 # free
    occ_map[arr > high_prob_thresh] = 1 # occupied
    
    return occ_map


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
    test_set = OccMapDataset(filename='./updated_description_ang0.csv', 
                             transform=transform, 
                             mode='test', 
                             odds_to_prob=True,
                             prob_scale=10, 
                             count_to_odd=True,
                             to_class=True,
                             with_seg=False)
    
    
    test_size = len(test_set)

    print(f'Test data size: {test_size}')

    # data loader
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # model
    net = UNetClassification(n_channels=3, n_classes=3, bilinear=True)
    
    # load the model
    model_path = args.model_path
    
    """
    ## Creating a dummy solver to test the model
    solver = Solver(net, optimizer='sgd', 
                    loss_fn='mse', 
                    lr=0.01, max_epoch=1, 
                    verbose=True, 
                    save_best=True, 
                    early_stop=5, 
                    outfile=model_path, 
                    save_full=True, 
                    loss_scale=1.0, 
                    device=device)
    
    solver.net = torch.load(model_path, map_location=solver.device)
    solver.net = solver.net.to(solver.device)
    solver.net.eval()

    """
    
    net = torch.load(model_path, map_location=device)
    net = net.to(device)
    net.eval()

    input_list = []
    gt_list = []
    pred_list = []
    ## Plotting results
    for data in tqdm(test_loader):
        images = data['input image']
        labels = data['target image'][:,0,:,:].long()
        
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
        
#         break
        
    o_inp = np.concatenate(input_list)
    o_gt = np.concatenate(gt_list)
    o_pred = np.concatenate(pred_list) 
    
#     outprefix = 'paper_plots/plot_files/'
#     np.savez(f'{outprefix}/clsfn_crossent_v2.npz', inp=o_inp, gt=o_gt, pred=o_pred)
#     exit()
    
    print('o_inp  stats (min, max, mean): ', o_inp.min(), o_inp.max(), o_inp.mean())
    print('o_pred stats (min, max ,mean): ', o_pred.min(), o_pred.max(), o_pred.mean())
    print('o_gt   stats (min, max, mean): ', o_gt.min(), o_gt.max(), o_gt.mean())
    
    matches = (o_pred.argmax(1) == o_gt)
    print(f'Total Accuracy: {matches.reshape((matches.shape[0],-1)).mean(1).mean()}')
    
    #########################################################
    ### Confusion  matrix
    cm_list = []
    for itr in range(o_pred.shape[0]):
        cm = confusion_matrix(y_true=o_gt[itr].flatten(), y_pred=o_pred[itr].argmax(axis=0).flatten(), normalize='all')
        cm_list.append(cm)
    
#     print(o_gt.shape, o_pred.shape)
    print('Confusion matrix. rows: true, cols=pred, free-> unk-> occ')
    print(np.array(cm_list).mean(axis=0))
    print()
    ##########################################################
    
    num_examples = min(5, len(images))
    images = torch.from_numpy(input_list[0][:num_examples])
    labels = nn.functional.one_hot(torch.from_numpy(gt_list[0][:num_examples]), num_classes=3).permute((0,3,1,2))
    preds = nn.functional.one_hot(torch.from_numpy(pred_list[0][:num_examples]).argmax(dim=1), num_classes=3).permute((0,3,1,2))
    image_path = model_path.replace('.pth', '_TEST.png')
    
#     save_image(make_grid(torch.cat([images, labels, preds], axis=0).cpu()*255., nrow=num_examples, normalize=True), image_path )
#     np.save('bce_pred.npy', pred_list[0][:num_examples])
    
    if args.show:        
        show(make_grid(torch.cat([images, labels, preds], axis=0).cpu()*255., nrow=num_examples, normalize=True))
        plt.savefig(image_path, dpi=300)
        plt.show()

    print('Done.')
