__author__ = "Vishnu Dutt Sharma"

import argparse
import numpy as np

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
from torchvision import transforms

from tqdm import tqdm
import time
from dataloader import OccMapDataset
from models import UNetClassification

from torchvision.utils import make_grid, save_image

## Setting random seeds
torch.manual_seed(1234)
import random
random.seed(1234)
np.random.seed(1234)

class Solver(object):
    def __init__(self, net, 
                 optimizer='sgd', 
                 loss_fn='mse', 
                 lr=0.1, 
                 max_epoch=10, 
                 verbose=True,
                 save_best=True, 
                 early_stop=None, 
                 outfile='./models/some_net.pth',
                 logdir='./tblogdir/',
                 save_freq=50,
                 save_full=True, 
                 loss_scale=1.0, 
                 device=None):
        # Your code 
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device

        self.net = net.to(self.device)
        
        if loss_fn == 'mse':
            self.criterion = nn.MSELoss()
        elif loss_fn == 'crossent':
            self.criterion = nn.CrossEntropyLoss()
        elif loss_fn == 'mse_prob':
            def MSEprobloss(pred_odds, gt_odds):
                pred_prob = torch.exp(pred_odds)/(1 + torch.exp(pred_odds))
                gt_prob = torch.exp(gt_odds)/(1 + torch.exp(gt_odds))
                return F.mse_loss(pred_prob, gt_prob)
            self.criterion = MSEprobloss
        elif loss_fn == 'stretched_mse':
            print('Using power MSE')
            def SMSE(input, target):
                return F.mse_loss(input=torch.pow(input,2), target=torch.pow(target,2))
            self.criterion = SMSE
        elif loss_fn == 'masked_mse':
            print('Using masked MSE')
            def MaskMSE(input, target):
                return F.mse_loss(input=input[input != 0.5], target=target[input != 0.5]).mean()
            self.criterion = MaskMSE
        elif loss_fn == 'masked_bce':
            print('Using masked BCE')
            def MaskBCE(input, target):
                return F.binary_cross_entropy(input=input[input != 0.5], target=target[input != 0.5]).mean()
            self.criterion = MaskBCE
        elif loss_fn == 'kl':
            print('Will use KL assuming inputs/outputs are log-odds')
            def KLloss(pred_odds, gt_odds):
                pred_prob = torch.exp(pred_odds)/(1 + torch.exp(pred_odds))
                gt_prob = torch.exp(gt_odds)/(1 + torch.exp(gt_odds))
                return F.kl_div(input=pred_prob, target=gt_prob)
            self.criterion = KLloss
        elif loss_fn == 'kl_log_n_prob':
            print('Uses log prob for input/prodictions and probs for output/target')
            def KLloss(input, target):
                pred_logprob = torch.log(input)
                gt_prob = target
                return F.kl_div(input=pred_logprob, target=gt_prob, log_target=False, reduce='batchmean')
            self.criterion = KLloss
            test_tensor = torch.rand((8,1,256,256))
            print(f'Testing: loss for same distributions is {self.criterion(test_tensor, test_tensor)}')
        elif loss_fn == 'kl_raw':
            print('Using KL loss diretcly')
            self.criterion = nn.KLDivLoss(log_target=True, reduce='batchmean')
            test_tensor = torch.rand((8,1,256,256))
            print(f'Testing: loss for same distributions is {self.criterion(test_tensor, test_tensor)}')
        elif loss_fn == 'soft_kl':
            print('Using Softmax-KL')
            def SoftKL(input, target):
                pred_soft = torch.distributions.categorical.Categorical(probs=input.flatten(start_dim=1))
                gt_soft = torch.distributions.categorical.Categorical(probs=target.flatten(start_dim=1))
                return torch.distributions.kl.kl_divergence(q=pred_soft, p=gt_soft).sum()
            self.criterion = SoftKL
            test_tensor = torch.rand((8,1,256,256))
            print(f'Testing: loss for same distributions is {self.criterion(test_tensor, test_tensor)}')
        elif loss_fn == 'bce_kl':
            print('Using BCE as KL')
            def BCE_KL(input, target):
                pred = input.flatten(start_dim=1)
                label = target.flatten(start_dim=1)
                cross_ent  = -label * torch.log(pred) - (1-label) * torch.log(1-pred)
                return cross_ent.sum(dim=1).mean()
            self.criterion = BCE_KL
            test_tensor = torch.rand((8,1,256,256))
            print(f'Testing: loss for same distributions is {self.criterion(test_tensor, test_tensor)}')
        elif loss_fn == 'jsdiv':
            print('Using BCE as KL')
            def JS_DIV(input, target):
                pred = input.flatten(start_dim=1)
                label = target.flatten(start_dim=1)
                jsdiv  = -(label+pred) * torch.log(pred) - (2-label-pred) * torch.log(1-pred) + pred * torch.log(label) + (1-pred) * torch.log(1-label)
                return jsdiv.mean()
            self.criterion = JS_DIV
            test_tensor = torch.rand((8,1,256,256))
            print(f'Testing: loss for same distributions is {self.criterion(test_tensor, test_tensor)}')
        elif loss_fn == 'bce':
            print('Using binary cross entropy loss')
            self.criterion = nn.BCELoss()
        elif loss_fn == 'ce_inpaint':
            print('Using CE over inpainted areas only')
            def CE_inpaint(input, target):
                mask = torch.tensor(target != 0.5, dtype=torch.float).clone().detach().requires_grad_(False)
                mask = mask.flatten(start_dim=1).to(self.device)
                pred = input.flatten(start_dim=1)
                label = target.flatten(start_dim=1)
                cross_ent  = -label * torch.log(pred) - (1-label) * torch.log(1-pred)
                cross_ent = cross_ent*mask
                return cross_ent.sum(dim=1).mean()
            self.criterion = CE_inpaint
        elif loss_fn == 'ce_n_mse':
            print('Using CE and MSE together')
            def CE_n_MSE(input, target):
                pred = input.flatten(start_dim=1)
                label = target.flatten(start_dim=1)
                cross_ent  = -label * torch.log(pred) - (1-label) * torch.log(1-pred)
                return cross_ent.sum(dim=1).mean() +torch.nn.functional.mse_loss(input, target)
            self.criterion = CE_n_MSE

        else: # Wasserstien
            raise NotImplementedError
        
        if optimizer == 'sgd': 
            self.optimizer = optim.SGD(self.net.parameters(), lr=lr, momentum=0.9)
        elif optimizer == 'adam':
            self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        else:
            self.optimizer = optim.Adadelta(self.net.parameters(), lr=lr)
        
        if early_stop is not None:
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=3)
#             self.scheduler = optim.lr_scheduler.CyclicLR(self.optimizer, base_lr= lr/10., max_lr=lr)
        
        self.max_epoch = max_epoch
        self.verbose = verbose
        self.early_stop = early_stop
        self.outfile = outfile
        self.logdir = logdir
        self.save_full = save_full
        self.loss_scale = torch.tensor(loss_scale).float()
        self.save_freq = save_freq
        
        self.writer = SummaryWriter(self.logdir + '/' + outfile.replace('.pth', ''))
        print('Model will use:')
        print(f'\tLoss function: {loss_fn}')
        print(f'\tOptimizer: {optimizer}')
        print(f'\tLR: {lr}')
        print(f'\tMax epochs: {max_epoch}')
        print(f'\tEarly stop: {early_stop}')
        print(f'\tLoss Scale: {loss_scale}')

    def train(self, train_loader, valid_loader=None):
        """Function to train the model

        Parameters
        ----------
            train_loader: Training data loader
            valid_loader: Validation data loader
        
        Returns
        -------
            None
        """
        # Your code 
        ## Initialing minimum loss with a large value
        min_valid_loss = np.inf
        
        epsilon = 1e-5
        
        ## Indicator for early stopping
        stopping = False

        ## Lists to save training and validation loss at each epoch
        training_loss_list = []
        validation_loss_list = []

        ## Patience counter for early stopping
        early_stop_count = self.early_stop
        
        ## Global counter
        global_count = 0
        
        ## Iterating over each epoch
        for ep in range(self.max_epoch):  
            ## Initializing episodic loss
            ep_loss = 0.0
            ## Iterating through batches fo data
            for idx, data in enumerate(train_loader):
                # Getting the inputs; data is a list of [inputs, labels]
                images = data['input image']
                labels = data['target image'][:,0,:,:].long()
                
                # placing data on device
                images = images.to(self.device)
                labels = labels.to(self.device)

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward propagation 
                preds = self.net(images)
                
                # calculating loss
                loss = self.loss_scale * self.criterion(input=preds, target=labels)
                # print(self.criterion(input=self.scale * images, target=self.scale * labels))
                # backprop
                loss.backward()
                self.optimizer.step()

                # Getting loss
                ep_loss += loss.item()
                
                """
                # printing progress
                if self.verbose and ((idx+1)% 20 == 0):    # print every 20 mini-batches
                    valid_loss  = self.test(valid_loader)
                    print(f'Episode: {ep+1}, Iteation: {idx+1}, Validation loss: {valid_loss}')
                    self.writer.add_scalar("GlobalLoss/valid", valid_loss, global_count)
                    self.net.train();
                """
                
                # Update global iteration count
                global_count += 1
            
            
            ## Getting the average loss 
            training_loss = ep_loss/len(train_loader)
            
            ## Saving the episodic loss
            training_loss_list.append(training_loss)

            if self.verbose:
                print(f'End of Episode {ep+1}, Training loss: {training_loss}')
            
            self.writer.add_scalar("Loss/train", training_loss, ep+1)
            # self.writer.add_scalar("GlobalLoss/train", training_loss, global_count)

            ## Calculaing the validation loss for this epoch
            valid_loss = self.test(valid_loader)
            ## Moving model back to training model
            self.net.train();

            ## Saving the validation loss for this epoch
            validation_loss_list.append(valid_loss)
            
            ## Printing progress
            if self.verbose:
                print(f'Validation loss: {valid_loss}')
            
            self.scheduler.step(valid_loss)
            # self.scheduler.step()

            self.writer.add_scalar("Loss/valid", valid_loss, ep+1)
            # self.writer.add_scalar("GlobalLoss/valid", valid_loss, global_count)
            
            self.writer.add_scalar('LearningRate', self.optimizer.param_groups[0]['lr'], ep+1)

            if (ep+1)% self.save_freq == 0:
                ## Saving model or model state_dict
                if self.save_full:
                    torch.save(solver.net, self.outfile.replace('.pth', f'_{ep+1}.pth'))
                else:
                    torch.save(solver.net.state_dict(), self.outfile.replace('.pth', f'_{ep+1}.pth'))


            ## If current loss is less than minimum loss so far, update it and save model
            if (valid_loss-min_valid_loss) < -epsilon:
                min_valid_loss = valid_loss
                
                ## Saving model or model state_dict
                if self.save_full:
                    torch.save(solver.net, self.outfile)
                else:
                    torch.save(solver.net.state_dict(), self.outfile)
                
                if self.verbose:
                    print('Saving model')

                ## If early_stopping is enabled, then reset the patience
                if self.early_stop is not None:
                    early_stop_count = self.early_stop

            elif self.early_stop > 0: # if current validation loss is larger than the minimum loss so far, reduce patience
                early_stop_count -= 1
                ## If patience is 0, stop training
                if early_stop_count == 0:
                    stopping = True

            if stopping:
                print(f'Stoppping early')
                break
            
            
            self.writer.flush()
        
        self.writer.close()
        
        print('Training completed')
        

        #### Plotting trainig and test curves
#         plt.plot(np.arange(1,len(training_loss_list)+1), training_loss_list, 'b', label='Training')
#         if valid_loader is not None:
#             plt.plot(np.arange(1,len(validation_loss_list)+1), validation_loss_list, 'g', label='Validation')
        
#         plt.xlabel('#Epochs')
#         plt.ylabel('Loss (Cross-Ent)')
#         plt.legend(loc="upper right")

    def test(self, loader):
        """Function to test the model

        Parameters
        ----------
            loader: Validation or test loader

        Returns
        -------
            float: loss
            float: accuracy
        """
        
#         ## Placeholder to save predictions and GT labels
#         preds_list = []
#         label_list = []
        
        ## Initlaizing the loss
        test_loss  = 0
        
        ## Moving model to eval model
        self.net.eval();
        
        for data in loader:
            # Getting the inputs; data is a list of [inputs, labels]
            images = data['input image']
            labels = data['target image'][:,0,:,:].long()

            # placing data on device
            images = images.to(self.device)
            labels = labels.to(self.device)

            # We don't need gradients here
            with torch.no_grad():
                # forward propagation 
                preds = self.net(images)
            
            # calculating loss
            loss = self.loss_scale * self.criterion(input=preds, target=labels)

#             # Saving the predictions and labels
#             preds_list.append(preds.cpu().data.argmax(1).numpy())
#             label_list.append(labels.cpu().data.numpy())

            # Adding the batch loss to the total loss
            test_loss += loss.item()

#         # Converting lists to arrays for easier processing
#         preds_np = np.concatenate(preds_list)
#         label_arr = np.concatenate(label_list)

#         # Calculating test accuracy
#         test_acc = 100*(preds_np == label_arr).sum()/len(label_arr)
        
        # Calculating average loss
        test_loss_norm = test_loss/len(loader)

        return test_loss_norm#, test_acc
    
##########################################################################################
##########################################################################################
##########################################################################################

def parge_arguments():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--epochs', '-e', dest='ep', metavar='E', type=int, default=100, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=128, help='Batch size')
    parser.add_argument('--learning-rate', '-l', metavar='LR', type=float, default=0.001,
                        help='Learning rate', dest='lr')
    parser.add_argument('--load', '-f', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('--validation', '-v', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('--loss-function', '-lf', type=str, dest='loss_fn', default='mse', help='Loss function. Options are mse, mse_prob, kl_raw, wass')
    parser.add_argument('--loss-scale', '-ls', dest='ls', type=float, default=1.0, help='Scale parameters. Predictions and labels are multiplied with it before calculating the loss. Use only if loss is very small')
    parser.add_argument('--early-stop', '-es', dest='es', type=int, default=-1, help='Patience for early stoppping. Default is -1 i.e. not early stopping')
    parser.add_argument('--descfile', '-df', dest='descfile', type=str, default='./model_descriptions.txt', help='Patience for early stoppping. Default is -1 i.e. not early stopping')
    parser.add_argument('--logdir', '-ld', dest='logdir', type=str, default='./tblogdir/', help='Directory to save tensorboard logs in')

    return parser.parse_args()


if __name__ == '__main__':
    args = parge_arguments()
    
    # Defining transform
    transform = transforms.Compose([
                transforms.ToTensor(),
#                 transforms.RandomVerticalFlip(p=0.5),  ## Removing as it doesn't apply same to both the images
                transforms.ConvertImageDtype(torch.float)
            ])
    
    # load the data
    trainval_set = OccMapDataset(filename='./updated_description_ang0.csv', 
                                 transform=transform, 
                                 mode='train',
                                 odds_to_prob=True,
                                 prob_scale=10, 
                                 count_to_odd=True,
                                 to_class=True,
                                 with_seg=False)
    test_set = OccMapDataset(filename='./updated_description_ang0.csv', 
                             transform=transform, 
                             mode='test', 
                             odds_to_prob=True,
                             prob_scale=10, 
                             count_to_odd=True,
                             to_class=True,
                             with_seg=False)
    
    trainval_size = len(trainval_set)

    train_size = int((100 - args.val)/100. * trainval_size)
    valid_size = trainval_size - train_size
    test_size = len(test_set)

    print(f'Data sizes:\nTrain: {train_size}\nValid: {valid_size}\nTest: {test_size}')

    train_set, valid_set = torch.utils.data.random_split(trainval_set, [train_size, valid_size])

    # data loader
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=2*args.batch_size, shuffle=False, num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=2*args.batch_size, shuffle=False, num_workers=2)
    
    # model
    net = UNetClassification(n_channels=3, n_classes=3, bilinear=True)

    # train the model
    optimizer_name = 'adam'
    model_path = f"./saved_models/classfn_noseg_{optimizer_name}_{args.lr}_epoch_{args.ep}_{args.loss_fn}_scale_{args.ls}_es_{args.es}.pth"
    
    with open(args.descfile, 'a') as dfl:
        dfl.write(','.join([model_path, str(time.time())]) + '\n')
    
    solver = Solver(net, 
                    optimizer=optimizer_name, 
                    loss_fn=args.loss_fn, 
                    lr=args.lr, 
                    max_epoch=args.ep, 
                    verbose=True, 
                    save_best=True, 
                    early_stop=args.es,#5, 
                    outfile=model_path, 
                    save_full=True,
                    loss_scale=args.ls,
                    logdir=args.logdir)
    
    if not args.load:
        solver.train(train_loader, valid_loader)
    else:
        print(f'Loading pre-trained model from {model_path}')
        # solver.net.load_state_dict(torch.load(model_path))
        solver.net = torch.load(model_path)

    test_loss = solver.test(loader=test_loader)
    print(f'Test loss: {test_loss}')
    
    ## Plottig results
    for data in test_loader:
        images = data['input image']
        labels = data['target image']

        # placing data on device
        images = images.to(solver.device)
        labels = labels.to(solver.device)

        # We don't need gradients here
        with torch.no_grad():
            # forward propagation 
            preds = solver.net(images)
        break
    
    num_examples = 5
    image_path = model_path.replace('.pth', '.png')
    save_image(make_grid(torch.cat([images[:num_examples], labels[:num_examples], preds[:num_examples]], axis=0).cpu(), nrow=num_examples), image_path, normalize=True )
    
    print('Done.')
