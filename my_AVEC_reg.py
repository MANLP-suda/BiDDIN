import numpy as np
np.random.seed(1234)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torch.optim as optim

import argparse
import time

from sklearn.metrics import mean_absolute_error,mean_squared_error
import math
from scipy.stats import pearsonr
from scipy import stats

import pandas as pd

from best_model_revised import AVECModel, AVECBiModel, AVECBiModelTri,IEMOCAPBiModelTri,MaskedMSELoss
from my_model_for_speech import AVECBiModelBi_Reg
from dataloader import AVECDataset

def R(pY, Y):
    gradient, intercept, r_value, p_value, std_err = stats.linregress(Y, pY)
    return r_value

#def R2(Y, pY):
#    return r2_score(Y, pY)

def MAE(Y, pY):
    return mean_absolute_error(Y, pY)

def MSE(Y, pY):
    return mean_squared_error(Y, pY)
    
def RMSE(pY,Y):
    n=len(pY)
    return math.sqrt(sum([(pY[i]-Y[i])*(pY[i]-Y[i]) for i in range(n)])/n) 


def get_train_valid_sampler(trainset, valid=0.1):
    size = len(trainset)
    idx = range(size)
    split = int(valid*size)
    return SubsetRandomSampler(idx[split:]), SubsetRandomSampler(idx[:split])

def get_AVEC_loaders(path, batch_size=32, valid=0.1, num_workers=0, pin_memory=False):
    trainset = AVECDataset(path=path)
    train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid)
    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    valid_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = AVECDataset(path=path, train=False)
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def train_or_eval_model(model, loss_function, dataloader, epoch, optimizer=None, train=False):
    losses = []
    preds = []
    labels = []
    masks = []
    assert not train or optimizer!=None
    if train:
        model.train()
    else:
        model.eval()
    for data in dataloader:
        if train:
            optimizer.zero_grad()
        textf, visuf, acouf, qmask, umask, label =\
                                [d.cuda() for d in data] if cuda else data
        # pred = model(textf,acouf, qmask, umask) # batch*seq_len
        pred = model(textf,visuf,acouf, qmask, umask) # batch*seq_len
        labels_ = label.view(-1) # batch*seq_len
        umask_ = umask.view(-1) # batch*seq_len
        loss = loss_function(pred, labels_, umask_)

        preds.append(pred.data.cpu().numpy())
        labels.append(labels_.data.cpu().numpy())
        masks.append(umask_.cpu().numpy())

        losses.append(loss.item()*masks[-1].sum())
        if train:
            loss.backward()
#            if args.tensorboard:
#                for param in model.named_parameters():
#                    writer.add_histogram(param[0], param[1].grad, epoch)
            optimizer.step()

    if preds!=[]:
        preds  = np.concatenate(preds)
        labels = np.concatenate(labels)
        masks  = np.concatenate(masks)
#        print ('preds',type(pred), preds[50:150])
#        print ('labels',type(labels), labels[50:150])
#        print ('masks',type(masks), masks[50:150])
    else:
        return float('nan'), float('nan'), float('nan'), [], [], []
    use_preds = []
    use_labels = []
#    print ('preds size', len(preds))
    for i in range(len(preds)):
        mask = int(masks[i])
        if mask == 1:
            use_preds.append(preds[i])
            use_labels.append(labels[i])
            
    use_preds = np.array(use_preds)
    use_labels = np.array(use_labels)
    my_r = R(use_preds, use_labels)
    my_mae = MAE(use_preds, use_labels)
    
    avg_loss = round(np.sum(losses)/np.sum(masks),4)
    mae = round(mean_absolute_error(labels,preds,sample_weight=masks),4)
    pred_lab = pd.DataFrame(list(filter(lambda x: x[2]==1, zip(labels, preds, masks))))
    pear = round(pearsonr(pred_lab[0], pred_lab[1])[0], 4)
#    return avg_loss, mae, pear, labels, preds, masks, my_r, my_mae
    return avg_loss, mae, pear, labels, preds, masks

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='does not use GPU')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate')
    parser.add_argument('--l2', type=float, default=0.0001, metavar='L2',
                        help='L2 regularization weight')
    parser.add_argument('--rec-dropout', type=float, default=0.0,
                        metavar='rec_dropout', help='rec_dropout rate')
    parser.add_argument('--dropout', type=float, default=0.0, metavar='dropout',
                        help='dropout rate')
    parser.add_argument('--batch-size', type=int, default=30, metavar='BS',
                        help='batch size')
    parser.add_argument('--epochs', type=int, default=100, metavar='E',
                        help='number of epochs')
    parser.add_argument('--active-listener', action='store_true', default=False,
                        help='active listener')
    parser.add_argument('--attention', default='general', help='Attention type')
    parser.add_argument('--tensorboard', action='store_true', default=False,
                        help='Enables tensorboard log')
    parser.add_argument('--attribute', type=int, default=1, help='AVEC attribute')
    args = parser.parse_args()

    print(args)

    args.cuda = torch.cuda.is_available() and not args.no_cuda
    if args.cuda:
        print('Running on GPU')
    else:
        print('Running on CPU')

    # if args.tensorboard:
    #     from tensorboardX import SummaryWriter
    #     writer = SummaryWriter()

    batch_size = args.batch_size
    cuda       = args.cuda
    n_epochs   = args.epochs

    D_m_T = 100
    D_m_V = 512
    D_m_A = 100

    D_m = 100
    D_g = 100
    D_p = 100
    D_e = 100
    D_h = 100

    D_a = 100 # concat attention

    model = IEMOCAPBiModelTri(D_m_T, D_m_V, D_m_A, D_m, D_g, D_p, D_e, D_h,
                    listener_state=args.active_listener,
                    context_attention=args.attention,
                    dropout_rec=args.rec_dropout,
                    dropout=args.dropout)
    # model = AVECBiModelBi_Reg(D_m_T, D_m_A, D_m, D_g, D_p, D_e, D_h,
    #                 listener_state=args.active_listener,
    #                 context_attention=args.attention,
    #                 dropout_rec=args.rec_dropout,
    #                 dropout=args.dropout)

    if cuda:
        model.cuda()
    loss_function = MaskedMSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr,
                           weight_decay=args.l2)

    train_loader, valid_loader, test_loader =\
            get_AVEC_loaders('./DialogueRNN_features/DialogueRNN_features/AVEC_features/AVEC_features_{}.pkl'.format(1),
                                valid=0.0,
                                batch_size=32,
                                num_workers=2)

    best_loss, best_label, best_pred, best_mask, best_pear = None, None, None, None, None
    my_best_pear, my_best_mae = None, None
    for e in range(n_epochs):
        start_time = time.time()
        
        train_loss, train_mae, train_pear,_,_,_ = train_or_eval_model(model, loss_function,
                                               train_loader, e, optimizer, True)
        valid_loss, valid_mae, valid_pear,_,_,_ = train_or_eval_model(model, loss_function, valid_loader, e)
        test_loss, test_mae, test_pear, test_label, test_pred, test_mask = train_or_eval_model(model, loss_function, test_loader, e)

        if best_loss == None or best_loss > test_loss:
            best_loss, best_label, best_pred, best_mask, best_pear =\
                    test_loss, test_label, test_pred, test_mask, test_pear
        
        use_preds = []
        use_labels = []
    #    print ('preds size', len(preds))
        for i in range(len(test_pred)):
            mask = int(test_mask[i])
            if mask == 1:
                use_preds.append(test_pred[i])
                use_labels.append(test_label[i])
                
        use_preds = np.array(use_preds)
        use_labels = np.array(use_labels)
        my_r = R(use_preds, use_labels)
        my_mae = MAE(use_preds, use_labels)
        
        print ('epoch {}, test pear {}, test mae {}'.format(e+1,my_r,my_mae))
    
        print('epoch {} train_loss {} train_mae {} train_pear {} valid_loss {} valid_mae {} valid_pear {} test_loss {} test_mae {} test_pear {} time {}'.\
                format(e+1, train_loss, train_mae, train_pear, valid_loss, valid_mae,\
                        valid_pear, test_loss, test_mae, test_pear, round(time.time()-start_time,2)))

    print('Test performance..')
    print('Loss {} MAE {} r {}'.format(best_loss,
                                 round(mean_absolute_error(best_label,best_pred,sample_weight=best_mask),4),
                                 best_pear))
#        var_list = train_or_eval_model(model, loss_function,
#                                               train_loader, e, optimizer, True)
#        print ('var len', len(var_list))
#        train_loss, train_mae, train_pear, null1, null2, null3, my_train_pear, my_train_mae = train_or_eval_model(model, loss_function,
#                                               train_loader, e, optimizer, True)
#        valid_loss, valid_mae, valid_pear,null4,null5,null6, my_valid_pear, my_valid_mae = train_or_eval_model(model, loss_function, valid_loader, e)
#        test_loss, test_mae, test_pear, test_label, test_pred, test_mask, my_test_pear, my_test_mae = train_or_eval_model(model, loss_function, test_loader, e)
#        
#        if best_loss == None or best_loss > test_loss:
#            best_loss, best_label, best_pred, best_mask, best_pear =\
#                    test_loss, test_label, test_pred, test_mask, test_pear
#                    
#        if my_best_pear < my_test_pear:
#            my_best_pear = my_test_pear
#            my_best_mae = my_test_mae
#
#        print('epoch {} train_loss {} train_mae {} train_pear {} valid_loss {} valid_mae {} valid_pear {} test_loss {} test_mae {} test_pear {} time {}'.\
#                format(e+1, train_loss, train_mae, train_pear, valid_loss, valid_mae,\
#                        valid_pear, test_loss, test_mae, test_pear, round(time.time()-start_time,2)))
#        print('epoch {} train_loss {} my_train_mae {} my_train_pear {} valid_loss {} my_valid_mae {} my_valid_pear {} test_loss {} my_test_mae {} my_test_pear {} time {}'.\
#                format(e+1, train_loss, train_mae, train_pear, valid_loss, valid_mae,\
#                        valid_pear, test_loss, test_mae, test_pear, round(time.time()-start_time,2)))
#
#    print('Test performance..')
#    print('Loss {} MAE {} r {}'.format(best_loss,
#                                 round(mean_absolute_error(best_label,best_pred,sample_weight=best_mask),4),
#                                 best_pear))
#    print('my best MAE {} r {}'.format(round(my_best_mae,4),
#                                 my_best_pear))
