import pickle, time
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import torch
import numpy as np

from dataset import HV3D
from models import GCN_corr
from opt import Options, setup_folder, save_opt
from evaluation import *

from torch.utils.tensorboard import SummaryWriter

from utils import lr_decay, train_corr, test_corr 


torch.cuda.set_device(2)
print('GPU Index: {}'.format(torch.cuda.current_device()))


def main(opt):
    start_time = time.time()  
    torch.manual_seed(0)
    np.random.seed(0)

    # logging
    date_time = setup_folder(opt)
    writer_tr = SummaryWriter(opt.ckpt_tensorboard+'/tr')
    # writer_val = SummaryWriter(opt.ckpt_datetime+'/val')
    writer_test = SummaryWriter(opt.ckpt_tensorboard+'/test')
    save_opt(opt, writer_tr)
    is_cuda = torch.cuda.is_available()
    print('Loading data...')
    try:
        with open('Data/tmp_noVal.pickle', "rb") as f:
            data = pickle.load(f)
        data_train = data['train']
        data_test = data['test']
        print('Load preserved data...')
    except FileNotFoundError:
        sets = [[0, 1], [2], [3]]
        data_train = HV3D(opt.gt_dir, sets=sets, split=0, is_cuda=is_cuda)
        data_test = HV3D(opt.gt_dir, sets=sets, split=2, is_cuda=is_cuda)
        with open('Data/tmp.pickle', 'wb') as f:
            pickle.dump({'train': data_train, 'test': data_test}, f)

    train_loader = DataLoader(dataset=data_train, batch_size=opt.batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))

    model = GCN_corr()
    if is_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    lr_now = opt.lr
    with tqdm(range(opt.epoch_corr), desc=f'Training Correcter', unit="epoch") as tepoch:
        for epoch in tepoch:
            if (epoch + 1) % opt.lr_decay == 0:
                lr_now = lr_decay(optimizer, lr_now, opt.lr_gamma)
            tr_l = train_corr(train_loader, model, optimizer, is_cuda=is_cuda)
            # writer_tr.add_scalar("training/acc", tr_acc, epoch) 
            writer_tr.add_scalar("train/loss", tr_l, epoch) 
            with torch.no_grad():
                test_l, preds = test_corr(test_loader, model, is_cuda=is_cuda)                                
                writer_test.add_scalar('loss/corr_test', test_l, epoch)  
            tepoch.set_postfix(train_loss=tr_l.item())
    writer_tr.close()
    writer_test.close()

    # save model
    torch.save(model.state_dict(), opt.model_dir)


    model = GCN_corr()
    model.load_state_dict(torch.load(opt.model_dir))

    if is_cuda:
        model.cuda()

    with torch.no_grad():
        test_l, preds = test_corr(test_loader, model, is_cuda=is_cuda)                                

    with open(opt.result_pickle_dir, 'wb') as f:
        pickle.dump({'loss': test_l, 'preds': preds}, f)

    end_time = time.time()
    time_consuming = "Time consuming: {:.2f}".format(end_time - start_time)
    print(time_consuming) 

    # Evaluation of the Corrector
    savepath = option.ckpt+'/Evaluation/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savepath += date_time
    while os.path.exists(savepath):
        savepath += "_x"
    main_eval(date_time, savepath, separated=True, model_corr_path=opt.model_dir, model_class_path='Results-22Spring/Combined_v2/models/2022-06-14-16-34.pt', options=opt, model_combined='')
    print('Evaluation done.')


if __name__ == "__main__":
    option = Options().parse()
    main(option)
