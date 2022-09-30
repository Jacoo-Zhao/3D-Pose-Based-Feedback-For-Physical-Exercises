import time
import torch
import pickle
import numpy as np
from tqdm import tqdm
from evaluation import *
from dataset import EC3D
from torch.utils.data import DataLoader

from models import GCN_corr
from opt import Options, setup_folder, save_opt
from utils import lr_decay, train_corr, test_corr

from torch.utils.tensorboard import SummaryWriter


def main(opt):
    start_time = time.time()
    torch.manual_seed(0)
    np.random.seed(0) 

    date_time = setup_folder(opt)
    writer_tr = SummaryWriter(opt.ckpt_tensorboard+'/train')
    writer_test = SummaryWriter(opt.ckpt_tensorboard+'/test')
    
    save_opt(opt, writer_tr)
    is_cuda = torch.cuda.is_available()
  
    try:
        with open(opt.EC3D_data_path, "rb") as f:
            data = pickle.load(f)
        print('Loading reserved data.')
        data_train = data['train']
        data_test = data['test']
    except FileNotFoundError:
        print('Processing Dataset.')
        sets = [[0, 1, 2], [], [3]]
        data_train = EC3D(opt.raw_data_dir, sets=sets, split=0, is_cuda=is_cuda)
        data_test = EC3D(opt.raw_data_dir, sets=sets, split=2, is_cuda=is_cuda)
        with open('data/EC3D/tmp_wo_val.pickle', 'wb') as f:
            pickle.dump({'train': data_train, 'test': data_test}, f)

    train_loader = DataLoader(dataset=data_train, batch_size=opt.batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))

    model = GCN_corr( hidden_feature=opt.hidden)

    if is_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    lr_now = opt.lr

    whether_train_model = input('Train model? Otherwise use pretrained model.\ny/n:')
    if whether_train_model == 'y':
        print('Start training...')
        with tqdm(range(opt.epoch_corr), desc=f'Training Correcter', unit="epoch") as tepoch:
            for epoch in tepoch:
                if (epoch + 1) % opt.lr_decay == 0:
                    lr_now = lr_decay(optimizer, lr_now, opt.lr_gamma)
                tr_l = train_corr(train_loader, model, optimizer, is_cuda=is_cuda)
                
                # logging
                writer_tr.add_scalar("train/loss", tr_l, epoch) 
                
                with torch.no_grad():
                    te_l, preds = test_corr(test_loader, model, is_cuda=is_cuda)                                
                    writer_test.add_scalar('loss/corr_test', te_l, epoch)  
               
                tepoch.set_postfix(train_loss=tr_l.item())
        
        writer_tr.close()
        writer_test.close()
        train_result = 'Training_loss:{:.3f}'.format(tr_l.item())

        # save & load 
        torch.save(model.state_dict(), opt.model_dir)
    else:
        print('Use pretrained model...')
        opt.model_dir = 'pretrained_weights/Corrector.pt'

    model = GCN_corr( hidden_feature=opt.hidden)
    model.load_state_dict(torch.load(opt.model_dir))
    model.cuda()

    # test & save
    with torch.no_grad():
        te_l, preds = test_corr(test_loader, model, is_cuda=is_cuda)                                
        test_result = 'Test loss:{:.3f}\n'.format(te_l.item())

    end_time = time.time()
    time_comsuming = 'Time consuming:{:.2f}s \n'.format(end_time-start_time)        
    result_str =  (time_comsuming + train_result + test_result) if ('train_result' in vars()) else (time_comsuming + test_result)

    with open(opt.result_Preds_dir, 'wb') as f:
        pickle.dump({'loss': te_l, 'preds': preds}, f)
    
    print('\nStart evaluation')
    main_eval(date_time, opt, data_test,  separated=True)
    print('Evaluation done. Please see details in the ckeckpoint folder.\n')
    
    with open(opt.ckpt_tensorboard+'/args.txt', 'a') as f:
        f.write('\n'+date_time +' \n'+ result_str )

    print(result_str)

if __name__ == "__main__":
    torch.cuda.set_device(3)
    print('GPU Index: {}'.format(torch.cuda.current_device()))
    option = Options().parse()
    main(option)