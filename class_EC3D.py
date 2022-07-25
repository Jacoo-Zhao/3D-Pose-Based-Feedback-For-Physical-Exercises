import time, os, sys, pdb
import pickle
from turtle import mode
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

sys.path.append("..")
from PoseCorrection.dataset import HV3D, NTU60
from PoseCorrection.model import GCN_class, GCN_class_22Spring
from PoseCorrection.opt1 import Options, setup_folder, save_opt
from PoseCorrection.utils1 import *

from torch.utils.tensorboard import SummaryWriter

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

    try:
        with open('Data/tmp_noVal.pickle', "rb") as f:
            data = pickle.load(f)
        data_train = data['train']
        data_test = data['test']
    except FileNotFoundError:
        # sets = [[0, 1], [2], [3]]
        # discard validation set
        sets = [[0,1,2],[3]] 
        data_train = HV3D(opt.gt_dir, sets=sets, split=0, is_cuda=is_cuda)
        data_test = HV3D(opt.gt_dir, sets=sets, split=1, is_cuda=is_cuda)
        with open('Data/tmp_noVal.pickle', 'wb') as f:
            pickle.dump({'train': data_train, 'test': data_test}, f)
    
    train_loader = DataLoader(dataset=data_train, batch_size=opt.batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))
    
    model = GCN_class_22Spring(hidden_feature=opt.hidden, p_dropout=opt.dropout, dataset_name='HV3D')
    if is_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    lr_now = opt.lr
    
    with tqdm(range(opt.epoch_class), desc=f'Training Classifier', unit="epoch") as tepoch:
        for epoch in tepoch:
            if (epoch + 1) % opt.lr_decay == 0:
                lr_now = lr_decay(optimizer, lr_now, opt.lr_gamma)
            tr_l, tr_acc = train_class(train_loader, model, optimizer, is_cuda=is_cuda, level=1)
             # visualization
            writer_tr.add_scalar("train/acc", tr_acc, epoch) 
            writer_tr.add_scalar("train/loss", tr_l, epoch) 
            with torch.no_grad():
                te_l, te_acc, _, _ = test_class(test_loader, model, is_cuda=is_cuda, level=1)
                #visualization
                writer_test.add_scalar("test/acc", te_acc, epoch)  
                writer_test.add_scalar("test/loss", te_l, epoch) 
            tepoch.set_postfix(train_loss=tr_l.item(), train_accuracy=tr_acc)

    result_str = 'Training_accuracy:{:.3f}%, Training_loss:{:.3f}\n'.format(tr_acc,tr_l)
    writer_tr.close()
    writer_test.close()
    
     # save model
    torch.save(model.state_dict(), opt.model_dir)

    model = GCN_class_22Spring(hidden_feature=opt.hidden, p_dropout=opt.dropout, dataset_name='HV3D')
    model.load_state_dict(torch.load(opt.model_dir))

    if is_cuda:
        model.cuda()

    with torch.no_grad():
        test_l, te_acc, summary, cmt = test_class(test_loader, model, is_cuda=is_cuda, level=1)

    end_time = time.time()
    time_consuming = "Time consuming: {:.2f}".format(end_time - start_time)
    result_str += 'Test_accuracy:{:.3f}%, Test_loss:{:.3f}'.format(te_acc,te_l)
    print(result_str) 
    print("Time consuming: {:.2f}seconds".format(end_time - start_time))

    with open(opt.result_pickle_dir, 'wb') as f:
        pickle.dump({'loss': test_l, 'acc': te_acc, 'sum': summary, 'cmt': cmt}, f)
        
    with open(opt.ckpt_tensorboard+'/args.txt', 'a') as f:
        f.write('\n'+date_time +' \n'+ result_str + '\n' + time_consuming)
    

if __name__ == "__main__":
    option = Options().parse()
    main(option)
