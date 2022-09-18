import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import sys, os
import torch
import numpy as np

from models import GCN_class
from opt import Options
from dataset import HV3D

from utils import lr_decay, train_class, test_class 
# from utils import *
from torch.utils.tensorboard import SummaryWriter
torch.cuda.set_device(2)
print('GPU Index: {}'.format(torch.cuda.current_device()))

def main(opt):
    time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    torch.manual_seed(0)
    np.random.seed(0) 

    is_cuda = torch.cuda.is_available()
    ckpt_tensorboard = 'Results-22Spring/tensorboard/'+time
    while os.path.exists(ckpt_tensorboard):
        ckpt_tensorboard += "_x"
    os.makedirs(ckpt_tensorboard)  
    opt.ckpt_tensorboard = ckpt_tensorboard
    writer_tr = SummaryWriter(opt.ckpt_tensorboard+'/tr')
    writer_test = SummaryWriter(opt.ckpt_tensorboard+'/test')

    print('Loading data...')
    try:
        with open('tmp.pickle', "rb") as f:
            data = pickle.load(f)
        data_train = data['train']
        data_test = data['test']
    except FileNotFoundError:
        sets = [[0, 1, 2], [], [3]]
        data_train = HV3D(opt.gt_dir, sets=sets, split=0, is_cuda=is_cuda)
        data_test = HV3D(opt.gt_dir, sets=sets, split=2, is_cuda=is_cuda)
        with open('Data/tmp.pickle', 'wb') as f:
            pickle.dump({'train': data_train, 'test': data_test}, f)

    train_loader = DataLoader(dataset=data_train, batch_size=opt.batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))

    model = GCN_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12)
    if is_cuda:
        model.cuda()

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    lr_now = opt.lr
    whether_train_model = input('Train model? Otherwise use pretrained model. (y/n) :')
    if whether_train_model == 'y':
        print('Start training...')
        with tqdm(range(int(opt.epoch_class)), desc=f'Training Classifier', unit="epoch") as tepoch:
            for epoch in tepoch:
                if (epoch + 1) % opt.lr_decay == 0:
                    lr_now = lr_decay(optimizer, lr_now, opt.lr_gamma)
                tr_l, tr_acc = train_class(train_loader, model, optimizer, is_cuda=is_cuda, level=1)
                writer_tr.add_scalar("train/acc", tr_acc, epoch) 
                writer_tr.add_scalar("train/loss", tr_l, epoch) 
                with torch.no_grad():
                    te_l, te_acc, _, _ = test_class(test_loader, model, is_cuda=is_cuda, level=1)
                    writer_test.add_scalar("test/acc", te_acc, epoch)  
                    writer_test.add_scalar("test/loss", te_l, epoch) 
                tepoch.set_postfix(train_loss=tr_l.item(), train_accuracy=tr_acc)

        torch.save(model.state_dict(), opt.class_model_dir)
        writer_tr.close()
        writer_test.close()
    
    print('Use pretrained model...')
    model = GCN_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12)
    model.load_state_dict(torch.load(opt.class_model_dir))

    if is_cuda:
        model.cuda()

    with torch.no_grad():
        test_l, te_acc, summary, cmt = test_class(test_loader, model, is_cuda=is_cuda, level=1)
    
    with open('Results-22Spring/CMT-'+time+'.pickle', 'wb') as f:
        pickle.dump({'loss': test_l, 'acc': te_acc, 'sum': summary, 'cmt': cmt}, f)
    print('Time:{}  Test_loss:{:.2f}  Test_acc:{:.2f}%'.format(time, float(test_l), (te_acc)))

if __name__ == "__main__":
    option = Options().parse()
    main(option)
