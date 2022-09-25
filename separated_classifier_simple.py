import pickle
from torch.utils.data import DataLoader
from tqdm import tqdm
import datetime
import os
import torch
import numpy as np

from models import GCN_class
from opt import Options, setup_folder, save_opt
from dataset import EC3D
import sys
# sys.path.append("..")
# from utils import *
from utils import lr_decay, train_class, test_class 
from torch.utils.tensorboard import SummaryWriter
torch.cuda.set_device(3)
print('GPU Index: {}'.format(torch.cuda.current_device()))

def main(opt):
    time =  setup_folder(opt)
    # time = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
    torch.manual_seed(0)
    np.random.seed(0) 
    is_cuda = torch.cuda.is_available()
    # date_time = setup_folder(opt)

    writer_tr = SummaryWriter(opt.ckpt_tensorboard+'/train')
    writer_test = SummaryWriter(opt.ckpt_tensorboard+'/test')
    save_opt(opt, writer_tr)
    print('Loading data...')
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


    model = GCN_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12)    
    if is_cuda:
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    lr_now = opt.lr

    whether_train_model = input('Train model? Otherwise use pretrained model?\ny/n:')
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
        writer_tr.close()
        writer_test.close()

        torch.save(model.state_dict(), opt.model_dir)
        model = GCN_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12)
        model.load_state_dict(torch.load(opt.model_dir))
    else:
        print('Use pretrained model...')
        opt.model_dir = 'pretrained_weights/Separated_Classifier(simple).pt' 
        model = GCN_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12)
        model.load_state_dict(torch.load(opt.model_dir))


    if is_cuda:
        model.cuda()

    # Tesing
    with torch.no_grad():
        test_l, te_acc, summary, cmt = test_class(test_loader, model, is_cuda=is_cuda, level=1)
    
    # Saving results
    # with open(opt.result_dir+'/CMT-'+time+'.pickle', 'wb') as f:
    with open(opt.result_CMT_dir, 'wb') as f:
       pickle.dump({'loss': test_l, 'acc': te_acc, 'sum': summary, 'cmt': cmt}, f)
    print('Time:{}  Test_loss:{:.2f}  Test_acc:{:.2f}%'.format(time, float(test_l), (te_acc)))

if __name__ == "__main__":
    option = Options().parse()
    main(option)
