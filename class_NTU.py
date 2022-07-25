import time, os, sys, pdb
from traceback import print_tb
import numpy as np
start_time = time.time()

from cgi import test
import torch 
import pickle
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

torch.cuda.set_device(1)
print('GPU Index: {}'.format(torch.cuda.current_device()))

sys.path.append("..")
from PoseCorrection.dataset import HV3D, NTU60
from PoseCorrection.model import GCN_class_22Spring, GCN_class_22Spring_vel, GCN_class_22Spring_ntu
from PoseCorrection.opt1 import Options, setup_folder, save_opt
from PoseCorrection.utils1 import *

def main(opt):
    torch.manual_seed(0)
    np.random.seed(0)

    # logging
    date_time = setup_folder(opt)
    writer_tr = SummaryWriter(opt.ckpt_tensorboard+'/tr')
    writer_val = SummaryWriter(opt.ckpt_tensorboard+'/val')
    writer_test = SummaryWriter(opt.ckpt_tensorboard+'/test')

    save_opt(opt, writer_tr)

    is_cuda = torch.cuda.is_available()    

    # dataset
    try:
        with open('Data-22Spring/ntu_tr_val_te.pickle', "rb") as f:
            data = pickle.load(f)
        data_train = data['train']
        data_val = data['val']
        data_test = data['test']
        print('Load preserved Dataset.')
    except FileNotFoundError:
        filepath = opt.NTU_data_pth
        data_train = NTU60(filepath=filepath, dct_n=opt.dct_n, split='train', is_cuda=is_cuda)
        data_val = NTU60(filepath=filepath, dct_n=opt.dct_n, split='validation', is_cuda=is_cuda)
        data_test = NTU60(filepath=filepath, dct_n=opt.dct_n, split='test', is_cuda=is_cuda)
        with open('Data-22Spring/ntu_tr_val_te_dct35.pickle', 'wb') as f:
            pickle.dump({'train': data_train, 'val': data_val, 'test': data_test}, f)
        print('Compile new Dataset.')

    train_loader = DataLoader(dataset=data_train, batch_size=opt.batch, shuffle=True, drop_last=True)
    val_loader = DataLoader(dataset=data_val, batch_size=len(data_val))
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))
    
    if opt.use_vel==1:
        model = GCN_class_22Spring_vel(hidden_feature=opt.hidden, input_feature=opt.dct_n, p_dropout=opt.dropout, dataset_name='NTU60')
    else:
        model = GCN_class_22Spring_ntu(num_block=opt.block, hidden_feature=opt.hidden, input_feature=opt.dct_n, p_dropout=opt.dropout, dataset_name='NTU60')

    if is_cuda: 
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr,weight_decay=opt.weight_decay)
    lr_now = opt.lr

    # train 
    print('Start training.')
    try:
        with tqdm(range(opt.epoch_class), desc=f'Training Classifier', unit="epoch") as tepoch:
            for epoch in tepoch:
                if (epoch + 1) % opt.lr_decay == 0:
                    lr_now = lr_decay(optimizer, lr_now, opt.lr_gamma)
                tr_l, tr_acc = train_class(train_loader, model, optimizer, is_cuda=is_cuda, level=1)
                writer_tr.add_scalar("training/acc", tr_acc, epoch) 
                writer_tr.add_scalar("training/loss", tr_l, epoch) 
                if (epoch + 1) % 5 == 0:
                    with torch.no_grad():
                        te_l, te_acc, _, _ = test_class(test_loader, model, is_cuda=is_cuda, level=2)
                        val_l, val_acc = evaluate_class(val_loader, model, is_cuda=True, level=2) 
                        #visualization
                        writer_val.add_scalar("val/acc", val_acc, epoch)
                        writer_val.add_scalar("val/loss", val_l, epoch) 
                        writer_test.add_scalar("test/acc", te_acc, epoch)  
                        writer_test.add_scalar("test/loss", te_l, epoch) 
                tepoch.set_postfix(train_loss=tr_l.item(), train_accuracy=tr_acc)
    except KeyboardInterrupt:
        print('KeyboardInterrupt: stop training')
    result_str = 'Training_accuracy:{:.3f}%, Training_loss:{:.3f}\n'.format(tr_acc,tr_l)
    writer_tr.close()
    writer_val.close()
    writer_test.close()
    # save model
    torch.save(model.state_dict(), opt.model_dir)

    # load model 
    if opt.use_vel==1:
        model = GCN_class_22Spring_vel(num_block=opt.block,hidden_feature=opt.hidden, input_feature=opt.dct_n, p_dropout=opt.dropout, dataset_name='NTU60')
    else:
        model = GCN_class_22Spring_ntu(num_block=opt.block, hidden_feature=opt.hidden, input_feature=opt.dct_n, p_dropout=opt.dropout, dataset_name='NTU60')

    model.load_state_dict(torch.load(opt.model_dir))
    
    # test
    print('Start testing.')
    if is_cuda:
        model.cuda()
    with torch.no_grad():
        test_l, test_acc, summary, cmt = test_class(test_loader, model, is_cuda=is_cuda, level=2)
    
    end_time = time.time()
    time_consuming = "Time consuming: {:.2f}seconds".format(end_time - start_time)
    
    result_str += 'Validate_accuracy:{:.3f}%, Validate_loss:{:.3f}\n'.format(val_acc,val_l)
    result_str += 'Test_accuracy:{:.3f}%, Test_loss:{:.3f}'.format(test_acc,test_l)
    
    print(result_str) 
    print("Time consuming: {:.2f}seconds".format(end_time - start_time))

    with open(opt.result_pickle_dir, 'wb') as f:
        pickle.dump({'loss': test_l, 'acc': test_acc, 'sum': summary, 'cmt': cmt}, f)
    
    # write results to txt
    with open('Results-22Spring/results_collection.txt', 'a') as f: 
        f.write(date_time +'||'+ 'Test_accuracy:{:.3f}%, Validate_accuracy:{:.3f}%n'.format(test_acc, val_acc))
        f.close()
    with open(opt.ckpt_tensorboard+'/args.txt', 'a') as f:
        f.write('\n'+date_time +' \n'+ result_str + '\n' + time_consuming)


if __name__ == "__main__":
    option = Options().parse()
    main(option)
