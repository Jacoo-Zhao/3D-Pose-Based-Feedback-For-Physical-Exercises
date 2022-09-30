import time
import torch
import pickle
import torch
import numpy as np
from tqdm import tqdm
from dataset import EC3D
from torch.utils.data import DataLoader

from torch.utils.tensorboard import SummaryWriter

from opt import Options, setup_folder, save_opt
from models import GCN_corr_class_ours
from utils import lr_decay, test_class, test_corr, train_corr_class_v4, test_class_v4, test_corr_v4
from evaluation import main_eval


def main(opt,version_name):
    start_time = time.time()    
    torch.manual_seed(0)
    np.random.seed(0)

    # logging
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
    beta = opt.beta
    
    model = GCN_corr_class_ours(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda()
    print('Model selected: Ours')

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    lr_now = opt.lr

    whether_train_model = input('Do you wanna start to train the model?(y/n)')
    if whether_train_model == 'y':
        print('Start training...\n')

        # Stop training when detecting keyboard interruption
        try:
            with tqdm(range(opt.epoch_corr_class), desc=f'Training model', unit="epoch") as tepoch:
                for epoch in tepoch:
                    if (epoch + 1) % opt.lr_decay == 0:
                        lr_now = lr_decay(optimizer, lr_now, opt.lr_gamma)
                    curriculum_learning_rate = 1 - epoch/opt.epoch_corr_class
                    tr_l, tr_acc, loss_corr= train_corr_class_v4(train_loader, model, curriculum_learning_rate, optimizer, opt.beta, is_cuda=is_cuda, level=1)
                    
                    writer_tr.add_scalar('loss/corr_train', loss_corr, epoch)
                    writer_tr.add_scalar("train/loss", tr_l, epoch) 
                    writer_tr.add_scalar("train/acc", tr_acc, epoch)  
                    
                    with torch.no_grad():
                        test_l_corr, preds, _ = test_corr_v4(test_loader, model, is_cuda=is_cuda, Use_label=False)
                        test_l_class, te_acc, _, _ = test_class_v4(test_loader, model, is_cuda=is_cuda, level=1, Use_label=False)
                        
                        writer_test.add_scalar('loss/corr_test', test_l_corr, epoch)  
                        writer_test.add_scalar("test/acc", te_acc, epoch)
                        writer_test.add_scalar("test/loss_classifier", test_l_class, epoch) 
                   
                    tepoch.set_postfix(train_loss=tr_l.item(), train_accuracy=tr_acc)
        except KeyboardInterrupt:
            print('KeyboardInterrupt: stop training')
        
        writer_tr.close()
        writer_test.close()

        # save 
        torch.save(model.state_dict(), opt.model_dir)
    else:
        print('Use the pre-trained model.')
        opt.model_dir = 'pretrained_weights/Ours.pt'

    # test accuracy    
    model = GCN_corr_class_ours(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda()
    model.load_state_dict(torch.load(opt.model_dir))
    model.cuda()

    with torch.no_grad():
        test_l_corr, preds, _ = test_corr_v4(test_loader, model, is_cuda=is_cuda, Use_label=False)
        test_l_class, te_acc, summary, cmt = test_class_v4(test_loader, model, is_cuda=is_cuda, level=1, Use_label=False)  
    
    test_l = test_l_class + beta * test_l_corr

    result_data = [{'train_loss(class+corr)': tr_l, 'train_acc(class)':tr_acc,
                'test_loss_corr':test_l_corr, 'test_loss_class':test_l_class, 'test_loss':test_l,
                'test_acc(classifier)': te_acc}] if ('tr_l' in vars()) else [{
                  'test_loss_corr':test_l_corr, 'test_loss_class':test_l_class, 'test_loss':test_l,
                  'test_acc(classifier)': te_acc}]
    
    with open(opt.result_CMT_dir, 'wb') as f:
       pickle.dump({'loss': test_l_class, 'acc': te_acc, 'sum': summary, 'cmt': cmt}, f)
    with open(opt.result_Preds_dir, 'wb') as f:
        pickle.dump({'loss': test_l_corr, 'preds': preds}, f)

    end_time = time.time()
    time_consuming = "Time consuming: {:.2f}".format(end_time - start_time)

    print('\nStart evaluation')
    main_eval(date_time, opt, data_test, model_version='Ours')
    print('Evaluation done. Please see details in the ckeckpoint folder.\n')

    with open(opt.ckpt_tensorboard+'/args.txt', 'a') as f:
        f.write('\n'+date_time + '\n' + time_consuming+'\n')
        if whether_train_model == 'y':
            f.write('Train accuracy:{:.3f}%, Test accuracy:{:.3f}\n'.format(tr_acc,te_acc))
        else:
            f.write('Test accuracy:{:.3f}\n'.format(te_acc))
        f.write(str(result_data))

    print("Time consuming: {:.2f}seconds, Test_acc(classifier): {:.3f}%\nRefer to the results folder and find evaluation information.".format(end_time - start_time, te_acc)) 

if __name__ == "__main__":
    torch.cuda.set_device(2)
    print('GPU Index: {}'.format(torch.cuda.current_device()))
    option = Options().parse()
    main(option,version_name='')
