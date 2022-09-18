import pickle, os, time, pdb
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np



from torch.utils.tensorboard import SummaryWriter
from dataset import HV3D
# from model_combined_v1 import *
# from model_combined_v2 import *
from opt import Options, setup_folder, save_opt
from models import GCN_corr_class_v1_22Spring, GCN_class, GCN_corr_ori
from utils import lr_decay, train_class, test_class, train_corr, test_corr, test_corr_v1, test_class_v1,train_corr_class
from evaluation import main_eval


torch.cuda.set_device(2)
print('GPU Index: {}'.format(torch.cuda.current_device()))

def main(opt,version_name):
    start_time = time.time()    
    torch.manual_seed(0)
    np.random.seed(0)

    # logging
    date_time = setup_folder(opt)
    writer_tr = SummaryWriter(opt.ckpt_tensorboard+'/tr')
    writer_test = SummaryWriter(opt.ckpt_tensorboard+'/test')
    save_opt(opt, writer_tr)
    is_cuda = torch.cuda.is_available()
    print('Loading data...')
    try:
        with open('Data/tmp_noVal.pickle', "rb") as f:
            data = pickle.load(f)
        data_train = data['train']
        data_test = data['test']
    except FileNotFoundError:
        sets = [[0, 1], [2], [3]]
        data_train = HV3D(opt.gt_dir, sets=sets, split=0, is_cuda=is_cuda)
        data_evaluation = HV3D(opt.gt_dir, sets=sets, split=1, is_cuda=is_cuda)
        data_test = HV3D(opt.gt_dir, sets=sets, split=2, is_cuda=is_cuda)
        with open('Data/tmp.pickle', 'wb') as f:
            pickle.dump({'train': data_train, 'evaluation': data_evaluation, 'test': data_test}, f)

    train_loader = DataLoader(dataset=data_train, batch_size=opt.batch, shuffle=True, drop_last=True)
    test_loader = DataLoader(dataset=data_test, batch_size=len(data_test))
    beta = opt.beta

    whether_train_model = 'y'
    if whether_train_model == 'y':
        if opt.combined == True:
            model = GCN_corr_class_v1_22Spring(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda()
            optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
            print('Model selected: GCN_corr_class__v1_22Spring')
        else:
            model_corr = GCN_corr_ori(hidden_feature=opt.hidden, p_dropout=opt.dropout).cuda()
            model_clas = GCN_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda()
            optimizer_corr = torch.optim.Adam(model_corr.parameters(), lr=opt.lr)
            optimizer_clas = torch.optim.Adam(model_clas.parameters(), lr=opt.lr)
            print('We use the seperated model here.')

        lr_now = opt.lr
        lr_now_corr = opt.lr
        lr_now_class = opt.lr

        # Stop training after detecting keyboard input
        try:
            with tqdm(range(opt.epoch_corr_class), desc=f'Training model', unit="epoch") as tepoch:
                for epoch in tepoch:
                    if opt.combined == True:
                        if (epoch + 1) % opt.lr_decay == 0:
                            lr_now = lr_decay(optimizer, lr_now, opt.lr_gamma)
                        tr_l, tr_acc, loss_corr= train_corr_class(train_loader, model, optimizer, opt.beta, is_cuda=is_cuda, level=1)
                    else:
                        if (epoch + 1) % opt.lr_decay == 0:
                            lr_now_corr = lr_decay(optimizer_corr, lr_now_corr, opt.lr_gamma)
                            lr_now_class = lr_decay(optimizer_clas, lr_now_class, opt.lr_gamma)
                        tr_l = train_corr(train_loader, model_corr, optimizer_corr, is_cuda=is_cuda)
                        _, tr_acc = train_class(train_loader, model_clas, optimizer_clas, is_cuda=is_cuda, level=1)
                    
                    writer_tr.add_scalar('loss/corr_train', loss_corr, epoch)
                    writer_tr.add_scalar("train/loss", tr_l, epoch) 
                    writer_tr.add_scalar("train/acc", tr_acc, epoch)  
                    
                    with torch.no_grad():
                        test_l_corr, preds = test_corr_v1(test_loader, model, is_cuda=is_cuda)
                        test_l_class, te_acc, _, _ = test_class_v1(test_loader, model, is_cuda=is_cuda, level=1)
                        
                        writer_test.add_scalar('loss/corr_test', test_l_corr, epoch)  
                        writer_test.add_scalar("test/acc", te_acc, epoch)
                        writer_test.add_scalar("test/loss", test_l_class, epoch) 
                    tepoch.set_postfix(train_loss=tr_l.item(), train_accuracy=tr_acc)
        except KeyboardInterrupt:
            print('KeyboardInterrupt: stop training')
        writer_tr.close()
        writer_test.close()

        # save model
        if opt.combined == True:
             torch.save(model.state_dict(), opt.model_dir)
        else:
            opt.corr_model_dir = 'Results1/' + version_name + 'corr.pt'
            opt.class_model_dir = 'Results1/' + version_name + 'class.pt'
            torch.save(model_corr.state_dict(), opt.corr_model_dir)
            torch.save(model_clas.state_dict(), opt.class_model_dir)

    else:
        print('Start to use the pre-trained model directly')

    if opt.combined == True:
        model = GCN_corr_class_v1_22Spring(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda()
        model.load_state_dict(torch.load(opt.model_dir))
    else:
        model_corr = GCN_corr_ori(hidden_feature=opt.hidden, p_dropout=opt.dropout).cuda()
        model_corr.load_state_dict(torch.load(opt.corr_model_dir))
        model_clas = GCN_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda()
        model_clas.load_state_dict(torch.load(opt.class_model_dir))

    with torch.no_grad():
        if opt.combined == True:
            test_l_corr, preds = test_corr_v1(test_loader, model, is_cuda=is_cuda)
            test_l_clas, te_acc, summary, cmt = test_class_v1(test_loader, model, is_cuda=is_cuda, level=1)
        else:
            test_l_corr, preds = test_corr(test_loader, model_corr, is_cuda=is_cuda)
            test_l_clas, te_acc, summary, cmt = test_class(test_loader, model_clas, is_cuda=is_cuda, level=1)
    test_l = test_l_clas + beta * test_l_corr


    result_data = [{'train_l(class+corr)': tr_l, 'train_acc(class)':tr_acc,
                  'test_l_corr':test_l_corr, 'test_l_clas':test_l_clas, 'test_l':test_l,
                  'test_acc(classifier)': te_acc}]

    with open(opt.result_pickle_dir, 'wb') as f:
        pickle.dump({ 'preds': preds, 'sum': summary, 'cmt': cmt}, f)
    
    end_time = time.time()
    time_consuming = "Time consuming: {:.2f}".format(end_time - start_time)

    print("Time consuming: {:.2f}seconds, Test_acc(classifier): {:.3f}".format(end_time - start_time, te_acc)) 

    with open(opt.ckpt_tensorboard+'/args.txt', 'a') as f:
        f.write('\n'+date_time + '\n' + time_consuming+'\n')
        f.write('Train accuracy:{:.3f}%, Test accuracy:{:.3f}\n'.format(tr_acc,te_acc))
        f.write(str(result_data))

    # Evaluation of the Corrector
    savepath = option.ckpt+'/Evaluation/'
    if not os.path.exists(savepath):
        os.makedirs(savepath)
    savepath += date_time
    while os.path.exists(savepath):
        savepath += "_x"
    main_eval(savepath, model_version='CC22S', model_combined=model, model_combined_path=opt.model_dir, options=opt)
    print('Evaluation done.')

if __name__ == "__main__":
    option = Options().parse()
    main(option,version_name='')
