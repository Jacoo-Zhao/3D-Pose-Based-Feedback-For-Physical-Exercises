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
from models import GCN_class_simple, GCN_class, GCN_corr, GCN_corr_class, GCN_corr_class_ours
# from utils import lr_decay, train_class, test_class, train_corr, test_corr, train_corr_class, test_class_v1, test_corr_v1, train_corr_class_v4, test_class_v4, test_corr_v4 
from utils import *
from evaluation import main_eval


def main(opt, model_version):
    print('Model Used: '+ model_version)
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

    lr_now = opt.lr
    models ={'Separated_Classifier_Simple': GCN_class_simple(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda(),                    
             'Separated_Classifier': GCN_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, dataset_name='EC3D').cuda(),
             'Separated_Corrector': GCN_corr( hidden_feature=opt.hidden,  p_dropout=opt.dropout).cuda(),
             'Combined_wo_Feedback': GCN_corr_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda(),
             'Ours': GCN_corr_class_ours(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda() 
            }

    model = models[model_version]
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    curriculum_learning_rate = 1 

    if is_cuda:
        model.cuda()
    
    
    model_dic = {'Separated_Classifier_Simple': 
                    {'model': GCN_class_simple(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12),  
                     'path': 'pretrained_weights/Classifier(simple).pt', 
                     'train': train_class(train_loader, models['Separated_Classifier_Simple'], optimizer, is_cuda=is_cuda, level=1), 
                     'test': test_class(test_loader,  models['Separated_Classifier_Simple'], is_cuda=is_cuda, level=1)},
                 'Separated_Classifier': 
                    {'model': GCN_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, dataset_name='EC3D').cuda(),  
                     'path': 'pretrained_weights/Classifer.pt',
                     'train': train_class(train_loader,  models['Separated_Classifier'], optimizer, is_cuda=is_cuda, level=1), 
                     'test': test_class(test_loader,  models['Separated_Classifier'], is_cuda=is_cuda, level=1)},
                 'Separated_Corrector': 
                    {'model': GCN_corr( hidden_feature=opt.hidden,  p_dropout=opt.dropout).cuda(),  
                     'path': 'pretrained_weights/Corrector.pt',
                     'train': train_corr(train_loader,  models['Separated_Corrector'], optimizer, is_cuda=is_cuda),
                     'test': test_corr(test_loader,  models['Separated_Corrector'], is_cuda=is_cuda)},
                 'Combined_wo_Feedback': 
                    {'model': GCN_corr_class(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda(),
                     'path': 'pretrained_weights/Combined_wo_Feedback.pt',
                     'train': train_corr_class(train_loader, models['Combined_wo_Feedback'], optimizer, opt.beta, is_cuda=is_cuda, level=1),
                     'test_corr': test_corr_v1(test_loader, models['Combined_wo_Feedback'], is_cuda=is_cuda),
                     'test_class': test_class_v1(test_loader,  models['Combined_wo_Feedback'], is_cuda=is_cuda, level=1)},
                 'Ours': 
                    {'mode': GCN_corr_class_ours(hidden_feature=opt.hidden, p_dropout=opt.dropout, classes=12).cuda(), 
                     'path': 'pretrained_weights/Ours.pt', 
                     'train': train_corr_class_v4(train_loader, models['Ours'], curriculum_learning_rate, optimizer, opt.beta, is_cuda=is_cuda, level=1), 
                     'test_corr': test_corr_v4(test_loader,  models['Ours'], is_cuda=is_cuda, Use_label=False), 
                     'test_class': test_class_v4(test_loader,  models['Ours'], is_cuda=is_cuda, level=1, Use_label=False)}
                }

    whether_train_model = input('Do you wanna start to train the model?(y/n)')
    if whether_train_model == 'y':
        print('Start training...\n')
        try:   # Stop training when detecting keyboard interruption
            with tqdm(range(opt.epoch), desc=f'Training model', unit="epoch") as tepoch:
                for epoch in tepoch:
                    if (epoch + 1) % opt.lr_decay == 0:
                        lr_now = lr_decay(optimizer, lr_now, opt.lr_gamma)
                        
                    if model_version[0] != 'S':
                        curriculum_learning_rate = 1 - epoch/opt.epoch
                        tr_l, tr_acc, loss_corr= model_dic[model_version]['train']
                        writer_tr.add_scalar('loss/corr_train', loss_corr, epoch)
                    elif model_version[11] == 'l':
                        tr_l, tr_acc = model_dic[model_version]['train']
                        train_result = 'Training_accuracy:{:.3f}%, Training_loss:{:.3f}'.format(tr_acc, tr_l)
                    else: 
                        tr_l = model_dic[model_version]['train']
                        train_result = 'Training_loss:{:.3f}'.format(tr_l.item())

                    writer_tr.add_scalar("train/loss", tr_l, epoch) 
                    if model_version[-2] != 'o': 
                        writer_tr.add_scalar("train/acc", tr_acc, epoch) 
                    
                    with torch.no_grad():
                        if model_version[0] != 'S':
                            test_l_corr, preds, _ = model_dic[model_version]['test_corr']
                            test_l_class, te_acc, _, _ = model_dic[model_version]['test_class']
                            writer_test.add_scalar('loss/corr_test', test_l_corr, epoch)  
                            writer_test.add_scalar("test/loss_classifier", test_l_class, epoch) 
                            writer_test.add_scalar("test/acc", te_acc, epoch)
                        elif model_version[11] == 'l':
                            te_l, te_acc, _, _ = test_class(test_loader, model, is_cuda=is_cuda, level=1)
                            writer_test.add_scalar("test/acc", te_acc, epoch)  
                            writer_test.add_scalar("test/loss", te_l, epoch) 
                        else: 
                            te_l, preds = test_corr(test_loader, model, is_cuda=is_cuda)                                
                            writer_test.add_scalar('loss/corr_test', te_l, epoch)    
                   
                    tepoch.set_postfix(train_loss=tr_l.item())
        except KeyboardInterrupt:
            print('KeyboardInterrupt: stop training')
        
        writer_tr.close()
        writer_test.close()
        torch.save(model.state_dict(), opt.model_dir)
    else:
        print('Use the pre-trained model.')
        opt.model_dir = model_dic[model_version]['path']

    # Test  
    model = models[model_version]
    model.load_state_dict(torch.load(opt.model_dir))
    model.cuda()

    with torch.no_grad():
        if model_version[0] != 'S':
            test_l_corr, preds, _ = model_dic[model_version]['test_corr']
            test_l_class, te_acc, summary, cmt = model_dic[model_version]['test_class']   
        elif model_version[11] == 'l':
            te_l, te_acc, _, _ = test_class(test_loader, model, is_cuda=is_cuda, level=1)
            te_l, te_acc, summary, cmt = test_class(test_loader, model, is_cuda=is_cuda, level=1)

            test_result = '    Test accuracy:{:.3f}%, Test loss:{:.3f}\n'.format(te_acc, te_l)        
        
        else: 
            te_l, preds = test_corr(test_loader, model, is_cuda=is_cuda)                                  
    
    ''' Saving '''
    end_time = time.time()
    time_consuming = "Time consuming: {:.2f}".format(end_time - start_time)
    
    if model_version[0] != 'S':
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

    elif model_version[11] == 'l':
        test_result = '    Test accuracy:{:.3f}%, Test loss:{:.3f}\n'.format(te_acc, te_l)
        result_str =  (time_consuming + train_result + test_result) if ('train_result' in vars()) else (time_consuming + test_result)
        with open(opt.result_CMT_dir, 'wb') as f:
            pickle.dump({'loss': te_l, 'acc': te_acc, 'sum': summary, 'cmt': cmt}, f)
        with open(opt.ckpt_tensorboard+'/args.txt', 'a') as f:
            f.write('\n'+date_time +' \n'+ result_str )
    else:
        test_result = 'Test loss:{:.3f}\n'.format(te_l.item())
        result_str =  (time_consuming + train_result + test_result) if ('train_result' in vars()) else (time_consuming + test_result)
        with open(opt.result_Preds_dir, 'wb') as f:
            pickle.dump({'loss': te_l, 'preds': preds}, f)
        with open(opt.ckpt_tensorboard+'/args.txt', 'a') as f:
            f.write('\n'+date_time +' \n'+ result_str )

    ''' Evaluation '''
    print('\nStart evaluation')
    if model_version in ['Combined_wo_Feedback', 'Ours']:
        main_eval(date_time, opt, data_test, model_version=model_version)
    elif model_version == 'Separated_Corrector':
        main_eval(date_time, opt, data_test, separated=True)
    print('Evaluation done. Please see details in the ckeckpoint folder.\n')

    
    ''' Printing '''
    with open(opt.ckpt_tensorboard+'/args.txt', 'a') as f:
        if model_version[0] == 'S':
            f.write('\n'+date_time +' \n'+ result_str )
            print(result_str)
        else:
            f.write('\n' + date_time + '\n' + time_consuming+'\n')
            if whether_train_model == 'y':
                f.write('Train accuracy:{:.3f}%, Test accuracy:{:.3f}\n'.format(tr_acc,te_acc))
            else:
                f.write('Test accuracy:{:.3f}\n'.format(te_acc))
            f.write(str(result_data))
            print("Time consuming: {:.2f}seconds, Test_acc(classifier): {:.3f}%\nRefer to the results folder and find evaluation information.".format(end_time - start_time, te_acc)) 


if __name__ == "__main__":
    torch.cuda.set_device(3)
    print('GPU Index: {}'.format(torch.cuda.current_device()))
    
    model_options = ['Separated_Classifier_Simple', 'Separated_Classifier', 'Separated_Corrector', 'Combined_wo_Feedback', 'Ours']
    print('Model Options:\t' + "   ".join(str(x) for x in model_options))

    while True:
        model_version = input('Input the model version you would like to use from the options: ') 
        if( model_version in model_options):
            break;      
        print('Please input the right model name!')
    option = Options().parse()
    main(option, model_version)