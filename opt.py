import argparse
from pprint import pprint
import datetime, os

class Options:
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.opt = None

    def _initial(self):
        # ===============================================================
        #                     General options
        # ===============================================================
        self.parser.add_argument('--datetime', type=str, default="", help='datatime now')
        self.parser.add_argument('--ckpt', type=str, default='Combined_v1', help='path to checkpoint', required=True)
        self.parser.add_argument('--note', type=str, default="", help='any other notes')
       
        self.parser.add_argument('--result_CMT_dir', type=str, default='', help='path to save classification results')
        self.parser.add_argument('--result_Preds_dir', type=str, default='', help='path to save evaluation results')
        self.parser.add_argument('--result_EMT_dir', type=str, default='', help='path to save evaluation results')
        self.parser.add_argument('--model_dir', '-ccmd', type=str, default='Running_logs/model.pt',
                                 help='path to saved file')
        # self.parser.add_argument('--model', type=str, default='Combined_v2', help='model choose')
        # self.parser.add_argument("--combined", "-c", default=True, help='Choose the combined model')
        self.parser.add_argument('--raw_data_dir', type=str, default='data/EC3D/data_3D.pickle', help='path to source data of EC3D')
        self.parser.add_argument('--NTU_data_path', type=str, default='data/NTU/tu_uniformed.pickle', help='path to NTU data')
        self.parser.add_argument('--EC3D_data_path', type=str, default='data/EC3D/tmp_wo_val.pickle', help='path to dataset')

        # ===============================================================
        #                     Model & Running options
        #
        self.parser.add_argument('--dct_n', type=int, default=25, help='Number of DCT coefficients !invalid')
        self.parser.add_argument('--use_vel', type=int, default=0, help='Whether to use velocity.')
        self.parser.add_argument('--batch', type=int, default=32, help='Batch size')
        self.parser.add_argument('--hidden', type=int, default=256, help='Number of hidden features')
        self.parser.add_argument('--dropout', type=float, default=0.5, help='Dropout probability, 1 for none')
        self.parser.add_argument('--block', type=int, default=2, help='Number of GC blocks, valid in corrector') 
        self.parser.add_argument('--lr', type=float, default=0.01, help='Learning rate')
        self.parser.add_argument('--lr_decay', type=int, default=5, help='every lr_decay epoch do lr decay')
        self.parser.add_argument('--lr_gamma', type=float, default=0.9, help='decay coefficient')
        self.parser.add_argument('--epoch_corr_class', '-ecc', type=int, default=50, help='Number of epochs for correction and classification')
        self.parser.add_argument('--epoch_corr', type=int, default=50, help='Number of epochs for correction')
        self.parser.add_argument('--epoch_class', type=int, default=200, help='Number of epochs for classification')
        self.parser.add_argument('--beta', type=float, default=1)
        self.parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
        
    def _print(self):
        print("\n==================Options=================")
        pprint(vars(self.opt), indent=4)
        print("==========================================\n")

    def parse(self):
        self._initial()
        self.opt = self.parser.parse_args()
        self.opt.datetime = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")
        # do some pre-check
        ckpt = os.path.join('Running_logs', self.opt.ckpt)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self.opt.ckpt = ckpt  # i.e. Running_logs/ckpt
        self._print()
        return self.opt

def setup_folder(opt):
    """
        set dir: tensorboard, model, result 
        opt.ckpt_tensorbaord = Running_logs/opt.ckpt/tensorboard/date_time 
        opt.model_dir = Running_logs/opt.ckpt/models/date_time 
        opt.result_pickle_dir = Running_logs/opt.ckpt/result/date_time 
    """

    date_time= opt.datetime

    # tensorboard
    ckpt_tensorboard = opt.ckpt+'/tensorboard/'+date_time 
    while os.path.exists(ckpt_tensorboard):
        ckpt_tensorboard += "_x"
    os.makedirs(ckpt_tensorboard)  
    opt.ckpt_tensorboard = ckpt_tensorboard

    # model_directory
    model_dir = opt.ckpt+'/models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  
    model_datetime = model_dir + date_time
    while os.path.exists(model_datetime+'.pt'):
        model_datetime += "_x"
    opt.model_dir = model_datetime+".pt"

    """ result """
    result_dir =  opt.ckpt+'/result/'   
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    # confusion matrix of classifcation task
    result_CMT_datetime= result_dir + 'CMT-'+ date_time
    while os.path.exists(result_CMT_datetime + '.pickle'):
        result_CMT_datetime += "_x"
    opt.result_CMT_dir= result_CMT_datetime + '.pickle'

    # results of correction task
    result_Preds_datetime= result_dir + 'Preds-'+ date_time
    while os.path.exists(result_Preds_datetime + '.pickle'):
        result_Preds_datetime += "_x"
    opt.result_Preds_dir= result_Preds_datetime + '.pickle'
    
    # evaluation matrix of correction task
    result_EMT_datetime= result_dir + 'EMT-'+ date_time
    while os.path.exists(result_EMT_datetime):
        result_EMT_datetime += "_x"
    opt.result_EMT_dir= result_EMT_datetime 

    return date_time  

def save_opt(opt, writer):
    with open(opt.ckpt_tensorboard+'/args.txt', 'w') as f:
        my_str = ""
        for key, value in vars(opt).items():
            if not key == "note":
                my_str += str(key)+": "+str(value)+"\n"
            elif value != "":
                my_str += "\n********\nNOTE: "+str(value)+"\n"
        f.write(my_str)
        writer.add_text("Notes/", my_str)