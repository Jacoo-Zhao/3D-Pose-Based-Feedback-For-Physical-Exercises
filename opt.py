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
        self.parser.add_argument('--model_dir', type=str, default='', help='path to save model.pt')
        self.parser.add_argument('--result_pickle_dir', type=str, default='', help='path to save result.pickle')
        self.parser.add_argument('--model', type=str, default='Combined_v2', help='model choose')

        self.parser.add_argument("--combined", "-c", default=True, help='Choose the combined model')
        self.parser.add_argument('--NTU_data_pth', type=str, default='/cvlabdata2/home/ziyi/3D-Motion-Correction-22SPring/Motion-Correction-master/PoseCorrection/Data-22Spring/ntu/3D_PC_output/ntu_uniformed.pickle', help='pth to NTU data')

        self.parser.add_argument('--vibe_dir', type=str, default='../data_3D_VIBE.pickle', help='path to dataset')
        self.parser.add_argument('--gt_dir', type=str, default='Data/data_3D.pickle', help='path to dataset')
        self.parser.add_argument('--corr_model_dir', type=str, default='Results-22Spring',
                                 help='path to saved file')
        self.parser.add_argument('--class_model_dir', type=str, default='Results1/model_class26-1.pt',
                                 help='path to saved file')
        self.parser.add_argument('--corr_class_model_dir', '-ccmd', type=str, default='Results/model_corr_class.pt',
                                 help='path to saved file')
       

        # ===============================================================
        #                     Model & Running options
        # ===============================================================
        self.parser.add_argument('--dct_n', type=int, default=25, help='Number of DCT coefficients !invalid')
        self.parser.add_argument('--use_vel', type=int, default=0, help='Whether to use vel.')
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
        ckpt = os.path.join('Results', self.opt.ckpt)
        if not os.path.isdir(ckpt):
            os.makedirs(ckpt)
        self.opt.ckpt = ckpt  # i.e. Results-22Spring/ckpt
        self._print()
        return self.opt

def setup_folder(opt):
    """
        set dir: tensorboard, model, result 
        opt.ckpt_tensorbaord = Results-22Spring/opt.ckpt/tensorboard/date_time 
        opt.model_dir = Results-22Spring/opt.ckpt/models/date_time 
        opt.result_pickle_dir = Results-22Spring/opt.ckpt/result_pickle/date_time 
    """

    date_time= opt.datetime

    # tensorboard
    ckpt_tensorboard = opt.ckpt+'/tensorboard/'+date_time 
    while os.path.exists(ckpt_tensorboard):
        ckpt_tensorboard += "_x"
    os.makedirs(ckpt_tensorboard)  
    opt.ckpt_tensorboard = ckpt_tensorboard

    # model
    model_dir = opt.ckpt+'/models/'
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)  
    model_datetime = model_dir + date_time
    while os.path.exists(model_datetime+'.pt'):
        model_datetime += "_x"
    opt.model_dir = model_datetime+".pt"

    #result_pickle
    result_dir =  opt.ckpt+'/result_pickle/'   
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    result_datetime= result_dir + date_time
    while os.path.exists(result_datetime + '.pickle'):
        result_datetime += "_x"
    opt.result_pickle_dir= result_datetime+'.pickle'

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