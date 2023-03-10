import torch
import argparse
import sys
from datetime import datetime
from functions.utils import GPUManager

def report_args(obj):
    print('------------ Options -------------')
    for key, val in vars(obj).items():
        print("--{:24} {}".format(key, val))
    print('-------------- End ----------------')

class Log(object):
    '''Save log file of the current experiment display'''
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'w')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

    def close(self):
        self.log.close()

opt = None

class Options(object):
    def __init__(self):
        parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

        parser.add_argument('--stage',default=1,type=int)
        parser.add_argument('--detector_type',default='1-interpret-2-stage-MixAug',type=str)

        # experiment specific options
        parser.add_argument('--exp_name', default='None', type=str, help='experiment name')
        parser.add_argument('--gpu_id', default='-1', type=str, help='use which gpu (-1: select automatically')
        parser.add_argument('--function', default='attack', type=str, help='function')

        parser.add_argument('--dataset', default='cifar10', help='use which dataset (mnist, cifar10, imagenet)')
        parser.add_argument('--attack',
                            default='FGSM-U',help='attack method (FGSM,  CW, PGD, DFool, DDN)')

        parser.add_argument('--interpret_method', default='VG', help='saliency method (VG, GBP, GCAM)')
        parser.add_argument('--howtousedata', default=0, type=int, choices=(0,1,2), help='0:using images only; 1:using gradients only; 2:using both images and gradients')

        parser.add_argument('--classifier_net', default='vgg11bn', type=str, help='cnn network structure for classifier (default as vgg11bn)')
        parser.add_argument('--classifier_classes', default=10, type=int, help='the number of classes of classifier\'s output ')
        parser.add_argument('--detector_net', default='vgg11bn', type=str, help='cnn network structure for detector (default as vgg11bn)')
        parser.add_argument('--detector_classes', default=2, type=int, help='the number of classes of detctor\'s output ')
        parser.add_argument('--rectifier_net', default='vgg11bn', type=str, help='cnn network structure for rectifier (default as vgg11bn)')

        # train model options
        parser.add_argument('--detector_mode', default='train', type=str, help='mode of running detectors')
        parser.add_argument('--num_epoches', default=30, type=int, help='train epoches')
        parser.add_argument('--lr', default=0.001, type=float, help='learning rate of training phase')
        parser.add_argument('--lr_mom', default=0.9, type=float, help='factor momentum of learning rate')
        parser.add_argument('--lr_step', default=5, type=int, help='decay lr by a factor of lr_gama every lr_step epoches')
        parser.add_argument('--lr_gama', default=0.5, type=float, help='decay lr by a factor of lr_gama')

        # attack method options
        parser.add_argument('--cw_lr', default=0.01, type=float)
        parser.add_argument('--cw_max_iterations', default=100, type=int)
        parser.add_argument('--cw_confidence', default=0.01, type=float)
        parser.add_argument('--fgsm_epsilons', nargs='+', default=[0.031], type=float)
        parser.add_argument('--fgsm_train_max', default=10001, type=int, help='max size of train set for fgsm maps per epsilon')
        parser.add_argument('--fgsm_test_max', default=30001, type=int, help='max size of test set for fgsm maps per epsilon')
        parser.add_argument('--dfool_max_iterations', default=100, type=int)
        parser.add_argument('--dfool_overshoot', default=0.02, type=float)
        parser.add_argument('--dnn_steps', default=100, type=int)
        parser.add_argument('--pgd_eps', default=0.031, type=float)
        parser.add_argument('--pgd_iterations', default=20, type=int)
        parser.add_argument('--pgd_eps_iter', default=0.0078125, type=float)



        # basic options
        parser.add_argument('--train_batchsize', default=256, type=int, help='trian batch size')
        parser.add_argument('--val_batchsize', default=256, type=int, help='val or test batch size')
        parser.add_argument('--workers', default=0, type=int, help='num of threads to load data')
        parser.add_argument('--image_channels', default=3, type=int, help='cifar10 and imagenet use 3; mnist uses 1')
        parser.add_argument('--classifier_root', default='../classifier_pth/', type=str, help='root for saving classifiers')
        parser.add_argument('--detector_root', default='../detector_pth/', type=str, help='root for saving classifiers')
        parser.add_argument('--rectifier_root', default='../rectifier_pth/', type=str, help='root for saving rectifiers')
        parser.add_argument('--data_root', default='../data/', type=str, help='dataset root')
        parser.add_argument('--adversarial_root', default='../adversarial_data/', type=str)
        parser.add_argument('--saliency_root', default='../saliency_image/', type=str)
        parser.add_argument('--masked_image_root', default='../masked_image/', type=str)
        parser.add_argument('--experiment_root', default='../experiment_log/', type=str, help='experiment log files root')


        parser.add_argument('--loss_type',default='cdr',type=str)
        parser.add_argument('--attack_interpret_method',default='all',type=str)
        parser.add_argument('--forest_input',default='logit',type=str)
        parser.add_argument('--attack_box', default='grey', type=str)
        parser.add_argument("--unvaccinated",action="store_true")
        parser.add_argument('--see_image_flag',default=False,type=bool)
        parser.add_argument('--e2eonly',default=0,type=int)
        
        parser.add_argument('--reducer',default="PCA_Reducer",type=str)

        self.parser = parser
        self.device = ''

    # must call parse_arguments to initialize at the first time
    def parse_arguments(self, print_=False):
        args = self.parser.parse_args()
        if args.dataset == 'mnist':
            args.image_channels = 1


        # Create log files
        timeStamp = datetime.now()
        formatTime = timeStamp.strftime("%m-%d %H-%M-%S")
        sys.stdout = Log(args.experiment_root + args.exp_name + '({}).log'.format(formatTime), sys.stdout)
        # sys.stderr = Logger('./experiments/' + args.exp_name + '_error({}).log'.format(formatTime), sys.stderr)

        if print_:
            print('------------ Options -------------')
            for key, val in sorted(vars(args).items()):
                print("--{:24} {}".format(key, val))
            print('-------------- End ----------------')

        # Set device
        use_cuda = True
        print("CUDA Available: ", torch.cuda.is_available())
        if use_cuda and torch.cuda.is_available():
            if args.gpu_id == '-1':
                # auto_choose gpu number
                gm = GPUManager()
                gpu_id = gm.auto_choice()
            else:
                gpu_id = args.gpu_id
                print("Using GPU : {}".format(gpu_id))
            self.device = torch.device("cuda:{}".format(gpu_id))
        else :
            self.device = torch.device("cpu")
        args.device =self.device
        global opt
        opt = args
        return args

def get_opt():
    return opt