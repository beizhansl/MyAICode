from easydict import EasyDict as edict

default = edict()

default.snapshot_path = './snapshot/'
default.vis_path = './visulization/'
default.data_path = './data/'

config = edict()
# Hyper-parameters
config.multi_gpu = False

# Setting path
config.snapshot_path = default.snapshot_path
config.pretrained_path = default.snapshot_path
config.vis_path = default.vis_path
# config.data_path = default.data_path

# Setting training parameters
config.task_name = "train"
config.G_LR = 5e-4
config.D_LR = 5e-4 * 2
config.num_epochs = 200
config.num_epochs_decay = 100
config.ndis = 2
config.snapshot_step = 250
config.log_step = 10
config.vis_step = config.snapshot_step
config.batch_size = 1
config.checkpoint = ""


def merge_cfg_arg(config, args):
    config.batch_size = args.batch_size
    config.vis_step = args.vis_step
    config.snapshot_step = args.vis_step
    config.ndis = args.ndis
    config.G_LR = args.LR
    config.D_LR = args.LR * 2
    config.num_epochs_decay = args.decay
    config.num_epochs = args.epochs
    config.task_name = args.task_name
    config.data_path = args.data_path
    config.checkpoint = args.checkpoint
    return config

