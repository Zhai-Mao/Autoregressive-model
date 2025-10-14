from yacs.config import CfgNode as Node
cfg = Node()
cfg.data_path = 'data/data_to_predict.csv'
cfg.nrows = 9888
cfg.step = 10
cfg.usecols = 1
cfg.future_steps = 15

cfg.hid_size = 3
cfg.num_layers = 1

cfg.lr = 0.01
cfg.epochs = 500

# cfg.model = Node()
# cfg.model.base_outdim = 64
# cfg.model.k = 5
# cfg.model.drop_rate = 0.1
# cfg.model.layer = 20
#
# cfg.fit = Node()
# cfg.fit.lr = 0.008
# cfg.fit.max_epochs = 4000
# cfg.fit.patience = 1500
# cfg.fit.batch_size = 8192
# cfg.fit.virtual_batch_size = 256

# 解析命令行参数并更新配置
def update_cfg_from_args(cfg, args):
    if args.data_path:
        cfg.data_path = args.data_path
    if args.nrows:
        cfg.nrows = args.nrows
    if args.step:
        cfg.step = args.step
    if args.usecols:
        cfg.usecols = args.usecols
    if args.future_steps:
        cfg.future_steps = args.future_steps
    if args.hid_size:
        cfg.hid_size = args.hid_size
    if args.num_layers:
        cfg.num_layers = args.num_layers
    if args.lr:
        cfg.lr = args.lr
    if args.epochs:
        cfg.epochs = args.epochs
    return cfg