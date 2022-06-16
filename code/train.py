import argparse, sys
import collections
import torch
import numpy as np
import data_loader.data_loaders as module_data
# import data_loader.tumorgenerator_dataloader as module_data  # change when load new data
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch # change when load new model
from utils.parse_config import ConfigParser
from utils.trainer import Trainer
from utils.util import prepare_device


# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config, save_option = None, random_seed = None):
    logger = config.get_logger('train')
    task = config['name']


    # setup data_loader instances
    data_loader = config.init_obj('data_loader', module_data)
    valid_data_loader = data_loader.get_val_loader()
    if task == 'tumorsyn':
        # experiment, repeat train multiple models with the same training data
        # presaved: load presaved input for training; save: save the current training data; none: normal training
        saved_input_dir = "/project/labname-lab/authorid/trained_model/tumorsyn/shortcut/20211118_1432/"
        if save_option == "save":
            logger.info('Save training data')
            train_data_loader = data_loader.get_train_loader(save_inputs = saved_input_dir)
        elif save_option == "presaved":
            logger.info('Load presaved training data')
            train_data_loader = data_loader.get_val_loader_presaved(saved_input_dir, shuffle= False)
        else:
            train_data_loader = data_loader.get_train_loader()
    else:
        train_data_loader = data_loader.get_train_loader()
    # # setting the modality
    # modality = config["xai"]["modality"]
    # input_modality = config["data_loader"]["args"]['input_modality']
    # mods = [modality[i] for i in range(len(input_modality)) if input_modality[i]==1]
    # logger.info("Training on modality {}".format(mods))
    # modaltiy ablation experiment. Not applicable when fill in the ablated modality with noises.
    # for data in train_data_loader:
    #     assert data['image'].shape[1] == sum(input_modality), "ERROR! The data input modality doesn't match!"
    #     break


    # build model architecture, then print to console
    if random_seed: # experiment to train multiple model with different seeds
        logger.info("random_seed: {}".format(random_seed))
        np.random.seed(random_seed)
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
    model = config.init_obj('arch', module_arch)
    # print out model parameters to check seed
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    # model = config.init_obj('arch', densenet121)
    # logger.debug(model)

    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config['n_gpu'])
    model = model.to(device)
    if len(device_ids) > 1:
        model = torch.nn.DataParallel(model, device_ids=device_ids)

    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    scalar_metrics = [getattr(module_metric, met) for met in config['metrics']]
    non_scalar_metrics = [getattr(module_metric, met) for met in config['non_scalar_metrics']]
    metrics = (scalar_metrics, non_scalar_metrics)

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      device=device,
                      data_loader=train_data_loader,
                      valid_data_loader=valid_data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='XAI_MIA')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')
    args.add_argument('-s', '--save', default=None, type=str,
                      help='options to save training data/use the presaved training data')
    args.add_argument('-e', '--seed', default=None, type=str,
                      help='random seed to generate the ')
    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size'),
        CustomArgs(['--fold'], type=int, target='data_loader;args;fold')
    ]
    config = ConfigParser.from_args(args, options)
    save_option = None
    random_seed = None
    if '--save' in sys.argv:
        save_option = sys.argv[ sys.argv.index("--save") +1]
    if '--seed' in sys.argv:
        random_seed = sys.argv[ sys.argv.index("--seed") +1]
        random_seed = int(random_seed)
    main(config, save_option, random_seed)
