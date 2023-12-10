import json
import os
from argparse import ArgumentParser

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from trainer_ae import TrainerAE

import utils.losses as loss_functions
from data.data_manager import DataManager
from networks.StageDiscriminator import StageDiscriminator
from trainer import Trainer
from utils.data_logs import save_logs_about
from vit_pytorch.cytran import CyTran
from vit_pytorch.cytran_masked_ae import CyTranAE


def main():
    config = json.load(open('./config.json'))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        os.makedirs(os.path.join(config['exp_path'], config['exp_name']))
    except FileExistsError:
        print("Director already exists! It will be overwritten!")

    logs_writer = SummaryWriter(os.path.join('runs', config['exp_name']))

    model = CyTran(input_nc=9, output_nc=1, n_downsampling=4, depth=5, heads=5).to(config['device'])
    if not os.path.exists("./checkpoints/latest_checkpoint_ae.pkl"):
        raise Exception("The pretrained model does not exist in the checkpoints directory."
                        "Most likely you forgot to run the script in autoencoder mode.")
    weights = torch.load("./checkpoints/latest_checkpoint_ae_classic_pretraining.pkl")['model_weights']
    weights.pop('small.3.weight')
    weights.pop('small.3.bias')
    weights.pop('high.0.weight')
    weights.pop('high.0.bias')
    weights.pop('medium.0.weight')
    weights.pop('medium.0.bias')
    model.load_state_dict(weights, strict=False)
    model.to(config['device'])

    # Discriminator
    discriminator_struct = {}
    for tech in config['teachers']:
        discriminator = StageDiscriminator(no_teachers=1).to(config['device'])
        optimizerD = optim.Adam(discriminator.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        discriminator_struct[tech] = {
            "disc": discriminator,
            "optim": optimizerD
        }

    criterionD = torch.nn.BCELoss()

    # Save info about experiment
    save_logs_about(os.path.join(config['exp_path'], config['exp_name']), json.dumps(config, indent=2))
    criterion = getattr(loss_functions, config['loss_function'])

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_sch_step'], gamma=config['lr_sch_gamma'])

    data_manager = DataManager(config)
    if config['data_set'] == 'av':
        train_loader, test_loader = data_manager.get_train_test_dataloaders_avenue()
    else:
        train_loader, test_loader = data_manager.get_train_test_dataloaders_shanghai()

    trainer = Trainer(model, train_loader, test_loader, criterion, optimizer, lr_scheduler, logs_writer, config,
                      discriminator_struct, criterionD)
    trainer.train()


def main_ae():
    config = json.load(open('./config_ae.json'))
    config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

    try:
        os.mkdir(os.path.join(config['exp_path'], config['exp_name']))
    except FileExistsError:
        print("Director already exists! It will be overwritten!")

    logs_writer = SummaryWriter(os.path.join('runs', config['exp_name']))

    model = CyTranAE(input_nc=9, output_nc=1, n_downsampling=4, depth=5, heads=5, pretraining=True).to(config['device'])
    model.to(config['device'])

    save_logs_about(os.path.join(config['exp_path'], config['exp_name']), json.dumps(config, indent=2))

    criterion = getattr(loss_functions, "stage_loss_ae")

    optimizer = optim.Adam(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, config['lr_sch_step'], gamma=config['lr_sch_gamma'])

    data_manager = DataManager(config)
    if config['data_set'] == "avenue":
        train_loader = data_manager.get_dataloaders_avenue_AE()
    else:
        train_loader = data_manager.get_dataloaders_shanghai_AE()

    trainer = TrainerAE(model, train_loader, criterion, optimizer, lr_scheduler, logs_writer, config)
    trainer.train()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", default='anomaly-detection', type=str, help="Possible values: autoencoder or anomaly-detection ")
    args = parser.parse_args()
    if args.mode == 'anomaly-detection':
        main()
    elif args.mode == 'autoencoder':
        main_ae()
    else:
        raise Exception(f"Mode {args.mode} is not supported")