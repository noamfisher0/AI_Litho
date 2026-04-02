import torch
import argparse
from Modules.FNO2d_pl import FNO2d_pl
from utils.utils_read import read_cli, get_workdir, save_id, save_params_to_json, load_params_from_json
import wandb

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

import copy
import json
import os
import sys

import pandas as pd
import torch
import pprint

os.environ["WANDB__SERVICE_WAIT"] = "300"
os.environ["WANDB_DIR"] = "/cluster/work/math/braonic/TrainedModels/general_project_wandb_logs"

if __name__ == "__main__":
    #if len(sys.argv) <= 2:
    '''
        config = {

            "which_example": 'ns_pwc',
            "n_layers": 4,
            "width": 128,
            "modes": (16,16),
            "hidden_dim": 128,
            "use_conv": True,
            "conv_filters": [3,5],
            "padding": None,
            "include_grid": True,
            "is_time": True,

            "lr": 0.001,
            "batch_size": 32,
            "weight_decay": 1e-6,
            "scheduler_step": 15,
            "scheduler_gamma": 0.98,
            "epochs": 100,

            "time_steps": 7,          # How many time steps to select?
            "dt": 2,                  # What is the time step? (1 means include entire traj, 2 means taking every other step, etc.
            "num_samples": 128,        # How many training samples?
            "time_input": 1,          # Should we include time in the input channels?
            "allowed": 'all',         # All2ALL (train) - all , or One2All (train) - one2all, AR training - one
        }

        # AVAILABLE EXPERIMENTS:
        # "ns_brownian", "ns_pwc", "ns_gauss", "ns_sin", "ns_vortex", "ns_shear
        # "ns_pwc_t:
        # "eul_kh", "eul_riemann", "eul_riemann_kh", "eul_riemann_cur", "eul_gauss"
        # "rich_mesh", "rayl_tayl" "kolmogorov"
        # "wave_seismic", "wave_gauss", "allen_cahn"
        # "airfoil", "poisson_gauss", "helmholtz"

        # WHAT IS THE EXPERIMENT?
        which_example = config["which_example"]

        folder = "/cluster/work/math/braonic/TrainedModels/FNO_Conv_"+which_example

    else:
        raise ValueError("To many args")
    '''

    parser = argparse.ArgumentParser(description="Load parameters")
    args = read_cli(parser).parse_args()
    sweep_folder = args.sweep_dir
    try:
        # First, try to parse it as a JSON string
        config = json.loads(args.config)
    except (json.decoder.JSONDecodeError, ValueError):
        try:
            # Try to open the file and load it
            with open(args.config, 'r') as file:
                config = json.load(file)
        except (FileNotFoundError, ValueError):
            raise ValueError("The input is not a a valid JSON")

    #---------------------------------------------------------
    # Load parameters related to the specific experiment -- "DataLoaders/all_experiments.json"

    Dict_EXP = json.load(open("Dataloaders/all_experiments_config.json"))
    if config['which_example'] in Dict_EXP:
        config =  config | Dict_EXP[config['which_example']]
    else:
        raise ValueError("Please specify different benchmark")


    if "workdir" in args and args.workdir and os.path.isfile(args.workdir + '/ids.txt'):

        raise Exeption("No yet tested feature")
        '''
        # Restart from existing run
        ids = dict()

        with open(args["workdir"] + '/ids.txt', 'r') as f:
            for line in f:
                key, value = line.strip().split(':')
                ids[key.strip()] = value.strip()

        run_id = ids["Run ID"]
        print("Found existing ID. Resuming Wandb")
        config = load_params_from_json(args.workdir)
        workdir = args["workdir"]

        run = wandb.init(entity="bogdanraonic",
                         project=params.wandb_project_name,
                         name=params.wandb_run_name,
                         config=config,
                         id=run_id,
                         resume=True)

        parameters = dict(wandb.config)
        '''
    else:
        print("Initalize Wandb")

        run = wandb.init(entity="bogdanraonic",
                         project=args.wandb_project_name,
                         name=args.wandb_run_name,
                         config=config)

        #parameters = dict(wandb.config)
        workdir = get_workdir(run,
                              sweep_folder,
                              main_path= "/cluster/work/math/braonic/TrainedModels/general_project/") if ("workdir" not in config or not config["workdir"]) else os.path.join(os.getcwd(), config["workdir"])

        if not os.path.isdir(workdir):
            os.mkdir(workdir)

        save_id(run, workdir)
        config["workdir"] = workdir

    print(f"Working and saving directory {workdir}")
    pprint.pprint(config)
    save_params_to_json(config, workdir)


    ###########
    #config["include_zero"] = False
    #config["time_input"] = False
    #config["in_dim"] = 2
    ###########


    #---------------------------------------------------------
    # Which transitions during the training are allowed?
    _allowed = []
    if "include_zero" in config and config["include_zero"]:
        start_t = 0
    else:
        start_t = 1

    if config['allowed'] == 'all':
        for t in range(start_t, config["time_steps"] + 1):
            _allowed.append(t)
    elif config['allowed']  == "one2all":
        _allowed = None
    elif config['allowed'] == 'one':
        _allowed = [1]

    config["allowed_tran"] = _allowed

    #df = pd.DataFrame.from_dict([config]).T
    #df.to_csv(folder + '/config.txt', header=False, index=True, mode='w')


    #---------------------------------------------------------
    # Initialize CNO

    config_arch = config["arch"]
    model = FNO2d_pl(in_dim = config["in_dim"],
                     out_dim = config["out_dim"],
                     config_arch = config_arch,
                     is_time = config_arch["is_time"],
                     lr =config["lr"],
                     batch_size = config["batch_size"],
                     weight_decay = config["weight_decay"],
                     scheduler_step = config["scheduler_step"],
                     scheduler_gamma = config["scheduler_gamma"],
                     config = config)

    #---------------------------------------------------------

    ver = 123 # Just a random string to be added to the model name

    checkpoint_callback = ModelCheckpoint(dirpath = workdir+"/model"+str(ver), monitor='mean_val_loss')
    early_part = 10
    early_stop_callback = EarlyStopping(monitor="mean_val_loss", patience=config["epochs"]//early_part)

    logger = TensorBoardLogger(save_dir=workdir, version=ver, name="logs")
    trainer = Trainer(devices = -1,
                    max_epochs = config["epochs"],
                    callbacks = [checkpoint_callback,early_stop_callback],
                    logger=logger)
    trainer.fit(model)
    trainer.validate(model)

    #---------------------------------------------------------


'''
config = dict()
config["which"] = "example"
config["separate"] = True
config["separate_dim"] = [1,1]

model = FNO2d_pl(in_dim =2,
                 out_dim = 2,
                 n_layers = 4,
                 width = 64,
                 modes = (16,16),
                 hidden_dim = 128,
                 use_conv = True,
                 conv_filters = [3,5],
                 padding = None,
                 include_grid = True,
                 is_time = True,
                 config = config)

model.model.print_size()

X = torch.zeros((16,2,128,128))
T = torch.ones((16))
Y_pred = model(X, T)
Y = torch.ones((16,2,128,128))

#print(model.training_step((T, X, Y)))
model.validation_step((T, X, Y), 0)
model.on_validation_epoch_end()
'''
