import json
import os
import wandb
import time

def read_cli(parser):
    """Reads command line arguments."""

    parser.add_argument("--config", type=str, help="Path to  JSON string")
    parser.add_argument("--config_exp", type=str, default="/cluster/home/braonic/FNO/Dataloaders/all_experiments_config.json", help="Path to Benchmark JSON string with benchmark details")
    parser.add_argument("--workdir", type=str, help="Working directory")
    parser.add_argument("--sweep_dir", type=str, default=None, help="Sweep Directory")
    parser.add_argument("--run_id", type=str, default=None, help="Wandb Run Id")

    parser.add_argument("--wandb-run-name", type=str, required=False, default=None, help="Name of the run in wandb")
    parser.add_argument("--wandb-project-name", type=str, default="general_project", help="Name of the wandb project")

    '''
    parser.add_argument("--n_layers", type=int, default=4, help = "")
    parser.add_argument("--width", type=int, default=128, help = "")
    parser.add_argument("--modes", type=tuple, default=(16, 16), help = "")
    parser.add_argument("--hidden_dim", type=int, default=128, help = "")
    parser.add_argument("--use_conv", type=bool, default=False, help = "")
    parser.add_argument("--conv_filters", type=list, default=[3,5], help = "")
    parser.add_argument("--padding", type=int, default=None, help = "")
    parser.add_argument("--include_grid", type=bool, default=True, help = "")
    parser.add_argument("--is_time", type=bool, default=True, help = "")
    '''

    return parser

def save_params_to_json(param_dict, filepath):
    # Save the dictionary to a file as JSON
    with open(filepath + "/parameters.json", 'w') as json_file:
        json.dump(param_dict, json_file, indent=4, )

def load_params_from_json(filepath):
    with open(filepath + "/parameters.json", 'r') as json_file:
        params_dict = json.load(json_file)
    return params_dict

def save_id(run_, filepath):
    print(f"Sweep ID: {run_.sweep_id}")
    print(f"Run ID: {run_.id}")
    with open(filepath + '/ids.txt', 'w') as file:
        file.write(f"Sweep ID: {run_.sweep_id}\n")
        file.write(f"Run ID: {run_.id}")

def get_workdir(run_, sweep_folder, main_path=os.getcwd()):
    # Access the run's name
    run_name = run_.name
    # Access the sweep ID (if exists)
    sweep_id = run_.sweep_id
    if sweep_id is not None and sweep_folder is not None:
        base_path = main_path + os.sep + sweep_folder + os.sep
        if not os.path.isdir(base_path):
            wait = np.random.uniform(1,5)
            time.sleep(wait)
            if not os.path.isdir(base_path):
                os.mkdir(base_path)
    else:
        if sweep_folder is not None:
            base_path = "/cluster/work/math/braonic/TrainedModels/finetuned_models/"
        else:
            base_path = main_path + os.sep

    return base_path + run_name
