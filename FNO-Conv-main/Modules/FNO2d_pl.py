from Modules.FNO2d import FNO2d
from Modules.GeneralModel_pl import GeneralModel_pl

class FNO2d_pl(GeneralModel_pl):
    def __init__(self,
                in_dim,
                out_dim,
                config_arch = dict(),
                is_time = True,
                lr = 0.0001,
                batch_size = 20,
                weight_decay = 1e-6,
                scheduler_step = 15,
                scheduler_gamma = 0.98,
                config = dict(),

                wandb_aggregation = 20 # After this many gradient steps, the metrics are logged
                ):
        super().__init__(in_dim,
                        out_dim,
                        lr = lr,
                        batch_size = batch_size,
                        weight_decay = weight_decay,
                        scheduler_step = scheduler_step,
                        scheduler_gamma = scheduler_gamma,
                        config = config,
                        wandb_aggregation = wandb_aggregation)

        self.model = FNO2d(in_dim = in_dim,
                        out_dim = out_dim,
                        n_layers = config_arch["n_layers"],
                        width = config_arch["width"],
                        modes =  config_arch["modes"],
                        hidden_dim =  config_arch["hidden_dim"],
                        use_conv = config_arch["use_conv"],
                        conv_filters =  config_arch["conv_filters"],
                        padding = config_arch["padding"],
                        include_grid = config_arch["include_grid"],
                        is_time =  config_arch["is_time"])
