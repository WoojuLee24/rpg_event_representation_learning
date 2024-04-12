import pdb
import json

class WandbLogger():
    def __init__(self,
                 config,
                 args=None,
                 wandb_init_kwargs=None,
                 interval=10,
                 log_map_every_iter=True,
                 log_checkpoint=False,
                 log_checkpoint_metadata=False,
                 **kwargs):
        if config is None:
            self.use_wandb = False
        else:
            self.use_wandb = True
            self.config = config
            self.args = args
            self.import_wandb()


    def import_wandb(self):
        try:
            import wandb
        except ImportError:
            raise ImportError(
                'Please run "pip install wandb" to install wandb')
        self.wandb = wandb

    def log(self, key, value):
        self.wandb.log({key: value})

    def convert_to_df(self, wandb_input):
        df = pd.DataFrame(wandb_input)
        df = df.set_axis(labels=self.class_list, axis=0)
        df = df.set_axis(labels=self.class_list, axis=1)
        df_wandb = self.wandb.Table(data=df)
        return df_wandb

    ###########
    ### run ###
    ###########
    def before_run(self):
        if self.use_wandb:
            if self.wandb is None:
                self.import_wandb()
            if self.config:
                self.wandb.init(**self.config)
            else:
                self.wandb.init()


    def log_evaluate(self, wandb_input):
        if self.use_wandb:
            # log wandb features
            for key, value in wandb_input.items():
                self.wandb.log({key: value})


    def log_analysis(self, wandb_input):
        if self.use_wandb:
            if 'test_dg_table' in wandb_input:
                test_dg_table = self.wandb.Table(data=wandb_input['test_dg_table'])
                self.wandb.log({"additional_loss": test_dg_table})
            if 'test_dg_features' in wandb_input:
                for key, value in wandb_input['test_dg_features'].items():
                    self.wandb.log({key: value})