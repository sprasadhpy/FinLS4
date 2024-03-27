# my_config.py

class Config:
    def __init__(self):
        # General configuration
        self.h = 1
        self.l = 20  # Condition length
        self.pred = 1


        # Dataset and ETFs-Stocks list locations
        self.dataloc = "./Data/"
        self.etflistloc = "./Data/stocks-etfs-list.csv"

        # Number of available GPUs
        self.ngpu = 1

        # Locations for saving results
        self.loc = "./Results"
        self.modelsloc = "./TrainedModels/"
        self.plotsloc = "./Plots/"
        self.resultsloc = "./Results/"

        # Model parameters
        self.tanh_coeff = 100
        self.z_dim = 20  # Noise dimension
        self.hid_d = 8
        self.hid_g = 8

        # Training parameters
        self.checkpoint_epoch = 20
        self.batch_size = 4096
        self.diter = 1

        self.n_epochs = 10
        self.ngrad = 5
        self.vl_later = False

        # Data split ratios
        self.tr = 0.8
        self.vl = 0.1

        # Plotting settings
        self.plot = False

        # Adjusted model parameters
        self.z_dim_adjusted = 8
        self.hid_d_s = [8]
        self.hid_g_s = []

        # Optional exploration of different learning rates
        self.lrg_s = [0.00001]
        self.lrd_s = [0.00001]
        self.vl_later_adjusted = True

        # Results processing
        self.nres = len(self.lrg_s)
        self.resultsname = "./Results/results.csv"

        # Matplotlib figure size setting
        self.figure_size = [15.75, 9.385]

# Instantiate the configuration object
config = Config()
