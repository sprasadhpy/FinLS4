import os

class Config:
    def __init__(self, model='ForGAN LSTM', tickers=[]):

        self.model = model
        # Tickers to train on
        self.tickers = tickers

        # General configuration
        self.h = 1 # Horizon
        self.l = 20  # Condition length
        self.pred = 1 # Prediction length



        # Define the data start and end dates
        self.start_date = "01-01-2000"
        self.test_start_date = "2022-01-01"

        # Get the directory where the Configuration.py file exists
        current_directory = os.path.dirname(os.path.realpath(__file__))

        # Dataset and ETFs-Stocks list locations
        self.rawstocksloc = os.path.join(current_directory, "RawData/data_2_final.csv")
        self.rawetfsloc = os.path.join(current_directory, "RawData/ETFs-data.csv")

        self.dataloc = os.path.join(current_directory, "Data")

        self.etflistloc = os.path.join(current_directory, "RawData/stocks-etfs-list.csv")

        # Number of available GPUs
        self.device = 'cpu'
        self.ngpu = 1

        # Locations for saving results
        self.loc = os.path.join(current_directory, "Results")
        self.modelsloc = os.path.join(current_directory, "TrainedModels")
        self.plotsloc = os.path.join(current_directory, "Plots")
        self.resultsloc = os.path.join(current_directory, "Results")

        # Model parameters
        self.tanh_coeff = 100
        self.z_dim = 20  # Noise dimension
        self.hid_g = 8
        self.hid_d = 8

        # Training parameters
        self.checkpoint_epoch = 20
        self.batch_size = 1024
        self.diter = 1

        self.n_epochs = 10
        self.n_grad = 25
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
        self.lrg_s = 0.00001
        self.lrd_s = 0.00001
        self.vl_later_adjusted = True

        # Results processing
        # self.nres = len(self.lrg_s)
        self.resultsname = "./Results/results.csv"

        # Matplotlib figure size setting
        self.figure_size = [15.75, 9.385]

        # Data Loader parameters
        self.num_workers = 4


config = Config()