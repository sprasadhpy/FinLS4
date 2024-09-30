import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import yfinance as yf
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import warnings

from Utils.excessreturns import excessreturns
from Utils.rawreturns import rawreturns


# Define a function to fit the HMM on given data
def fit_hmm(data):
    # Set up the parameter grid for HMM
    n_components_list = [2, 3, 4, 5]  # Vary number of hidden states
    covariance_type_list = ['full', 'diag', 'tied']  # Try different covariance types
    n_iter_list = [100, 200, 300]  # Different numbers of iterations
    random_state_list = [42, 100, 200]  # Try different random seeds
    best_score = -np.inf
    best_params = None
    best_model = None



    for n_components in n_components_list:
        for covariance_type in covariance_type_list:
            for n_iter in n_iter_list:
                for random_state in random_state_list:
                    try:
                        # Initialize and fit the HMM model
                        model = hmm.GaussianHMM(n_components=n_components, covariance_type=covariance_type,
                                                n_iter=n_iter, random_state=random_state)

                        # Initialize start probabilities and transition matrix
                        model.startprob_ = np.ones(n_components) / n_components
                        transmat = np.full((n_components, n_components), 1 / n_components)
                        np.fill_diagonal(transmat, 0.7)

                        # Normalize the transition matrix to ensure each row sums to 1
                        row_sums = transmat.sum(axis=1, keepdims=True)
                        transmat = transmat / row_sums
                        model.transmat_ = transmat

                        # Fit the model
                        model.fit(data)

                        # Get the log likelihood score
                        score = model.score(data)

                        # Save the best model
                        if score > best_score:
                            best_score = score
                            best_params = {
                                'n_components': n_components,
                                'covariance_type': covariance_type,
                                'n_iter': n_iter,
                                'random_state': random_state
                            }
                            best_model = model

                    except Exception as e:
                        print(
                            f"Model failed to converge with params: n_components={n_components}, covariance_type={covariance_type}, n_iter={n_iter}, random_state={random_state}. Error: {e}")
                        continue

    return best_model, best_params, best_score


def analyse_regime(cfg):
    """
    Analyse the regime of the stock
    """

    print("Regime analysis started...")
    stock = cfg.current_ticker
    etf = cfg.current_etf
    dataloc = cfg.dataloc
    stock_path = dataloc + '/' + stock + '.csv'
    etf_path = dataloc + '/' + etf + '.csv'
    stock_df = pd.read_csv(stock_path)
    etf_df = pd.read_csv(etf_path)

    if cfg.current_ticker[0] != 'X':
        excess_returns, dates_dt = excessreturns(cfg, stock_df, etf_df)
    else:
        excess_returns, dates_dt = rawreturns(cfg, stock_df, etf_df)

    scaled_data = StandardScaler().fit_transform(excess_returns.reshape(-1, 1))
    # Fit the HMM model
    model, params, score = fit_hmm(scaled_data)


    regimes = model.predict(scaled_data)

    print(len(dates_dt))
    print(len(excess_returns))
    print(len(regimes))
    print(len(scaled_data))


    # combine excess returns and regimes into a DataFrame
    df = pd.DataFrame({'excess_returns': excess_returns, 'regime': regimes})


    # Create a color map for the regimes
    unique_regimes = df['regime'].unique()
    colors = plt.cm.rainbow(np.linspace(0, 1, len(unique_regimes)))
    color_map = dict(zip(unique_regimes, colors))

    # Create a new figure
    fig, ax = plt.subplots(2, 1, figsize=(15, 10), sharex=True)

    # Iterate over the unique regimes
    for regime in unique_regimes:
        # Create a mask for the current regime
        mask = df['regime'] == regime

        # Plot the excess returns for the current regime with its corresponding color
        ax[0].plot(df.loc[mask, 'excess_returns'], color=color_map[regime], label=f'Regime {regime}')

    ax[0].set_title(f'{stock} Excess Returns')
    ax[0].legend()

    # Plot the regimes
    ax[1].plot(df['regime'], label='Regime', color='black')
    ax[1].set_title(f'{stock} Regime')
    ax[1].legend()

    plt.show()



    print("HMM fit successful!")
