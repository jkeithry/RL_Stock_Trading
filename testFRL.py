##### Modified from autotrain.py #####
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.lines import Line2D
import PyQt5
from sklearn import preprocessing
import tkinter

matplotlib.use("Qt5Agg")
import datetime

from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
from finrl.trade.backtest import backtest_stats, get_baseline, backtest_plot
from stable_baselines3 import SAC
from stable_baselines3 import DDPG


def train_one():
    """
    train an agent
    """
    print("==============Start Fetching Data===========")
    df = YahooDownloader(
        start_date=config.START_DATE,
        end_date=config.END_DATE,
        ticker_list=['FXAIX'],
    ).fetch_data()
    print("==============Start Feature Engineering===========")
    fe = FeatureEngineer(
        use_technical_indicator=True,
        tech_indicator_list=config.TECHNICAL_INDICATORS_LIST,
        use_turbulence=True,
        user_defined_feature=False,
    )

    processed = fe.preprocess_data(df)

    # Training & Trading data split
    train = data_split(processed, config.START_DATE, config.START_TRADE_DATE)
    trade = data_split(processed, config.START_TRADE_DATE, config.END_DATE)

    # calculate state action space
    stock_dimension = len(train.tic.unique())
    state_space = (
        1
        + 2 * stock_dimension
        + len(config.TECHNICAL_INDICATORS_LIST) * stock_dimension
    )
    env_kwargs = {
        "hmax": 100, 
        "initial_amount": 1000000, 
        "buy_cost_pct": 0.001, 
        "sell_cost_pct": 0.001, 
        "state_space": state_space, 
        "stock_dim": stock_dimension, 
        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
        "action_space": stock_dimension, 
        "reward_scaling": 1e-4
        }
    e_train_gym = StockTradingEnv(df=train, **env_kwargs)
    e_trade_gym = StockTradingEnv(df=trade, turbulence_threshold=250, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    env_trade, obs_trade = e_trade_gym.get_sb_env()

    agent = DRLAgent(env=env_train)

    print("==============Model Training===========")
    now = datetime.datetime.now().strftime("%Y%m%d-%Hh%M")
    user_input = input('train model? 1 train 0 don\'t train')
    if user_input == 1:
        model_sac = agent.get_model("sac")
        trained_sac = agent.train_model(
            model=model_sac, tb_log_name="sac", total_timesteps=8000
        )
        trained_sac.save("sac_8k"+df.tic[0]+"_frl")
    else:
        trained_sac = SAC.load('sac_80k_msft_working')
    print("==============Start Trading===========")
    df_account_value, df_actions = DRLAgent.DRL_prediction(trained_sac, e_trade_gym)
    df_account_value.to_csv(
        "./" + config.RESULTS_DIR + "/SAC_df_account_value_" + df.tic[0] + "_" + now + ".csv"
    )
    df_actions.to_csv("./" + config.RESULTS_DIR + "/SAC_df_actions_" + df.tic[0] + "_" + now + ".csv")

    # print("==============Get Backtest Results===========")
    perf_stats_all = backtest_stats(df_account_value)
    perf_stats_all = pd.DataFrame(perf_stats_all)
    perf_stats_all.to_csv("./" + config.RESULTS_DIR + "/SAC_perf_stats_all_" + df.tic[0] + "_" + now + ".csv")

    #plot acc value
    actions = df_actions['actions']
    x = np.arange(0, df_account_value['account_value'].shape[0])
    y = df_account_value['account_value']


    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)

    fig, axs = plt.subplots(2, 1, sharex=True, sharey=False)

    # plt.plot(x, y)


    # Use a boundary norm instead
    cmap = ListedColormap(['r', 'g', 'b'])
    norm = BoundaryNorm([-100, -0.1, 0.1, 100], cmap.N)
    lc = LineCollection(segments, cmap=cmap, norm=norm)
    lc.set_array(actions)
    lc.set_linewidth(2)
    line = axs[0].add_collection(lc)
    # fig.colorbar(line, ax=axs)

    axs[1].set_xlabel('Trading Day (' + 'From ' + config.START_TRADE_DATE + " to " + config.END_DATE + ')')
    axs[0].set_ylabel('Account Value (10000 of USD)')
    axs[0].set_title("Trading Test on " + df.tic[0])

    axs[0].set_xlim(x.min(), x.max())
    axs[0].set_ylim(y.min(), y.max())

    custom_lines = [Line2D([0], [0], color=cmap(0.), lw=4),
                    Line2D([0], [0], color=cmap(.5), lw=4),
                    Line2D([0], [0], color=cmap(1.), lw=4)]

    # lines = ax.plot(data)
    axs[0].legend(custom_lines, ['Sell', 'Hold', 'Buy'])

    #plot stock value
    tx = np.arange(0, df_account_value['account_value'].shape[0])
    ty = trade['close']
    plt.ylabel('Price (USD)')
    plt.title(df.tic[0] + " Closing Price")
    plt.plot(tx, ty)

    plt.savefig("./"+ config.RESULTS_DIR + "/plots/" "SAC_plot_" + df.tic[0] + "_" + now + ".png")
train_one()