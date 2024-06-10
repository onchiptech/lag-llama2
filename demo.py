"""
Download our pretrained model weights from [HuggingFace](https://huggingface.co/time-series-foundation-models/Lag-Llama)
!huggingface-cli download time-series-foundation-models/Lag-Llama lag-llama.ckpt --local-dir /content/lag-llama
"""

from itertools import islice

from matplotlib import pyplot as plt
import matplotlib.dates as mdates

import torch
from gluonts.evaluation import make_evaluation_predictions, Evaluator
from gluonts.dataset.repository.datasets import get_dataset

from gluonts.dataset.pandas import PandasDataset
from gluonts.dataset.common import ListDataset

import pandas as pd
import numpy as np

from lag_llama.gluon.estimator import LagLlamaEstimator

from datetime import datetime
from datetime import datetime, timedelta
from nsepython import *

import time

PREDICTION_LENGTH=20

def get_lag_llama_predictions(dataset, prediction_length, device, context_length=32, use_rope_scaling=False, num_samples=100):
    ckpt = torch.load("lag-llama.ckpt", map_location=device) # Uses GPU since in this Colab we use a GPU.
    estimator_args = ckpt["hyper_parameters"]["model_kwargs"]

    print("estimator_args: ", estimator_args)

    rope_scaling_arguments = {
        "type": "linear",
        "factor": max(1.0, (context_length + prediction_length) / estimator_args["context_length"]),
    }
    
    estimator = LagLlamaEstimator(
        ckpt_path="lag-llama.ckpt",
        prediction_length=prediction_length,
        context_length=context_length, # Lag-Llama was trained with a context length of 32, but can work with any context length

        # estimator args
        input_size=estimator_args["input_size"],
        n_layer=estimator_args["n_layer"],
        n_embd_per_head=estimator_args["n_embd_per_head"],
        n_head=estimator_args["n_head"],
        scaling=estimator_args["scaling"],
        time_feat=estimator_args["time_feat"],
        rope_scaling=rope_scaling_arguments if use_rope_scaling else None,

        batch_size=1,
        num_parallel_samples=num_samples,
        device=device,
    )

    lightning_module = estimator.create_lightning_module()
    transformation = estimator.create_transformation()
    predictor = estimator.create_predictor(transformation, lightning_module)
    
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=dataset,
        predictor=predictor,
        num_samples=num_samples
    )
    forecasts = list(forecast_it)
    tss = list(ts_it)

    return forecasts, tss




def download_symbol_history(symbol, period=364*3):

    end_date = datetime.datetime.now().strftime("%d-%m-%Y")
    end_date = str(end_date)

    start_date = (datetime.datetime.now()- datetime.timedelta(days=period)).strftime("%d-%m-%Y")
    start_date = str(start_date)

    #  the start date
    #print(f"download '{symbol}' from : {start_date}")

    #data = yf.download(tickers=f"{symbol}.NS", period=f"{period}d", interval="15m")
    #data.index = data.index.tz_localize(None)
    print(f"Dowloading {symbol} from {start_date} to {end_date}")
    df = equity_history(symbol,"EQ",start_date,end_date)
    if len(df) > 0:
      df.rename(columns={'CH_TIMESTAMP': 'date', 'CH_SYMBOL': 'tic', 'CH_OPENING_PRICE': 'open','CH_TRADE_HIGH_PRICE': 'high', 'CH_TRADE_LOW_PRICE': 'low', 'CH_CLOSING_PRICE': 'close', 'VWAP': 'vwap'}, inplace=True)

      df = df[["date", "open", "high", "low", "close", "vwap"]].copy()
      df.sort_values(by='date', inplace=True)
      df['date'] = pd.to_datetime(df['date'])
      df['date'] = df['date'].dt.strftime('%Y-%m-%d')
      df.index = df.date.tolist()

    print("Done.")
    return df



def process_symbol(symbol):

    if os.path.exists(f"{symbol}.csv"):
        df_source = pd.read_csv(f"{symbol}.csv")
    else:    
        df_source = download_symbol_history(symbol)
        df_source.reset_index(drop=True, inplace=True)
        df_source.to_csv(f"{symbol}.csv", index=False)

    df_source.drop_duplicates(inplace=True)  
    df_source.reset_index(inplace=True)


    # Convert the 'date' column to datetime if it's not already in datetime format
    df_source['date'] = pd.to_datetime(df_source['date'])

    # append next date after end date for resample to work properly
    next_date = df_source.date.max() + pd.Timedelta(days=1)
    next_row = df_source.iloc[[-1]].copy()
    next_row['date'] = next_date
    df_source = pd.concat([df_source, next_row], ignore_index=True)

    df_source["observed"] = True

    start_date = pd.to_datetime(df_source.date.min())
    end_date = pd.to_datetime(df_source.date.max())
    date_range = pd.date_range(start=start_date, end=end_date, freq='1D')

    df_range = pd.DataFrame({"date": date_range.tolist()})

    df_source = pd.merge(df_range, df_source, on="date", how='left')
    df_source.observed.fillna(False, inplace=True)
    df_source.ffill(axis = 0, inplace=True)


    # Set the 'date' column as the index
    df_source.set_index('date', inplace=True)

    # Resample the data to 6-hour frequency
    df_target = df_source.resample('6H').ffill()
    # Drop the rows with the next date
    df_target.reset_index(drop=False, inplace=True)
    df_target = df_target[df_target.date < next_date]
    

    # Now 'df_target' contains rows with 6-hour frequency and the same values for 'open', 'low', 'high', and 'close'
    # Update 'target' based on hour
    df_target.loc[df_target.date.dt.hour == 0, 'target'] = df_target.loc[df_target.date.dt.hour == 0, 'open']
    df_target.loc[df_target.date.dt.hour == 6, 'target'] = df_target.loc[df_target.date.dt.hour == 6, 'low']
    df_target.loc[df_target.date.dt.hour == 12, 'target'] = df_target.loc[df_target.date.dt.hour == 12, 'high']
    df_target.loc[df_target.date.dt.hour == 18, 'target'] = df_target.loc[df_target.date.dt.hour == 18, 'close']


    # Set all row values to NaN except 'date' and 'observed' columns
    df_target.loc[~df_target['observed'], 'target'] = float('nan')

    df_target.to_csv(f"{symbol}-ts.csv", index=False)

    # Create the GluonTS dataset
    dataset = ListDataset(
        data_iter=[{'start': start_date, 'target': df_target["target"].tolist()}],
        freq='6H'  # 15-minute frequency
    )

    
    stime = time.time()
    context_length = 32
    num_samples = 100
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    prediction_length = PREDICTION_LENGTH

    forecasts, tss = get_lag_llama_predictions(dataset, prediction_length=prediction_length, num_samples=num_samples, context_length=context_length, device=device)

    return forecasts, tss
if __name__ == "__main__":
  

    forecasts, tss = process_symbol("ITC")

    plt.figure(figsize=(20, 15))
    date_formater = mdates.DateFormatter('%b, %d')
    plt.rcParams.update({'font.size': 15})

    # Iterate through the first 9 series, and plot the predicted samples
    for idx, (forecast, ts) in islice(enumerate(zip(forecasts, tss)), 9):
        ax = plt.subplot(3, 3, idx+1)

        plt.plot(ts[-4 * PREDICTION_LENGTH:].to_timestamp(), label="target", )
        forecast.plot( color='g')
        plt.xticks(rotation=60)
        ax.xaxis.set_major_formatter(date_formater)
        ax.set_title(forecast.item_id)

    plt.gcf().tight_layout()
    plt.legend()
    #plt.show()
    plt.savefig('output.png')
    