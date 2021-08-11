from features import *
import numpy as np
import pandas as pd


# Read csv
df = pd.read_csv('bearing_signals.csv')


# Feature Engineering
bearing_signals = cleanup_dataset(df)
df_mean_axis = mean_axis_feature(bearing_signals)
df_fft = fft_feature(bearing_signals)
df_fft_mean = fft_mean_feature(df_fft)
df_amp = amp_feature(df_fft)
df_amp_max = max_amp_feature(df_amp)
df_amp_mean = mean_amp_feature(df_amp)


# Merge all df's into one with all features
def merge_df():
    df_final = (pd.merge(df_mean_axis, df_fft, on='bearing_id'))
    df_final = (pd.merge(df_final, df_fft_mean, on='bearing_id'))
    df_final = (pd.merge(df_final, df_amp, on='bearing_id'))
    df_final = (pd.merge(df_final, df_amp_max, on='bearing_id'))
    df_final = (pd.merge(df_final, df_amp_mean, on='bearing_id'))
    return df_final


# Write new df into csv
df_final = merge_df()
df_final.to_csv('bearing_final_data.csv')