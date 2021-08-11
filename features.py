import numpy as np
import pandas as pd
from scipy.fft import fft, fftfreq, fftshift


# Cleaning up DataFrame
def cleanup_dataset(df):

    # Drop Good Bearings (status = 1)
    df.drop(df[df['experiment_id'] == 1].index, inplace=True)
    for row in range(101, 113):
        df.drop(df[df['experiment_id'] == row].index, inplace=True)

    # Drop columns & Set Index ("bearing_id")
    df = df.drop(['bearing_1_id','experiment_id'], axis=1)
    df.rename(columns={'bearing_2_id': 'bearing_id'}, inplace=True)
    df = df.set_index('bearing_id')

    # Reduce Timestamp feature
    df = df[df['timestamp'] > 0.25]
    df = df[df['timestamp'] <= 1.5]
    return df


# Feature 1: "Mean" value of x-y-z axis (a1 + a2 sensors)
def mean_axis_feature(df):
    df_mean = df.groupby('bearing_id').mean()
    df_mean.columns = ['a1_x_mean','a1_y_mean','a1_z_mean','a2_x_mean','a2_y_mean','a2_z_mean','rpm_mean','hz_mean','w_mean']
    return df_mean


# Feature 2: "fft" signal of x-y-z axis (a1 + a2 sensors)
def fft_feature(df):
    df_fft = df.copy()
    acceleration_cols = ['a1_x','a1_y','a1_z','a2_x','a2_y','a2_z']
    acceleration_fft = fft(df[acceleration_cols].values)
    df_fft['a1_x_fft'] = abs(acceleration_fft[:,0])
    df_fft['a1_y_fft'] = abs(acceleration_fft[:,1])
    df_fft['a1_z_fft'] = abs(acceleration_fft[:,2])
    df_fft['a2_x_fft'] = abs(acceleration_fft[:,3])
    df_fft['a2_y_fft'] = abs(acceleration_fft[:,4])
    df_fft['a2_z_fft'] = abs(acceleration_fft[:,5])
    return df_fft


# Feature 3: "Mean fft" value of x-y-z axis (a1 + a2 sensors)
def fft_mean_feature(df):
    df_fft_mean = df.groupby('bearing_id').mean()
    df_fft_mean = df_fft_mean.drop(['a1_x', 'a1_y', 'a1_z', 'a2_x', 'a2_y', 'a2_z', 'rpm', 'hz', 'w'], axis=1)
    df_fft_mean.columns = ['a1_x_fft_mean', 'a1_y_fft_mean', 'a1_z_fft_mean', 'a2_x_fft_mean', 'a2_y_fft_mean', 'a2_z_fft_mean']
    return df_fft_mean


# Feature 4: "Max Amplitude" value of x-y-z axis (a1 + a2 sensors)
def amp_feature(df):
    df_amplitude = df.copy()
    amp_cols = ['a1_x_fft','a1_y_fft','a1_z_fft','a2_x_fft','a2_y_fft','a2_z_fft']
    amplitude = np.abs(df[amp_cols].values)**2
    df_amplitude['a1_x_amp'] = abs(amplitude[:,0])
    df_amplitude['a1_y_amp'] = abs(amplitude[:,1])
    df_amplitude['a1_z_amp'] = abs(amplitude[:,2])
    df_amplitude['a2_x_amp'] = abs(amplitude[:,3])
    df_amplitude['a2_y_amp'] = abs(amplitude[:,4])
    df_amplitude['a2_z_amp'] = abs(amplitude[:,5])
    return df_amplitude


# Feature 5: "Max Amplitude" value of x-y-z axis (a1 + a2 sensors)
def max_amp_feature(df):
    df_amp_max = df.copy()
    df_amp_max = df_amp_max.groupby('bearing_id').max()
    df_amp_max = df_amp_max.drop(['a1_x_fft', 'a1_y_fft', 'a1_z_fft', 'a2_x_fft', 'a2_y_fft', 'a2_z_fft'], 1)
    df_amp_max.columns = ['a1_x_amp_max', 'a1_y_amp_max', 'a1_z_amp_max', 'a2_x_amp_max', 'a2_y_amp_max', 'a2_z_amp_max']
    return df_amp_max


# Feature 6: "Mean Amplitude" value of x-y-z axis (a1 + a2 sensors)
def mean_amp_feature(df):
    df_amp_mean = df.copy()
    df_amp_mean = df_amp_mean.groupby('bearing_id').mean()
    df_amp_mean = df_amp_mean.drop(['a1_x_fft', 'a1_y_fft', 'a1_z_fft', 'a2_x_fft', 'a2_y_fft', 'a2_z_fft'], 1)
    df_amp_mean.columns = ['a1_x_amp_mean', 'a1_y_amp_mean', 'a1_z_amp_mean', 'a2_x_amp_mean', 'a2_y_amp_mean', 'a2_z_amp_mean']
    return df_amp_mean