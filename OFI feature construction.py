# -*- coding: utf-8 -*-
"""
Created on Fri May  2 08:49:25 2025

@author: User
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import statsmodels.api as sm

file = r"C:\Users\User\Downloads\first_25000_rows.csv"
df = pd.read_csv(file, parse_dates=["ts_event"])
df_sorted = df.sort_values("ts_event").reset_index(drop=True)

def simulate_second_asset(df, name='sim'):
    df_sim = df.copy()
    np.random.seed(42)
    for m in range(10):
        df_sim[f'bid_px_0{m}'] *= (1 + 0.0005 * np.random.randn())
        df_sim[f'ask_px_0{m}'] *= (1 + 0.0005 * np.random.randn())
        df_sim[f'bid_sz_0{m}'] = np.maximum(1, df_sim[f'bid_sz_0{m}'] + np.random.randint(-1, 2, len(df_sim)))
        df_sim[f'ask_sz_0{m}'] = np.maximum(1, df_sim[f'ask_sz_0{m}'] + np.random.randint(-1, 2, len(df_sim)))
    df_sim.columns = [col if col == 'ts_event' else f'{name}_{col}' for col in df_sim.columns]
    return df_sim

def compute_nlevel_ofi(df, m):
    ofi = [0]
    
    for i in range(1, len(df)):
        if df_sorted[f'bid_px_0{m}'][i] == df_sorted[f'bid_px_0{m}'][i-1]:
            of_b = df_sorted[f'bid_sz_0{m}'][i] - df_sorted[f'bid_sz_0{m}'][i-1]
        elif df_sorted[f'bid_px_0{m}'][i] > df_sorted[f'bid_px_0{m}'][i-1]:
            of_b = df_sorted[f'bid_sz_0{m}'][i]
        elif df_sorted[f'bid_px_0{m}'][i] < df_sorted[f'bid_px_0{m}'][i-1]:
            of_b = -df_sorted[f'bid_sz_0{m}'][i-1]
            
        if df_sorted[f'ask_px_0{m}'][i] == df_sorted[f'ask_px_0{m}'][i-1]:
            of_a = df_sorted[f'ask_sz_0{m}'][i] - df_sorted[f'ask_sz_0{m}'][i-1]
        elif df_sorted[f'ask_px_0{m}'][i] > df_sorted[f'ask_px_0{m}'][i-1]:
            of_a = df_sorted[f'ask_sz_0{m}'][i]
        elif df_sorted[f'ask_px_0{m}'][i] < df_sorted[f'ask_px_0{m}'][i-1]:
            of_a = -df_sorted[f'ask_sz_0{m}'][i-1]
        ofi.append(of_b - of_a)
        
    return ofi

def compute_q(df, levels):
    q_list = []
    
    for i in range(len(df)):
        depth_sum = 0
        for m in range(levels):
            depth_sum += (df_sorted[f'bid_sz_0{m}'][i] + df_sorted[f'ask_sz_0{m}'][i])/2
        q = depth_sum/levels
        q_list.append(q)
        
    return np.array(q_list)

def compute_bestlevel_ofi(df):
    bestofi = compute_nlevel_ofi(df, 0)
    df['bestlevel_ofi'] = bestofi
    
    return df

def compute_multilevel_ofi(df, levels=10):
    ofi_matrix = []
    for i in range(levels):
        ofi_matrix.append(compute_nlevel_ofi(df, i))
    ofi_matrix_np = np.array(ofi_matrix).T
    q_array = compute_q(df_sorted, levels=10)
    norm_ofi = ofi_matrix_np[1:] / q_array[1:, np.newaxis]
    for i in range(levels):
       df[f'norm_ofi_level_{i}'] = np.nan
       df.loc[1:, f'norm_ofi_level_{i}'] = norm_ofi[:, i]
       
    return df

def compute_integrated_ofi(df, levels=10):
    ofi_cols = [f'norm_ofi_level_{i}' for i in range(levels)]
    df_multi = compute_multilevel_ofi(df, 10)
    ofi_matrix = df_multi[ofi_cols].dropna().values
    
    pca = PCA(n_components=1)
    pca.fit(ofi_matrix)
    w1 = pca.components_[0]
    w1 /= np.sum(np.abs(w1))
    integrated_ofi = ofi_matrix @ w1
    
    full_integrated_ofi = [np.nan] + list(integrated_ofi)
    df['integrated_ofi'] = full_integrated_ofi
    
    return df

def compute_cross_asset_regression(df, levels=10):
    df_sim = simulate_second_asset(df)
    
    df = compute_integrated_ofi(df, levels)
    df_sim = compute_integrated_ofi(df_sim, levels)
    
    df_merged = pd.merge(
        df[['ts_event', 'bid_px_00', 'integrated_ofi']].rename(columns={
            'integrated_ofi': 'ofi_self',
            'bid_px_00': 'price_self'
        }),
        df_sim[['ts_event', 'integrated_ofi']].rename(columns={
            'integrated_ofi': 'ofi_cross'
        }),
        on='ts_event',
        how='inner'
    )

    df_merged['return'] = df_merged['price_self'].pct_change()
    df_reg = df_merged.dropna(subset=['return', 'ofi_self', 'ofi_cross'])
    X = sm.add_constant(df_reg[['ofi_self', 'ofi_cross']])
    y = df_reg['return']
    model = sm.OLS(y, X).fit()

    return model.summary()

bestlevel_ofi = compute_bestlevel_ofi(df_sorted)
multilevel_ofi = compute_multilevel_ofi(df_sorted)
integrated_ofi = compute_integrated_ofi(df_sorted)
crossasset_ofi = compute_cross_asset_regression(df_sorted)