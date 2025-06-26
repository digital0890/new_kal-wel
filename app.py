# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import pytz
from pykalman import KalmanFilter
import yfinance as yf
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.signal import argrelextrema
import os
import pywt
import streamlit.components.v1 as components

# جایگزینی ایمن برای import pad
try:
    from pywt._dwt import pad
except ImportError:
    from pywt._doc_utils import pad

from sklearn.metrics import mean_squared_error
import warnings
import streamlit as st
from datetime import datetime
import time

# تنظیمات اولیه
warnings.filterwarnings("ignore", category=UserWarning)
np.random.seed(42)

# تزریق استایل حرفه‌ای
def inject_pro_style():
    pro_css = """
    <style>
        /* Reset و پایه */
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', 'Inter', 'SF Pro Display', -apple-system, BlinkMacSystemFont, sans-serif;
        }
        :root {
            --primary: #6366f1;
            --primary-dark: #4f46e5;
            --secondary: #10b981;
            --accent: #8b5cf6;
            --dark: #0f172a;
            --darker: #0c1220;
            --light: #f1f5f9;
            --gray: #94a3b8;
            --success: #10b981;
            --warning: #f59e0b;
            --danger: #ef4444;
            --card-bg: rgba(15, 23, 42, 0.7);
            --card-border: rgba(255, 255, 255, 0.08);
        }
        .stApp {
            background: linear-gradient(135deg, var(--darker) 0%, var(--dark) 50%, #1e293b 100%);
            background-attachment: fixed;
            color: var(--light);
            min-height: 100vh;
        }
        header {
            background: linear-gradient(90deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
            backdrop-filter: blur(10px);
            padding: 1rem 2rem;
            border-bottom: 1px solid var(--card-border);
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.25);
            position: sticky;
            top: 0;
            z-index: 100;
        }
        .header-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .logo {
            display: flex;
            align-items: center;
            gap: 12px;
        }
        .logo-icon {
            background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
            width: 40px;
            height: 40px;
            border-radius: 12px;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
        }
        .logo-text {
            font-size: 1.5rem;
            font-weight: 700;
            background: linear-gradient(90deg, var(--primary) 0%, var(--accent) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .nav-links {
            display: flex;
            gap: 2rem;
        }
        .nav-link {
            color: var(--light);
            text-decoration: none;
            font-weight: 500;
            font-size: 1rem;
            position: relative;
            padding: 0.5rem 0;
            transition: all 0.3s ease;
        }
        .nav-link:after {
            content: '';
            position: absolute;
            bottom: 0;
            left: 0;
            width: 0;
            height: 2px;
            background: var(--primary);
            transition: width 0.3s ease;
        }
        .nav-link:hover {
            color: var(--primary);
        }
        .nav-link:hover:after {
            width: 100%;
        }
        .main-container {
            max-width: 1400px;
            margin: 2rem auto;
            padding: 0 2rem;
        }
        .hero {
            text-align: center;
            padding: 3rem 0;
            margin-bottom: 3rem;
        }
        .hero h1 {
            font-size: 3.5rem;
            font-weight: 800;
            background: linear-gradient(90deg, var(--light) 0%, var(--gray) 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1rem;
            line-height: 1.2;
        }
        .hero p {
            font-size: 1.2rem;
            color: var(--gray);
            max-width: 700px;
            margin: 0 auto 2rem;
            line-height: 1.6;
        }
        .card {
            background: var(--card-bg);
            backdrop-filter: blur(10px);
            border: 1px solid var(--card-border);
            border-radius: 20px;
            padding: 1.5rem;
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.2);
            transition: all 0.3s ease;
            margin-bottom: 1.5rem;
        }
        .card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.3);
            border-color: rgba(99, 102, 241, 0.3);
        }
        .card-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 1.5rem;
        }
        .card-title {
            font-size: 1.5rem;
            font-weight: 700;
            color: var(--light);
            display: flex;
            align-items: center;
            gap: 10px;
        }
        .stButton>button {
            background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%);
            border: none;
            border-radius: 12px;
            color: white;
            padding: 0.8rem 1.5rem;
            font-weight: 600;
            font-size: 1rem;
            cursor: pointer;
            transition: all 0.3s ease;
            box-shadow: 0 4px 15px rgba(99, 102, 241, 0.3);
            width: 100%;
            margin-top: 1rem;
        }
        .stButton>button:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 25px rgba(99, 102, 241, 0.5);
            background: linear-gradient(135deg, var(--primary-dark) 0%, var(--accent) 100%);
        }
        .stTextInput>div>div>input, .stSelectbox>div>div>select, 
        .stDateInput>div>div>input, .stTimeInput>div>div>input,
        .stNumberInput>div>div>input {
            background: rgba(30, 41, 59, 0.7) !important;
            border: 1px solid var(--card-border) !important;
            color: var(--light) !important;
            border-radius: 12px !important;
            padding: 0.8rem 1rem !important;
            font-size: 1rem;
            transition: all 0.3s ease;
        }
        .stTextInput>div>div>input:focus, .stSelectbox>div>div>select:focus, 
        .stDateInput>div>div>input:focus, .stTimeInput>div>div>input:focus,
        .stNumberInput>div>div>input:focus {
            border-color: var(--primary) !important;
            box-shadow: 0 0 0 3px rgba(99, 102, 241, 0.2) !important;
        }
        .stRadio>div {
            flex-direction: row !important;
            gap: 2rem;
        }
        .stRadio>div>label {
            display: flex;
            align-items: center;
            gap: 8px;
            background: rgba(30, 41, 59, 0.5);
            padding: 0.8rem 1.5rem;
            border-radius: 12px;
            border: 1px solid var(--card-border);
            transition: all 0.3s ease;
        }
        .stRadio>div>label:hover {
            border-color: var(--primary);
        }
        .stRadio>div>label[data-baseweb="radio"]>div:first-child {
            background: rgba(30, 41, 59, 0.7) !important;
            border-color: var(--card-border) !important;
        }
        .stRadio>div>label[data-baseweb="radio"]>div:first-child>div {
            background: var(--primary) !important;
        }
        .stSidebar {
            background: linear-gradient(180deg, rgba(15, 23, 42, 0.9) 0%, rgba(15, 23, 42, 0.8) 100%) !important;
            backdrop-filter: blur(10px) !important;
            border-right: 1px solid var(--card-border) !important;
        }
        .stSidebar .sidebar-content {
            padding: 2rem 1.5rem;
        }
        .stSidebar .sidebar-section {
            margin-bottom: 2rem;
        }
        .stSidebar .section-title {
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.1rem;
            font-weight: 600;
            color: var(--primary);
            margin-bottom: 1rem;
            padding-bottom: 0.5rem;
            border-bottom: 1px solid var(--card-border);
        }
        footer {
            background: linear-gradient(90deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.8) 100%);
            backdrop-filter: blur(10px);
            padding: 2rem;
            border-top: 1px solid var(--card-border);
            margin-top: 3rem;
        }
        .footer-content {
            max-width: 1400px;
            margin: 0 auto;
            display: flex;
            flex-wrap: wrap;
            justify-content: space-between;
        }
        .footer-section {
            flex: 1;
            min-width: 250px;
            margin-bottom: 1.5rem;
        }
        .footer-title {
            font-size: 1.2rem;
            font-weight: 600;
            color: var(--light);
            margin-bottom: 1rem;
        }
        .footer-links {
            list-style: none;
        }
        .footer-links li {
            margin-bottom: 0.5rem;
        }
        .footer-links a {
            color: var(--gray);
            text-decoration: none;
            transition: color 0.3s ease;
        }
        .footer-links a:hover {
            color: var(--primary);
        }
        .copyright {
            text-align: center;
            padding-top: 1.5rem;
            border-top: 1px solid var(--card-border);
            margin-top: 1rem;
            color: var(--gray);
        }
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(20px); }
            to { opacity: 1; transform: translateY(0); }
        }
        .animate-fade {
            animation: fadeIn 0.6s ease-out forwards;
        }
        .delay-100 { animation-delay: 0.1s; }
        .delay-200 { animation-delay: 0.2s; }
        .delay-300 { animation-delay: 0.3s; }
        .delay-400 { animation-delay: 0.4s; }
        @media (max-width: 768px) {
            .header-content {
                flex-direction: column;
                gap: 1rem;
            }
            .nav-links {
                gap: 1rem;
                flex-wrap: wrap;
                justify-content: center;
            }
            .hero h1 {
                font-size: 2.5rem;
            }
            .footer-section {
                min-width: 100%;
            }
        }
    </style>
    """
    st.markdown(pro_css, unsafe_allow_html=True)
    
    # Hero Section (removed title and description)
    st.markdown("""
    <div class="main-container">
        <div class="hero animate-fade">
        </div>
    """, unsafe_allow_html=True)

# تبدیل تاریخ به فرمت آگاه از منطقه زمانی
def make_timezone_aware(dt, timezone_str):
    tz = pytz.timezone(timezone_str)
    if dt.tzinfo is None or dt.tzinfo.utcoffset(dt) is None:
        return tz.localize(dt)
    else:
        return dt.astimezone(tz)

# دانلود داده‌ها
def download_filtered_data(symbol, start_datetime, end_datetime, interval, timezone=None):
    start_dt = pd.to_datetime(start_datetime)
    end_dt = pd.to_datetime(end_datetime)

    if timezone:
        start_dt = make_timezone_aware(start_dt, timezone)
        end_dt = make_timezone_aware(end_dt, timezone)

    data = yf.download(
        symbol,
        start=start_dt.date().isoformat(),
        end=(end_dt + pd.Timedelta(days=1)).date().isoformat(),
        interval=interval
    )

    data.index = pd.to_datetime(data.index)

    if timezone:
        if data.index.tz is None:
            data.index = data.index.tz_localize('UTC')
        data.index = data.index.tz_convert(timezone)
    else:
        data.index = data.index.tz_localize(None)

    data = data.loc[(data.index >= start_dt) & (data.index <= end_dt)]

    data.rename(columns={
        'Open': f'Open_{symbol}',
        'High': f'High_{symbol}',
        'Low': f'Low_{symbol}',
        'Close': f'Close_{symbol}',
        'Volume': f'Volume_{symbol}'
    }, inplace=True)

    if 'Adj Close' in data.columns:
        data.drop(columns=['Adj Close'], inplace=True)

    return data

# یافتن قله‌ها و دره‌ها
def find_peaks_valleys(residuals, window=5):
    peaks = []
    valleys = []
    
    for i in range(window, len(residuals) - window):
        if all(residuals[i] > residuals[i-j] for j in range(1, window+1)) and \
           all(residuals[i] > residuals[i+j] for j in range(1, window+1)):
            peaks.append(i)
        
        if all(residuals[i] < residuals[i-j] for j in range(1, window+1)) and \
           all(residuals[i] < residuals[i+j] for j in range(1, window+1)):
            valleys.append(i)
    
    return peaks, valleys

# محاسبه روند با ویولت
def compute_wavelet_trend(signal):
    wavelets = ['db4', 'sym5', 'coif3', 'bior3.3', 'haar']
    results = {}
    
    if len(signal) > 10:
        wavelet_temp = pywt.Wavelet('db4')
        max_lvl = pywt.dwt_max_level(len(signal), wavelet_temp.dec_len)
        level = max(1, min(max_lvl - 1, 5))
    else:
        level = 1

    for wavelet_name in wavelets:
        try:
            padded_length = 2**level - len(signal) % 2**level if len(signal) % 2**level != 0 else 0
            signal_padded = pad(signal, (0, padded_length), 'symmetric')
            
            coeffs = pywt.wavedec(signal_padded, wavelet_name, mode='periodization', level=level)
            
            uthresh_coeffs = []
            for c in coeffs[1:]:
                if len(c) > 0:
                    sigma = np.median(np.abs(c)) / 0.6745
                    uthresh = sigma * np.sqrt(2 * np.log(len(c)))
                    uthresh_coeffs.append(uthresh)
                else:
                    uthresh_coeffs.append(0)
            
            coeffs_thresh = [coeffs[0]]
            for i in range(1, len(coeffs)):
                coeffs_thresh.append(pywt.threshold(coeffs[i], uthresh_coeffs[i-1], mode='soft'))
            
            trend_padded = pywt.waverec(coeffs_thresh, wavelet_name, mode='periodization')
            trend = trend_padded[:len(signal)]
            
            mse = mean_squared_error(signal, trend)
            results[wavelet_name] = mse
            
        except Exception:
            results[wavelet_name] = float('inf')
    
    best_wavelet = min(results, key=results.get) if results else 'db4'
    
    try:
        padded_length = 2**level - len(signal) % 2**level if len(signal) % 2**level != 0 else 0
        signal_padded = pad(signal, (0, padded_length), 'symmetric')
        
        coeffs = pywt.wavedec(signal_padded, best_wavelet, mode='periodization', level=level)
        
        uthresh_coeffs = []
        for c in coeffs[1:]:
            if len(c) > 0:
                sigma = np.median(np.abs(c)) / 0.6745
                uthresh = sigma * np.sqrt(2 * np.log(len(c)))
                uthresh_coeffs.append(uthresh)
            else:
                uthresh_coeffs.append(0)
        
        coeffs_thresh = [coeffs[0]]
        for i in range(1, len(coeffs)):
            coeffs_thresh.append(pywt.threshold(coeffs[i], uthresh_coeffs[i-1], mode='soft'))
        
        trend_padded = pywt.waverec(coeffs_thresh, best_wavelet, mode='periodization')
        trend = trend_padded[:len(signal)]
        
        return trend, best_wavelet, level
    except Exception:
        return signal, 'db4', 1

# محاسبه اکسترمم‌ها و میانگین‌ها
def compute_extrema_and_averages(residuals, method_type):
    if method_type == 'kalman':
        peaks_idx = argrelextrema(residuals, np.greater, order=3)[0]
        valleys_idx = argrelextrema(residuals, np.less, order=3)[0]
    else:
        peaks_idx, valleys_idx = find_peaks_valleys(residuals, window=5)

    peaks = residuals[peaks_idx]
    valleys = residuals[valleys_idx]

    mean_peak = np.mean(peaks) if len(peaks) > 0 else 0
    mean_valley = np.mean(valleys) if len(valleys) > 0 else 0

    high_peaks = [p for p in peaks if p > mean_peak] if len(peaks) > 0 else []
    low_valleys = [v for v in valleys if v < mean_valley] if len(valleys) > 0 else []

    mean_high_peak = np.mean(high_peaks) if len(high_peaks) > 0 else mean_peak
    mean_low_valley = np.mean(low_valleys) if len(low_valleys) > 0 else mean_valley
    
    filtered_peaks_idx = [i for i in peaks_idx if residuals[i] > mean_peak]
    filtered_valleys_idx = [i for i in valleys_idx if residuals[i] < mean_valley]
    filtered_peaks = residuals[filtered_peaks_idx]
    filtered_valleys = residuals[filtered_valleys_idx]

    return {
        'peaks_idx': peaks_idx,
        'valleys_idx': valleys_idx,
        'peaks': peaks,
        'valleys': valleys,
        'mean_peak': mean_peak,
        'mean_valley': mean_valley,
        'mean_high_peak': mean_high_peak,
        'mean_low_valley': mean_low_valley,
        'filtered_peaks_idx': filtered_peaks_idx,
        'filtered_valleys_idx': filtered_valleys_idx,
        'filtered_peaks': filtered_peaks,
        'filtered_valleys': filtered_valleys
    }

# تابع اصلی تحلیل
def run_analysis(symbol, start_date, start_hour, start_minute, end_date, end_hour, end_minute, interval, 
                 initial_state_mean, auto_initial_state, show_main, show_residual, show_orig_candle, show_filt_candle,
                 method, uploaded_file=None):
    
    if uploaded_file is not None:
        try:
            with st.spinner('Processing uploaded data...'):
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                
                data = pd.read_csv(uploaded_file, index_col=0, parse_dates=True)
                data.index = pd.to_datetime(data.index, errors='coerce')
                data = data[~data.index.isna()]
                
                close_col = None
                for col in data.columns:
                    if 'close' in col.lower():
                        close_col = col
                        break
                
                if close_col is None:
                    st.error("❌ Close column not found in file")
                    return
                    
                for col in data.columns:
                    data[col] = pd.to_numeric(data[col], errors='coerce')
                data.dropna(subset=[close_col], inplace=True)
                
                st.success("✅ Data processed successfully")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}")
            return
    else:
        start_time = f"{start_hour.zfill(2)}:{start_minute.zfill(2)}"
        end_time = f"{end_hour.zfill(2)}:{end_minute.zfill(2)}"
        start_datetime = f"{start_date} {start_time}"
        end_datetime = f"{end_date} {end_time}"
        
        timezone = "Asia/Tehran"

        try:
            with st.spinner(f'Downloading {symbol} data from Yahoo Finance...'):
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete + 1)
                
                data = download_filtered_data(symbol, start_datetime, end_datetime, interval, timezone)
                close_col = f'Close_{symbol}'
                st.success(f"✅ Data for {symbol} downloaded successfully")
        except Exception as e:
            st.error(f"Error downloading data: {e}")
            return

    # مقداردهی اولیه initial_state_mean برای هر دو حالت Yahoo و آپلود فایل
    if (method == 'Kalman' or method == 'Kalman+Wavelet') and len(data) > 0:
        if auto_initial_state:
            initial_state_mean = data[close_col].iloc[0]
        # else: مقدار initial_state_mean همان مقدار ورودی کاربر باقی می‌ماند

    # تحلیل بر اساس روش انتخابی
    analysis_method = method
    with st.spinner(f'Running {analysis_method} analysis...'):
        progress_bar = st.progress(0)
        for percent_complete in range(100):
            time.sleep(0.02)
            progress_bar.progress(percent_complete + 1)
        
        if method == 'Kalman':
            try:
                observations = data[close_col].values.reshape(-1, 1)
                kf = KalmanFilter(
                    transition_matrices=[[1.0, 1.0], [0.0, 1.0]],
                    observation_matrices=[[1.0, 0.0]],
                    initial_state_mean=[initial_state_mean, 0.0],
                    n_dim_state=2,
                    n_dim_obs=1
                )
                kf = kf.em(observations, n_iter=70)
                state_means, _ = kf.filter(observations)
                filtered_close = state_means[:, 0]
                data['Filtered_Close'] = filtered_close
                data['Residual'] = data[close_col] - data['Filtered_Close']
                method_name = 'Kalman'
            except Exception as e:
                st.error(f"Error applying Kalman filter: {e}")
                return
        
        elif method == 'Wavelet':
            try:
                signal = data[close_col].values.flatten()
                trend, best_wavelet, level = compute_wavelet_trend(signal)
                data['Filtered_Close'] = trend
                data['Residual'] = data[close_col] - data['Filtered_Close']
                method_name = f'Wavelet ({best_wavelet}, level: {level})'
            except Exception as e:
                st.error(f"Error in wavelet analysis: {e}")
                return
                
        elif method == 'Kalman+Wavelet':
            try:
                observations = data[close_col].values.reshape(-1, 1)
                kf = KalmanFilter(
                    transition_matrices=[[1.0, 1.0], [0.0, 1.0]],
                    observation_matrices=[[1.0, 0.0]],
                    initial_state_mean=[initial_state_mean, 0.0],
                    n_dim_state=2,
                    n_dim_obs=1
                )
                kf = kf.em(observations, n_iter=70)
                state_means, _ = kf.filter(observations)
                kalman_filtered = state_means[:, 0]
                data['Kalman_Filtered'] = kalman_filtered
                data['Kalman_Residual'] = data[close_col] - kalman_filtered
                
                signal = data[close_col].values.flatten()
                wavelet_trend, best_wavelet, level = compute_wavelet_trend(signal)
                data['Wavelet_Filtered'] = wavelet_trend
                data['Wavelet_Residual'] = data[close_col] - wavelet_trend
                
                method_name = 'Kalman + Wavelet'
            except Exception as e:
                st.error(f"Error in combined analysis: {e}")
                return

    # نمایش نمودارها
    if show_main:
        with st.expander("📈 Price Analysis", expanded=True):
            fig_main = go.Figure()
            fig_main.add_trace(go.Scatter(
                x=data.index,
                y=data[close_col],
                mode='lines',
                name='Actual Price',
                line=dict(color='#6366f1', width=1.5)
            ))
            
            if method == 'Kalman+Wavelet':
                fig_main.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Kalman_Filtered'],
                    mode='lines',
                    name='Trend (Kalman)',
                    line=dict(color='#10b981', width=2.5)
                ))
                fig_main.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Wavelet_Filtered'],
                    mode='lines',
                    name='Trend (Wavelet)',
                    line=dict(color='#8b5cf6', width=2.5)
                ))
            else:
                fig_main.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Filtered_Close'],
                    mode='lines',
                    name=f'Trend ({method_name})',
                    line=dict(color='#10b981', width=2.5)
                ))
                
            fig_main.update_layout(
                title=f'Price Chart for {symbol}',
                xaxis_title='Date',
                yaxis_title='Price',
                height=500,
                template='plotly_dark',
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )
            st.plotly_chart(fig_main, use_container_width=True)

    if show_residual:
        with st.expander("📊 Residual Analysis", expanded=True):
            if method == 'Kalman+Wavelet':
                fig_res = make_subplots(specs=[[{"secondary_y": True}]])
                
                kalman_residuals = data['Kalman_Residual'].values
                kalman_results = compute_extrema_and_averages(kalman_residuals, 'kalman')
                
                fig_res.add_trace(go.Scatter(
                    x=data.index,
                    y=kalman_residuals,
                    mode='lines',
                    name='Kalman Residual',
                    line=dict(color='#10b981', width=2)
                ), secondary_y=False)
                
                if len(kalman_results['filtered_peaks_idx']) > 0:
                    fig_res.add_trace(go.Scatter(
                        x=data.index[kalman_results['filtered_peaks_idx']],
                        y=kalman_results['filtered_peaks'],
                        mode='markers',
                        name='Kalman Filtered Peak',
                        marker=dict(color='#10b981', size=10, symbol='triangle-up')
                    ), secondary_y=False)
                
                if len(kalman_results['filtered_valleys_idx']) > 0:
                    fig_res.add_trace(go.Scatter(
                        x=data.index[kalman_results['filtered_valleys_idx']],
                        y=kalman_results['filtered_valleys'],
                        mode='markers',
                        name='Kalman Filtered Valley',
                        marker=dict(color='#10b981', size=10, symbol='triangle-down')
                    ), secondary_y=False)
                
                fig_res.add_hline(
                    y=kalman_results['mean_peak'], 
                    line=dict(color='#10b981', width=2, dash='dash'),
                    annotation_text=f"Kalman Peak Avg: {kalman_results['mean_peak']:.4f}",
                    secondary_y=False
                )
                
                wavelet_residuals = data['Wavelet_Residual'].values
                wavelet_results = compute_extrema_and_averages(wavelet_residuals, 'wavelet')
                
                fig_res.add_trace(go.Scatter(
                    x=data.index,
                    y=wavelet_residuals,
                    mode='lines',
                    name='Wavelet Residual',
                    line=dict(color='#8b5cf6', width=2)
                ), secondary_y=True)
                
                if len(wavelet_results['filtered_peaks_idx']) > 0:
                    fig_res.add_trace(go.Scatter(
                        x=data.index[wavelet_results['filtered_peaks_idx']],
                        y=wavelet_results['filtered_peaks'],
                        mode='markers',
                        name='Wavelet Filtered Peak',
                        marker=dict(color='#8b5cf6', size=10, symbol='triangle-up')
                    ), secondary_y=True)
                
                if len(wavelet_results['filtered_valleys_idx']) > 0:
                    fig_res.add_trace(go.Scatter(
                        x=data.index[wavelet_results['filtered_valleys_idx']],
                        y=wavelet_results['filtered_valleys'],
                        mode='markers',
                        name='Wavelet Filtered Valley',
                        marker=dict(color='#8b5cf6', size=10, symbol='triangle-down')
                    ), secondary_y=True)
                
                fig_res.add_hline(
                    y=wavelet_results['mean_peak'], 
                    line=dict(color='#8b5cf6', width=2, dash='dash'),
                    annotation_text=f"Wavelet Peak Avg: {wavelet_results['mean_peak']:.4f}",
                    secondary_y=True
                )
                
                fig_res.update_layout(
                    title=f'Combined Residual Analysis',
                    xaxis_title='Date',
                    height=600,
                    template='plotly_dark',
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                fig_res.update_yaxes(title_text="Kalman Residual", secondary_y=False)
                fig_res.update_yaxes(title_text="Wavelet Residual", secondary_y=True)
                
                st.plotly_chart(fig_res, use_container_width=True)
                
            else:
                residuals = data['Residual'].values
                results = compute_extrema_and_averages(residuals, method.lower())
                
                RESIDUAL_COLOR = '#6366f1'
                FILTERED_PEAK_COLOR = '#10b981'
                FILTERED_VALLEY_COLOR = '#ec4899'
                MEAN_PEAK_COLOR = '#10b981'
                MEAN_VALLEY_COLOR = '#ec4899'
                HIGH_PEAK_COLOR = '#06b6d4'
                LOW_VALLEY_COLOR = '#f97316'
                ZERO_LINE_COLOR = '#ffffff'

                fig_res = go.Figure()
                fig_res.add_trace(go.Scatter(
                    x=data.index,
                    y=data['Residual'],
                    mode='lines',
                    name='Residual',
                    line=dict(color=RESIDUAL_COLOR, width=2)
                ))
                fig_res.add_shape(type='line', x0=data.index[0], x1=data.index[-1], y0=0, y1=0, 
                                line=dict(color=ZERO_LINE_COLOR, dash='dot', width=1.5))
                
                if len(results['filtered_peaks_idx']) > 0:
                    fig_res.add_trace(go.Scatter(
                        x=data.index[results['filtered_peaks_idx']],
                        y=results['filtered_peaks'],
                        mode='markers',
                        name='Filtered Peak',
                        marker=dict(color=FILTERED_PEAK_COLOR, size=10, symbol='triangle-up')
                    ))
                
                if len(results['filtered_valleys_idx']) > 0:
                    fig_res.add_trace(go.Scatter(
                        x=data.index[results['filtered_valleys_idx']],
                        y=results['filtered_valleys'],
                        mode='markers',
                        name='Filtered Valley',
                        marker=dict(color=FILTERED_VALLEY_COLOR, size=10, symbol='triangle-down')
                    ))
                
                fig_res.add_hline(
                    y=results['mean_peak'], 
                    line=dict(color=MEAN_PEAK_COLOR, width=2.5, dash='dash'),
                    annotation_text=f"Primary Peak Avg: {results['mean_peak']:.4f}"
                )
                
                fig_res.add_hline(
                    y=results['mean_valley'], 
                    line=dict(color=MEAN_VALLEY_COLOR, width=2.5, dash='dash'),
                    annotation_text=f"Primary Valley Avg: {results['mean_valley']:.4f}"
                )
                
                fig_res.update_layout(
                    title=f'Residual Analysis ({method_name})',
                    xaxis_title='Date',
                    yaxis_title='Residual Value',
                    height=600,
                    template='plotly_dark'
                )
                st.plotly_chart(fig_res, use_container_width=True)

    if show_orig_candle:
        with st.expander("🟢 Original Candlestick", expanded=True):
            open_col = None
            high_col = None
            low_col = None
            
            for col in data.columns:
                col_lower = col.lower()
                if 'open' in col_lower:
                    open_col = col
                elif 'high' in col_lower:
                    high_col = col
                elif 'low' in col_lower:
                    low_col = col
            
            if open_col is None:
                open_col = close_col
            if high_col is None:
                high_col = close_col
            if low_col is None:
                low_col = close_col

            fig_candle_orig = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data[open_col],
                high=data[high_col],
                low=data[low_col],
                close=data[close_col],
                increasing_line_color='#10b981',
                decreasing_line_color='#ef4444',
                name='Original'
            )])
            fig_candle_orig.update_layout(
                title=f'Original Candlestick for {symbol}',
                xaxis_title='Date',
                yaxis_title='Price',
                height=500,
                template='plotly_dark'
            )
            st.plotly_chart(fig_candle_orig, use_container_width=True)

    if show_filt_candle and method != 'Kalman+Wavelet':
        with st.expander("🔵 Filtered Candlestick", expanded=True):
            open_col = None
            high_col = None
            low_col = None
            
            for col in data.columns:
                col_lower = col.lower()
                if 'open' in col_lower:
                    open_col = col
                elif 'high' in col_lower:
                    high_col = col
                elif 'low' in col_lower:
                    low_col = col
            
            if open_col is None:
                open_col = close_col
            if high_col is None:
                high_col = close_col
            if low_col is None:
                low_col = close_col

            fig_candle_filt = go.Figure(data=[go.Candlestick(
                x=data.index,
                open=data[open_col],
                high=data[high_col],
                low=data[low_col],
                close=data['Filtered_Close'] if method != 'Kalman+Wavelet' else data['Kalman_Filtered'],
                increasing_line_color='#0ea5e9',
                decreasing_line_color='#f59e0b',
                name='Filtered'
            )])
            fig_candle_filt.update_layout(
                title=f'Filtered Candlestick for {symbol}',
                xaxis_title='Date',
                yaxis_title='Price',
                height=500,
                template='plotly_dark'
            )
            st.plotly_chart(fig_candle_filt, use_container_width=True)

# تابع نمایش ویجت‌های TradingView
def show_tradingview_widgets():
    tradingview_html = '''
    <div class="tradingview-widget-container">
      <div class="charts-grid">
        <div class="chart-cell" id="tradingview_30m"><div class="resize-handle">&#8690;</div></div>
        <div class="chart-cell" id="tradingview_15m"><div class="resize-handle">&#8690;</div></div>
        <div class="chart-cell" id="tradingview_3m"><div class="resize-handle">&#8690;</div></div>
        <div class="chart-cell" id="tradingview_1m"><div class="resize-handle">&#8690;</div></div>
      </div>
      <script src="https://s3.tradingview.com/tv.js"></script>
      <script>
        const widgetConfig = {
          width: "100%",
          height: "100%",
          autosize: true,
          symbol: "OANDA:XAUUSD",
          timezone: "Asia/Tehran",
          theme: "dark",
          style: "1",
          locale: "fa_IR",
          toolbar_bg: "#131722",
          enable_publishing: true,
          allow_symbol_change: true,
          withdateranges: true,
          hide_side_toolbar: false,
          details: true,
          hotlist: true,
          calendar: true,
          show_popup_button: true,
          popup_width: "1000",
          popup_height: "650",
          save_image: true,
          show_chart_property_settings: true,
          show_symbol_logo: true,
          hideideas: false,
          hide_volume: true,
          watchlist: [
            "OANDA:XAUUSD",
            "OANDA:EURUSD",
            "OANDA:GBPUSD",
            "OANDA:USDJPY",
            "OANDA:USDCAD",
            "OANDA:AUDUSD",
            "OANDA:USDCHF"
          ],
          supported_resolutions: [
            "1",   // 1m
            "3",   // 3m
            "15",  // 15m
            "30"   // 30m
          ]
        };

        function createWidget(containerId, interval) {
          new TradingView.widget({
            ...widgetConfig,
            interval,
            container_id: containerId
          });
        }

        [
          {id: "tradingview_30m", interval: "30"},
          {id: "tradingview_15m", interval: "15"},
          {id: "tradingview_3m", interval: "3"},
          {id: "tradingview_1m", interval: "1"}
        ].forEach(cfg => createWidget(cfg.id, cfg.interval));

        // Manual resizing logic for chart cells + double click to reset
        document.querySelectorAll('.chart-cell').forEach(cell => {
          const handle = cell.querySelector('.resize-handle');
          if (!handle) return;
          let isResizing = false, lastX = 0, lastY = 0, startW = 0, startH = 0;

          // ذخیره اندازه اولیه هر سلول
          if (!cell.dataset.initialWidth || !cell.dataset.initialHeight) {
            const computed = window.getComputedStyle(cell);
            cell.dataset.initialWidth = computed.width;
            cell.dataset.initialHeight = computed.height;
          }

          handle.addEventListener('mousedown', e => {
            e.preventDefault();
            isResizing = true;
            lastX = e.clientX;
            lastY = e.clientY;
            startW = cell.offsetWidth;
            startH = cell.offsetHeight;
            document.body.style.userSelect = 'none';
          });

          const mouseMove = e => {
            if (!isResizing) return;
            const dx = e.clientX - lastX;
            const dy = e.clientY - lastY;
            cell.style.width = Math.max(200, startW + dx) + 'px';
            cell.style.height = Math.max(150, startH + dy) + 'px';
            cell.style.minWidth = '100px';
            cell.style.minHeight = '100px';
            cell.style.maxWidth = '100vw';
            cell.style.maxHeight = '100vh';
            cell.style.flex = 'none';
            cell.style.position = 'relative';
            cell.style.zIndex = 100;
          };

          const mouseUp = () => {
            if (isResizing) {
              isResizing = false;
              document.body.style.userSelect = '';
            }
          };

          window.addEventListener('mousemove', mouseMove);
          window.addEventListener('mouseup', mouseUp);

          // Double click to reset size
          handle.addEventListener('dblclick', e => {
            e.preventDefault();
            cell.style.width = cell.dataset.initialWidth;
            cell.style.height = cell.dataset.initialHeight;
            cell.style.minWidth = '';
            cell.style.minHeight = '';
            cell.style.maxWidth = '';
            cell.style.maxHeight = '';
            cell.style.flex = '';
            cell.style.position = '';
            cell.style.zIndex = '';
          });
        });
      </script>
      <style>
        html, body {
          height: 100%;
          margin: 0;
          padding: 0;
          background: #181c25;
        }
        .tradingview-widget-container {
          width: 100vw;
          min-height: 100vh;
          margin: 0;
          padding: 0;
          background: #181c25;
          display: flex;
          align-items: center;
          justify-content: center;
          height: 100vh;
        }
        .charts-grid {
          display: grid;
          grid-template-columns: repeat(2, 1fr);
          grid-template-rows: repeat(2, 1fr);
          width: 100vw;
          height: 90vh;
          max-width: 100vw;
          max-height: 100vh;
          margin: 0;
          padding: 0;
          background: #181c25;
          align-items: stretch;
          justify-items: stretch;
        }
        .chart-cell {
          background: #181c25;
          overflow: hidden;
          min-width: 100px;
          min-height: 100px;
          width: 100%;
          height: 100%;
          display: flex;
          position: relative;
          resize: both;
          transition: width 0.2s, height 0.2s;
        }
        .chart-cell::-webkit-resizer {
          background: #444;
        }
        .resize-handle {
          position: absolute;
          width: 18px;
          height: 18px;
          right: 2px;
          bottom: 2px;
          background: rgba(255,255,255,0.15);
          border-radius: 3px;
          cursor: se-resize;
          z-index: 10;
          display: flex;
          align-items: flex-end;
          justify-content: flex-end;
          font-size: 16px;
          color: #aaa;
          user-select: none;
        }
        @media (max-width: 1100px) {
          .charts-grid {
            grid-template-columns: 1fr;
            grid-template-rows: repeat(4, 1fr);
            width: 100vw;
            height: 100vh;
          }
        }
        .chart-cell > div,
        .chart-cell iframe,
        .chart-cell .tradingview-widget-container__widget {
          width: 100% !important;
          height: 100% !important;
          min-width: 0 !important;
          min-height: 0 !important;
          max-width: 100% !important;
          max-height: 100% !important;
          display: block;
        }
      </style>
    </div>
    '''
    components.html(tradingview_html, height=800, scrolling=True)

# رابط کاربری Streamlit
def main():
    st.set_page_config(layout="wide")
    inject_pro_style()
    
    # Sidebar
    with st.sidebar:
        st.markdown("""
        <div class="sidebar-content">
            <div class="sidebar-section">
                <div class="section-title">
                    <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                        <circle cx="12" cy="12" r="10"></circle>
                        <line x1="12" y1="8" x2="12" y2="12"></line>
                        <line x1="12" y1="16" x2="12.01" y2="16"></line>
                    </svg>
                    DATA SOURCE
                </div>
        """, unsafe_allow_html=True)
        
        data_source = st.radio("Select Data Source", ["Yahoo Finance", "Upload CSV"], index=0, label_visibility="collapsed")
        
        uploaded_file = None
        if data_source == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV File", type=['csv'])
        else:
            symbols = [
                'GC=F', 'SI=F', 'EURUSD=X', 'GBPUSD=X', 'JPY=X', '^DJI',
                'BTC-USD', 'ETH-USD', 'XRP-USD', 'ADA-USD', 'LTC-USD', 'DOGE-USD',
                'BNB-USD', 'SOL-USD', 'DOT-USD', 'AVAX-USD', 'MATIC-USD', 'LINK-USD'
            ]
            symbol = st.selectbox('Symbol', symbols, index=0)
            interval = st.selectbox('Interval', ['15m', '30m', '1h', '2h', '4h'], index=1)
            
            st.markdown("""
            <div class="section-title" style="margin-top: 20px;">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <rect x="3" y="4" width="18" height="18" rx="2" ry="2"></rect>
                    <line x1="16" y1="2" x2="16" y2="6"></line>
                    <line x1="8" y1="2" x2="8" y2="6"></line>
                    <line x1="3" y1="10" x2="21" y2="10"></line>
                </svg>
                TIME RANGE
            </div>
            """, unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input('Start Date', value=datetime.today())
                start_hour = st.selectbox('Start Hour', [str(i).zfill(2) for i in range(24)], index=0)
                start_minute = st.selectbox('Start Minute', ['00', '15', '30', '45'], index=0)
            with col2:
                end_date = st.date_input('End Date', value=datetime.today())
                end_hour = st.selectbox('End Hour', [str(i).zfill(2) for i in range(24)], index=datetime.now().hour % 24)
                end_minute = st.selectbox('End Minute', ['00', '15', '30', '45'], index=1)
        
        st.markdown("""
            <div class="section-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"></path>
                    <polyline points="3.27 6.96 12 12.01 20.73 6.96"></polyline>
                    <line x1="12" y1="22.08" x2="12" y2="12"></line>
                </svg>
                ANALYSIS METHOD
            </div>
        """, unsafe_allow_html=True)
        
        method = st.selectbox('Method', ['Kalman', 'Wavelet', 'Kalman+Wavelet'], index=0, label_visibility="collapsed")
        
        st.markdown("""
            <div class="section-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <circle cx="12" cy="12" r="3"></circle>
                    <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-2 2 2 2 0 0 1-2-2v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1-2-2 2 2 0 0 1 2-2h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33H9a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 2-2 2 2 0 0 1 2 2v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 2 2 2 2 0 0 1-2 2h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
                </svg>
                FILTER SETTINGS
            </div>
        """, unsafe_allow_html=True)
        
        auto_initial = st.checkbox('Auto Initial State', value=True)
        initial_value = st.number_input('Initial Value', value=0.0, disabled=auto_initial)
        
        st.markdown("""
            <div class="section-title">
                <svg xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline>
                </svg>
                VISUALIZATION
            </div>
        """, unsafe_allow_html=True)
        
        chart_options = st.multiselect(
            'Select Charts to Display',
            ['Main Chart', 'Residual Chart', 'Candles', 'Filtered Candles', 'TradingView'],
            default=['Residual Chart', 'TradingView'],
            label_visibility="collapsed"
        )
        
        run_button = st.button("RUN ADVANCED ANALYSIS", type="primary", use_container_width=True)
        
        st.markdown("</div></div>", unsafe_allow_html=True)
    
    # Main content
    if run_button:
        if data_source == "Upload CSV" and uploaded_file is None:
            st.error("Please upload a CSV file")
            return
            
        if data_source == "Yahoo Finance":
            run_analysis(
                symbol,
                start_date.strftime('%Y-%m-%d'),
                start_hour,
                start_minute,
                end_date.strftime('%Y-%m-%d'),
                end_hour,
                end_minute,
                interval,
                initial_value,
                auto_initial,
                'Main Chart' in chart_options,
                'Residual Chart' in chart_options,
                'Candles' in chart_options,
                'Filtered Candles' in chart_options,
                method,
                uploaded_file=None
            )
        else:
            run_analysis(
                "UPLOADED",
                None, None, None, None, None, None, None,
                initial_value,
                auto_initial,
                'Main Chart' in chart_options,
                'Residual Chart' in chart_options,
                'Candles' in chart_options,
                'Filtered Candles' in chart_options,
                method,
                uploaded_file=uploaded_file
            )
        # نمایش ویجت TradingView اگر انتخاب شده باشد
        if 'TradingView' in chart_options:
            with st.expander("📊 TradingView Multi-Chart", expanded=True):
                show_tradingview_widgets()

if __name__ == "__main__":
    main()
