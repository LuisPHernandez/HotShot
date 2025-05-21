import pandas as pd
import numpy as np
import ta

"""Exponential Moving Averages"""
def calculate_emas(df, short=20, long=55):
    df[f'EMA_{short}'] = df['close'].ewm(span=short, adjust=False).mean()
    df[f'EMA_{long}'] = df['close'].ewm(span=long, adjust=False).mean()
    return df


"""ADX Indicator"""
def calculate_adx(df, period=14):
    adx_indicator = ta.trend.ADXIndicator(
        high=df['high'],
        low=df['low'],
        close=df['close'],
        window=period
    )
    df['ADX'] = adx_indicator.adx()
    return df


"""Squeeze Momentum Indicator"""
# Calculate the true range for Keltner Channels
def calculate_true_range(df):
    tr1 = df['high'] - df['low']
    tr2 = abs(df['high'] - df['close'].shift())
    tr3 = abs(df['low'] - df['close'].shift())
    return pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

# Calculate lineal regresion to determine direction and magnitude of a price movement
def linreg(series, length):
    x = np.arange(length)
    def lin_fit(y):
        if y.isnull().any():
            return np.nan
        A = np.vstack([x, np.ones(len(x))]).T
        m, b = np.linalg.lstsq(A, y, rcond=None)[0]
        return m * x[-1] + b
    return series.rolling(length).apply(lin_fit, raw=False)

# Calculate the SMI combining bb (bollinger bands) logic and kc (Keltner Channels) logic
def calculate_squeeze_momentum_indicator(df, bb_length=20, bb_mult=2.0, kc_length=20, kc_mult=1.5, linreg_length=20):
    source = df['close']

    """Bollinger Bands"""
    # Compute the rolling simple average of closes over `bb_length`
    bb_basis = source.rolling(bb_length).mean()
    # Compute the rolling standard deviation of closes over 'bb_length'
    bb_stddev = source.rolling(bb_length).std()
    # Upper Bollinger Band = basis + (multiplier × stddev)
    upper_bb = bb_basis + bb_mult * bb_stddev
    # Lower Bollinger Band = basis − (multiplier × stddev)
    lower_bb = bb_basis - bb_mult * bb_stddev

    """Keltner Channels"""
    # Compute the True Range for each bar
    tr = calculate_true_range(df)
    # Smooth the True Range into an “Average True Range” series over kc_length
    kc_range = tr.rolling(kc_length).mean()
    # Compute the rolling simple average of closes over kc_length
    kc_basis = source.rolling(kc_length).mean()
    # Upper Keltner Channel = basis + (multiplier × ATR)
    upper_kc = kc_basis + kc_mult * kc_range
    # Lower Keltner Channel = basis - (multiplier × ATR)
    lower_kc = kc_basis - kc_mult * kc_range

    """Squeeze conditions"""
    # Squeeze is “On” when Bollinger sits entirely inside Keltner (low volatility)
    sqz_on = (lower_bb > lower_kc) & (upper_bb < upper_kc)
    # Squeeze is “Off” when Bollinger sits entirely outside Keltner (volatility expansion)
    sqz_off = (lower_bb < lower_kc) & (upper_bb > upper_kc)
    # Neither on nor off (transition zone)
    no_sqz = ~(sqz_on | sqz_off)

    """Momentum via linear regression"""
    # Highest high over the last linreg_length bars
    highest_high = df['high'].rolling(linreg_length).max()
    # Lowest low over the last linreg_length bars
    lowest_low = df['low'].rolling(linreg_length).min()
    # Midpoint in the highest high to lowest low range
    avg_h = (highest_high + lowest_low) / 2
    # Simple average of close over linreg_length
    sma_close = source.rolling(linreg_length).mean()
    # Compute deviation = price minus average of (range midpoint + SMA)/2
    val_input = source - ((avg_h + sma_close) / 2)
    # Feed the deviations into linreg helper function to get one momentum value per bar
    val = linreg(val_input, linreg_length)

    # Assign to DataFrame
    df['squeeze_on'] = sqz_on
    df['squeeze_off'] = sqz_off
    df['no_squeeze'] = no_sqz
    df['squeeze_momentum'] = val

    # Color logic (LazyBear bcolor)
    prev = df['squeeze_momentum'].shift(1)
    df['squeeze_color'] = np.where(
        df['squeeze_momentum'] > 0,
        np.where(df['squeeze_momentum'] > prev, '#00FF00', '#008000'),
        np.where(df['squeeze_momentum'] < prev, '#FF0000', '#880000')
    )

    return df
