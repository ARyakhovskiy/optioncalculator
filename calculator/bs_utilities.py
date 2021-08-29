import numpy as np
import scipy.stats as stats


def time_averaged_volatility(volatility_model, T):
    return

def forward_price(current_price, T, r, div_values):
    return

def max_asset_price(strike_price, volatility, probability_of_exceeding, r, T):
    s_max = strike_price*4
    return s_max


def black_scholes_formula(S, T, r, K, sigma, option_type='call', option_style='EU', dividends=[], dividend_dates=[]):
    d1 = (np.log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    #if len(dividends) > 0:
    #    divs = np.array(dividends)
    #    div_dates = np.array(dividend_dates)
    #    S = S - np.sum(divs*np.exp(-r*div_dates))

    if option_type == 'call':
        V = S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)
    elif option_type == 'put':
        V = -S * stats.norm.cdf(-d1) + K * np.exp(-r * T) * stats.norm.cdf(-d2)
    else:
        print('Unknown option type')
        raise ValueError
    if option_style == 'US':
        #p = payoff(np.array(S), K, option_type)
        if option_type == 'call':
            V = max(V, S-K)
        else:
            V = max(V, K-S)
    return np.nan_to_num(V)


def implied_volatility(market_price, strike, expiry, asset, int_rate, error):
    vol = 0.4
    vega = 0.0
    dv = error+1
    if market_price == 0.0:
        return 0.0, 0.0
    while np.absolute(dv) > error:
        d1 = np.log(asset/strike) + (int_rate+0.5*vol*vol)*expiry
        d1 = d1/(vol*np.sqrt(expiry))
        d2 = d1 - vol*np.sqrt(expiry)
        price_error = asset*stats.norm.cdf(d1) - strike*np.exp(int_rate*expiry)*stats.norm.cdf(d2)-market_price
        vega = asset * np.sqrt(0.5*expiry/np.pi) * np.exp(-0.5*d1*d1)
        dv = price_error/vega
        vol = vol-dv
    iv = vol
    return np.nan_to_num(iv), np.nan_to_num(vega)


def iv_array(V, strike, expiry, S, int_rate, error):
    iv = np.zeros(V.shape)
    vega = np.zeros(V.shape)
    nt, ns = V.shape
    for i in range(nt):
        for j in range(ns):
            iv[i, j], vega[i, j] = implied_volatility(V[i, j], strike, expiry, S[j], int_rate, error)
    return iv, vega


def payoff(asset_prices, strike_price, option_type="call"):
    asset_prices = np.array(asset_prices)
    if strike_price < 0:
        raise ValueError('Strike price must be positive')
    if option_type == 'call':
        option_payoff = np.array([max(s-strike_price, 0) for s in asset_prices])
    elif option_type == 'put':
        option_payoff = np.array([max(strike_price-s, 0) for s in asset_prices])
    else:
        raise ValueError('Unknown option type')
    return option_payoff


def delta(asset_prices, option_values):
    d = np.zeros(option_values.shape)
    ns = len(asset_prices)
    for i in range(1, ns-1):
        d[:, i] = (option_values[:, i+1] - option_values[:, i-1])/(asset_prices[i+1] - asset_prices[i-1])
    d[:, -1] = d[:, -2]
    d[:, 0] = d[:, 1]
    return d


def gamma(asset_prices, option_values):
    g = np.zeros(option_values.shape)
    ns = len(asset_prices)
    for i in range(1, ns-1):
        g[:, i] = 1 / \
                  (asset_prices[i+1]-asset_prices[i]) / \
                  (asset_prices[i]-asset_prices[i-1]) * \
                  (option_values[:, i+1]-2*option_values[:, i]+option_values[:, i-1])
    g[:, 0] = g[:, 1]
    g[:, -1] = g[:, -2]
    return g
