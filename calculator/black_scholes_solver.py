import numpy as np
import scipy.stats as stats


def black_scholes_formula(spot_price, strike_price, time_till_maturity, volatility, interest_rate, option_type='call',
                          option_style='EU', dividends=np.array([]), dividend_dates=np.array([])):

    d1 = (np.log(spot_price/strike_price) + (interest_rate+0.5*volatility*volatility)*time_till_maturity) / \
         (volatility*np.sqrt(time_till_maturity))
    d2 = d1 - volatility*np.sqrt(time_till_maturity)

    if option_type == 'call':
        fair_value = spot_price*stats.norm.cdf(d1) - strike_price*np.exp(-interest_rate*time_till_maturity) \
                     * stats.norm.cdf(d2)
    elif option_type == 'put':
        fair_value = -spot_price*stats.norm.cdf(-d1) + strike_price*np.exp(-interest_rate*time_till_maturity)\
                     * stats.norm.cdf(-d2)
    else:
        print('Unknown option type')
        raise ValueError

    if option_style == 'US':
        fair_value = np.array([fair_value])
        if option_type == 'call':
            fair_value = np.maximum(fair_value, np.full(len(fair_value), spot_price-strike_price))
        else:
            fair_value = np.maximum(fair_value, np.full(len(fair_value), strike_price-spot_price))

    return fair_value
