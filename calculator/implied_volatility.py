import numpy as np
import scipy.optimize as sc
from scipy.misc import derivative
from . bs_volatility_models import constant_volatility
from . bs_solver import asset_prices_uniform, rannacher_timestepping

def make_v_of_sigma_function(method, S, T, r, K, option_type='call', option_style='EU', div_values=[], div_dates=[]):
    def v_of_sigma(sigma):
        asset_prices = asset_prices_uniform(2*K, 100)
        nt = 100
        option_values = rannacher_timestepping(asset_prices, K, sigma, constant_volatility,
                                                   r, T, nt, theta=0.5, rannacher_steps=100,
                                                  option_type=option_type, option_style=option_style,
                                                  dividend_dates=div_dates, dividend_function=div_values)
        #s_index = np.where(asset_prices == S)
        s_index = np.abs(asset_prices - S).argmin()
        v = option_values[-1, s_index]
        print(v)
        return v
    return v_of_sigma


def solve_implied_volatility(market_price, asset_price, T, r, strike_price, sigma0, tolerance,
                       option_type='call', option_style='EU',
                       div_values=[], div_dates=[]):
    solution = rannacher_timestepping

    def target_func(sigma):
        v_of_sigma = make_v_of_sigma_function(solution, asset_price, T, r,
                                              strike_price, option_type, option_style, div_values, div_dates)
        return v_of_sigma(sigma) - market_price

    def vega(sigma):
        return derivative(target_func, sigma, dx=1e-6)

    #if vega(v0) != 0:
    #iv = sc.newton(target_func, sigma0, fprime=vega, disp=False,  tol=tolerance, maxiter=50)
    #else:
    iv = sc.newton(target_func, sigma0, disp=False,  tol=tolerance, maxiter=50)
    return [iv, r]
