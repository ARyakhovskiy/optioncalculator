import numpy as np
import matplotlib.pyplot as plt
import bs_solver as bss
import bs_utilities as bsu
import bs_visualizer as bsv
from bs_volatility_models import constant_volatility
import datetime as dt
from matplotlib import cm
from implied_volatility import implied_volatility

def dividends(s):
    return 5


if __name__ == "__main__":
    num_asset_steps = 100
    nt = 200
    T = 2 #0.42739726027 # [till December 17] 0.10410958904 # [till August 20]; #0.1808219178 [till September 16];
    r = 0.05
    E = 100.0
    volatility = 0.285
    style = 'US'
    type = 'call'
    minBC = 'Dirichlet'
    maxBC = 'd2Vds2'
    #div_dates =  [dt.datetime(2021, 7, 21, 0, 0, 0, 0),
    #              dt.datetime(2021, 7, 28, 0, 0, 0, 0)]
    dividend_timenodes = [8, 41, 81, 121, 161]
    #dividend_timenodes = []

    asset_prices = bss.asset_prices_uniform(3*E, num_asset_steps)

    volatility_model = constant_volatility

    option_value = bss.rannacher_timestepping(asset_prices, E, volatility, volatility_model, r, T, nt, theta=0.5,
                                              rannacher_steps=200, option_type=type, option_style=style,
                                              dividend_dates=dividend_timenodes, dividend_function=dividends)

    num_sigma_vals = 5

    option_values3d = np.zeros([nt+1, num_asset_steps+1, num_sigma_vals])

    sigmas = np.linspace(0.0, 1.0, num=num_sigma_vals)


    for i in range(num_sigma_vals):
        option_values3d[:, :, i] = bss.rannacher_timestepping(asset_prices, E, sigmas[i], volatility_model, r, T, nt,
                                                              theta=0.5, rannacher_steps=200, option_type=type,
                                                              option_style=style, dividend_dates=dividend_timenodes,
                                                              dividend_function=dividends)

    delta = bsu.delta(asset_prices, option_value)
    gamma = bsu.gamma(asset_prices, option_value)

    print('Delta_Smax = ', delta[-1, -1])
    print('Gamma_Smax = ', gamma[-1, -1])
    print('V_Smax = ', option_value[-1, -1])

    print('Delta_Smin = ', delta[-1, 0])
    print('Gamma_Smin = ', gamma[-1, 0])
    print('V_Smin = ', option_value[-1, 0])

    print('Gamma_max= ', np.amax(gamma[-1, :]))

    exact_solution = np.zeros(option_value.shape)
    for i in range(nt+1):
        exact_solution[i, :] = bsu.black_scholes_formula(asset_prices, i*T/nt,
                                               r, E, volatility, option_type=type)

    tau = np.linspace(0, T, num=nt+1)

    bsv.slider_view(asset_prices, option_value, delta, gamma, tau)

