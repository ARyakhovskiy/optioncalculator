from django.test import TestCase
import bs_solver as bss
import numpy as np
# Create your tests here.

if __name__ == "__main__":
    num_asset_steps = 100
    nt = 200
    T = 2 #0.42739726027 # [till December 17] 0.10410958904 # [till August 20]; #0.1808219178 [till September 16];
    r = 0.05
    E = 123.0
    volatility = 0.2
    style = 'US'
    type = 'call'
    minBC = 'Dirichlet'
    maxBC = 'd2Vds2'
    #div_dates =  [dt.datetime(2021, 7, 21, 0, 0, 0, 0),
    #              dt.datetime(2021, 7, 28, 0, 0, 0, 0)]
    dividend_timenodes = [8, 41, 81, 121, 161]
    print(bss.run_solver(150, E, T, 0.2, r, option_type='call',
                          option_style='US', dividends=np.array([]), dividend_dates=dividend_timenodes))