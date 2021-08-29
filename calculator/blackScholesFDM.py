import numpy as np
import warnings


def payoff(S, K, option_type='call'):
    S = np.array(S)
    if option_type == 'call':
        return np.max(0, S-K)
    elif option_type == 'put':
        return np.max(0, K-S)
    else:
        raise ValueError


def thomas_method(a, b, c, d, p):
    nf = len(d) #number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    for it in range(1, nf):
        mc = ac[it-1] / bc[it-1]
        bc[it] = bc[it] - mc * cc[it-1]
        dc[it] = dc[it] - mc * dc[it-1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il+1]) / bc[il]

    return xc


def thomas_method_american(a, b, c, d, p):
    nf = len(d) #number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))
    for it in range(1, nf):
        mc = ac[it-1] / bc[it-1]
        bc[it] = bc[it] - mc * cc[it-1]
        dc[it] = dc[it] - mc * dc[it-1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf-2, -1, -1):
        xc[il] = max((dc[il] - cc[il] * xc[il+1]) / bc[il], payoff[il])
    return xc


def create_tridiagonal_system():
    a = np.zeros()
    b = np.zeros()
    c = np.zeros()

    return a, b, c


def create_asset_mesh(strike_price, steps_per_strike, max_asset_price_multiplier):
    mesh = np.linspace(0, strike_price*max_asset_price_multiplier, steps_per_strike*max_asset_price_multiplier+1)
    return mesh

def create_time_mesh(num_timesteps, time_till_maturity):
    return np.linspace(0, time_till_maturity, num_timesteps+1)

def solve(spot_price, strike_price, time_till_maturity, volatility, interest_rate, option_type='call',
                          option_style='EU', dividends=np.array([]), dividend_dates=np.array([])):
    num_timesteps = 100
    steps_per_strike = 60
    max_asset_price_multiplier = 3

    asset_mesh = create_asset_mesh(strike_price, steps_per_strike, max_asset_price_multiplier)
    time_mesh = create_time_mesh(num_timesteps, time_till_maturity)
    num_asset_nodes = steps_per_strike*max_asset_price_multiplier+1
    num_time_nodes = num_timesteps+1

    option_values = np.zeros(num_time_nodes, num_asset_nodes)

    option_values[0, :] = payoff(asset_mesh, strike_price, option_type)

    return 0.0