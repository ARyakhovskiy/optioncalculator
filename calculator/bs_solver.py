import numpy as np
import warnings
from . bs_volatility_models import constant_volatility
from .greeks import get_delta, get_gamma, get_theta, get_vega, get_rho


def intrinsic_value(asset_prices, strike_price, option_type="call"):
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


def get_fair_value(asset_mesh, option_values, spot_price):
    if spot_price < 0 or spot_price > asset_mesh[-1]:
        raise ValueError('Spot price is outside of solution range')
    idx = (np.abs(asset_mesh-spot_price)).argmin()

    if spot_price < asset_mesh[idx]:
        ds = asset_mesh[idx]-asset_mesh[idx-1]
        weight_left = (spot_price-asset_mesh[idx-1])/ds
        weight_right = 1-weight_left
        return weight_left*option_values[idx]+weight_right*option_values[idx-1]
    elif spot_price > asset_mesh[idx]:
        ds = asset_mesh[idx+1] - asset_mesh[idx]
        weight_left = (spot_price-asset_mesh[idx])/ds
        weight_right = 1-weight_left
        return weight_left*option_values[idx+1]+weight_right*option_values[idx]
    else:
        return option_values[idx]


def run_solver(spot_price, strike_price, time_till_maturity, volatility, interest_rate, option_type='call',
                          option_style='EU', dividends=np.array([]), dividend_dates=np.array([])):
    num_asset_steps = 100
    nt = 200
    T = time_till_maturity  # 0.42739726027 # [till December 17] 0.10410958904 # [till August 20]; #0.1808219178 [till September 16];
    r = interest_rate
    E = strike_price
    volatility = volatility
    style = option_style
    type = option_type
    minBC = 'Dirichlet'
    maxBC = 'd2Vds2'
    # div_dates =  [dt.datetime(2021, 7, 21, 0, 0, 0, 0),
    #              dt.datetime(2021, 7, 28, 0, 0, 0, 0)]
    #dividend_timenodes = [8, 41, 81, 121, 161]
    dividend_timenodes = []

    asset_prices = asset_prices_uniform(3 * E, num_asset_steps)

    volatility_model = constant_volatility

    def div_func(s):
        return 5

    option_value = rannacher_timestepping(asset_prices, E, volatility, volatility_model, r, T, nt, theta=0.5,
                                              rannacher_steps=200, option_type=type, option_style=style,
                                              dividend_dates=dividend_timenodes, dividend_function=div_func)

    payoff = intrinsic_value(asset_prices, strike_price, option_type)
    fair_value = get_fair_value(asset_prices, option_value[-1, :], spot_price)

    rendered_spot_prices = np.arange(spot_price-3, spot_price+4)
    print(rendered_spot_prices)
    rendered_spot_prices_str = [str(price) for price in rendered_spot_prices]
    print(rendered_spot_prices_str)
    rendered_option_prices = [get_fair_value(asset_prices, option_value[-1, :], price) for price in rendered_spot_prices]
    print(rendered_option_prices)
    rendered_payoff = intrinsic_value(rendered_spot_prices, strike_price, option_type).tolist()
    print(rendered_payoff)

    sigma_shift = max(1e-04, 0.01*volatility)
    r_shift = max(1e-04, 0.01*r)

    option_value_sigma_shift = rannacher_timestepping(asset_prices, E, volatility+sigma_shift, volatility_model,
                                                r, T, nt, theta=0.5,
                                                rannacher_steps=200, option_type=type, option_style=style,
                                                dividend_dates=dividend_timenodes, dividend_function=div_func)

    option_value_r_shift = rannacher_timestepping(asset_prices, E, volatility, volatility_model,
                                                r+r_shift, T, nt, theta=0.5,
                                                rannacher_steps=200, option_type=type, option_style=style,
                                                dividend_dates=dividend_timenodes, dividend_function=div_func)

    dt = T/nt

    greeks = {
        'delta': get_delta(asset_prices, option_value[-1, :], spot_price),
        'gamma': get_gamma(asset_prices, option_value[-1, :], spot_price),
        'vega': get_vega(asset_prices, option_value[-1, :], option_value_sigma_shift[-1, :], sigma_shift, spot_price),
        'theta': get_theta(asset_prices, dt, option_value, spot_price),
        'rho': get_rho(asset_prices, option_value[-1, :], option_value_r_shift[-1, :], r_shift, spot_price),
    }
    return fair_value, rendered_spot_prices_str, rendered_option_prices, rendered_payoff, greeks

def thomas_method(a, b, c, d, p):
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    if bc[-1] < 1e-06:
        warnings.warn("Warning: Matrix coefficient is too small")
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


def thomas_method_american_option(a, b, c, d, payoff):
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = max(dc[-1] / bc[-1], payoff[-1])

    for il in range(nf - 2, -1, -1):
        xc[il] = max((dc[il] - cc[il] * xc[il + 1]) / bc[il], payoff[il])

    return xc

def thomas_method_american_option_put(a, b, c, d, payoff):
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in reversed(range(1, nf)):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = max(dc[-1] / bc[-1], payoff[-1])

    for il in reversed(range(nf - 2, -1, -1)):
        xc[il] = max((dc[il] - cc[il] * xc[il + 1]) / bc[il], payoff[il])

def tridiagonal_system_crank_nicholson(asset_prices, dt, volatility, interest_rate, theta=0.5):
    num_asset_nodes = len(asset_prices)
    ns = num_asset_nodes - 1
    dS = asset_prices[1]-asset_prices[0]
    sigma = volatility
    r = interest_rate

    a = np.zeros(num_asset_nodes)
    b = np.zeros(num_asset_nodes)
    c = np.zeros(num_asset_nodes)
    for i in range(1, ns):
        a[i] = -0.5*theta*(sigma * sigma*i*i - r * i)*dt
        b[i] = 1 + theta*(sigma*sigma*i*i+r)*dt
        c[i] = -0.5*theta*(sigma * sigma*i*i + r * i)*dt

    return a, b, c


def rhs_crank_nicholson(option_values, ns, dt, sigma, r, theta=0.5):
    rhs = np.zeros(ns)
    for i in range(ns-1):
        a = (1-theta) * 0.5 * (sigma * sigma*i*i - r * i)*dt
        b = 1-(1-theta)*(sigma*sigma*i*i+r)*dt
        c = (1-theta) * 0.5 *(sigma * sigma*i*i + r * i)*dt
        rhs[i] = a*option_values[i-1]+b*option_values[i]+c*option_values[i+1]

    return rhs


def rhs_implicit_euler(options_values):
    return options_values


def tridiagonal_system_implicit_euler(asset_prices, dt, volatility, interest_rate):
    num_asset_nodes = len(asset_prices)
    ns = num_asset_nodes - 1
    dS = asset_prices[1]-asset_prices[0]
    sigma = volatility
    r = interest_rate

    a = np.zeros(num_asset_nodes)
    b = np.zeros(num_asset_nodes)
    c = np.zeros(num_asset_nodes)
    for i in range(1, ns):
        a[i] = -0.5*(sigma * sigma*i*i - r * i)*dt
        b[i] = 1 + (sigma*sigma*i*i+r)*dt
        c[i] = -0.5*(sigma * sigma*i*i + r * i)*dt

    return a, b, c


def apply_dirichlet_boundaries_implicit_euler(a, b, c, rhs, v, asset_prices,
                                              time_to_expiry, strike_price, interest_rate, dividends_paid_to_date,
                                              option_type='call', option_style='EU'):

    if option_type == 'call':
        value_at_smin = 0.0
        eu_call_payoff_at_smax = asset_prices[-1] - (dividends_paid_to_date + \
                                strike_price) * np.exp(-interest_rate * time_to_expiry)
        #eu_call_payoff_at_smax = asset_prices[-1] - \
        #                         strike_price * np.exp(-interest_rate * time_to_expiry)
        if option_style == 'EU':
            value_at_smax = eu_call_payoff_at_smax
        elif option_style == 'US':
            value_at_smax = max(asset_prices[-1] - strike_price, eu_call_payoff_at_smax)

    elif option_type == 'put':
        value_at_smax = 0.0
        eu_put_payoff_at_smin = strike_price*np.exp(-interest_rate*time_to_expiry)
        if option_style == 'EU':
            value_at_smin = eu_put_payoff_at_smin
        elif option_style == 'US':
            value_at_smin = max(strike_price, eu_put_payoff_at_smin)

    v_bc = np.copy(v)
    v_bc[0] = value_at_smin
    v_bc[-1] = value_at_smax
    rhs_bc = np.copy(rhs[1:-1])
    rhs_bc[0] -= value_at_smin*a[1]
    rhs_bc[-1] -= value_at_smax*c[-2]
    a_bc = np.copy(a[2:-1])
    b_bc = np.copy(b[1:-1])
    c_bc = np.copy(c[1:-2])

    return a_bc, b_bc, c_bc, rhs_bc, v_bc


def apply_d2vds2_boundaries_implicit_euler(a, b, c, rhs, v, asset_prices,
                                              time_to_expiry, strike_price, interest_rate, dividends_paid_to_date,
                                              option_type='call', option_style='EU'):

    if option_type == 'call':
        value_at_smin = 0.0
        eu_call_payoff_at_smax = asset_prices[-1] - (dividends_paid_to_date + \
                                strike_price) * np.exp(-interest_rate * time_to_expiry)
        #eu_call_payoff_at_smax = asset_prices[-1] - \
        #                         strike_price * np.exp(-interest_rate * time_to_expiry)
        if option_style == 'EU':
            value_at_smax = eu_call_payoff_at_smax
        elif option_style == 'US':
            value_at_smax = max(asset_prices[-1] - strike_price, eu_call_payoff_at_smax)

    elif option_type == 'put':
        value_at_smax = 0.0
        eu_put_payoff_at_smin = strike_price*np.exp(-interest_rate*time_to_expiry)
        if option_style == 'EU':
            value_at_smin = eu_put_payoff_at_smin
        elif option_style == 'US':
            value_at_smin = max(strike_price, eu_put_payoff_at_smin)

    v_bc = np.copy(v)
    v_bc[0] = value_at_smin
    rhs_bc = np.copy(rhs[1:-1])
    rhs_bc[0] -= value_at_smin*a[1]


    a_bc = np.copy(a[2:-1])
    b_bc = np.copy(b[1:-1])
    b_bc[-1] += 2*c[-2]
    a_bc[-1] -= c[-2]

    c_bc = np.copy(c[1:-2])

    return a_bc, b_bc, c_bc, rhs_bc, v_bc


def asset_prices_uniform(max_asset_price, num_asset_steps):
    return np.linspace(0, max_asset_price, num=num_asset_steps+1)


def apply_dividends(asset_prices, values_old, div_func, strike_price, option_type='call', option_style='EU'):
    ns = len(asset_prices)
    ds = asset_prices[1]-asset_prices[0]
    values_new = np.zeros(ns)
    for i in range(ns):
        inew = int((asset_prices[i] - div_func(asset_prices[i]))/ds)
        temp = (asset_prices[i] - div_func(asset_prices[i]) - inew*ds)/ds
        if inew >= ns-1:
            print('a')
            #n_over = inew-ns+1
            #dv = values_old[-1] - values_old[-2]
            values_new[i] = values_old[-1] #+ n_over*dv
        elif inew < 0:
            #n_under = -inew
            #dv = values_old[1] - values_old[0]
            if option_type == 'call':
                values_new[i] == 0.0
            elif option_type == 'put':
                values_new[i] = values_old[0] #+ n_under*dv
        else:
            values_new[i] = (1-temp)*values_old[inew] + temp*values_old[inew+1]

    if option_style == 'US':
        values_new = np.maximum(values_new, intrinsic_value(asset_prices, strike_price, option_type))
    return values_new


def zero_dividend(S):
    return 0.0


def rannacher_timestepping(asset_prices, strike_price, volatility, volatility_model, interest_rate, T,
                            nt, theta=0.5, rannacher_steps=0, option_type='call', option_style='EU', dividend_dates=[],
                            dividend_function=zero_dividend):

    num_asset_nodes = len(asset_prices)
    num_time_nodes = nt + 1
    dt = T / nt
    dt_rannacher = 0.5*dt
    max_asset_price = asset_prices[-1]
    option_value = np.zeros([num_time_nodes, num_asset_nodes])
    payoff = intrinsic_value(asset_prices, strike_price, option_type=option_type)
    option_value[0, :] = np.copy(payoff)

    if option_style == 'EU':
        method = thomas_method
    elif option_style == 'US':
        method = thomas_method_american_option
    else:
        raise ValueError('Unknown option style')

    dividend_timenodes = dividend_dates
    divs_to_date = 0.0

    def sigma_model(T):
        return volatility_model(volatility, strike_price, T)


    for i in range(1, rannacher_steps+1):
        rhs = np.copy(option_value[i-1, :])
        current_time = i*dt
        sigma = sigma_model(current_time)
        a, b, c = tridiagonal_system_implicit_euler(asset_prices, dt_rannacher, sigma, interest_rate)
        a_bc, b_bc, c_bc, rhs_bc, option_value[i, :] = \
            apply_d2vds2_boundaries_implicit_euler(a, b, c, rhs, option_value[i, :], asset_prices,
                                                     i*dt_rannacher, strike_price, interest_rate, divs_to_date,
                                                      option_type, option_style)

        # apply_dirichlet_boundaries_implicit_euler(a, b, c, rhs, option_value[i, :], asset_prices,
        #                                          i*dt, strike_price, interest_rate, divs_to_date,
        #                                          option_type, option_style)
        option_value_intermediate = np.copy(rhs)
        option_value_intermediate[1:-1] = method(a_bc, b_bc, c_bc, rhs_bc, payoff[1:-1])
        option_value_intermediate[-1] = 2*option_value_intermediate[-2] - option_value_intermediate[-3]

        rhs = np.copy(option_value_intermediate)
        a, b, c = tridiagonal_system_implicit_euler(asset_prices, dt_rannacher, sigma, interest_rate)
        a_bc, b_bc, c_bc, rhs_bc, option_value[i, :] = \
            apply_d2vds2_boundaries_implicit_euler(a, b, c, rhs, option_value_intermediate, asset_prices,
                                                     i*dt_rannacher, strike_price, interest_rate, divs_to_date,
                                                      option_type, option_style)

        # apply_dirichlet_boundaries_implicit_euler(a, b, c, rhs, option_value[i, :], asset_prices,
        #                                          i*dt, strike_price, interest_rate, divs_to_date,
        #                                          option_type, option_style)
        option_value[i, 1:-1] = method(a_bc, b_bc, c_bc, rhs_bc, payoff[1:-1])
        option_value[i, -1] = 2*option_value[i, -2] - option_value[i, -3]
        if i in dividend_timenodes:
            option_value[i, :] = apply_dividends(asset_prices, option_value[i, :], dividend_function,
                                                 strike_price, option_type, option_style)
            divs_to_date += dividend_function(asset_prices[-1])

    for i in range(rannacher_steps, nt + 1):
        current_time = i*dt
        sigma = sigma_model(current_time)
        rhs = np.copy(option_value[i - 1, :])
        a, b, c = tridiagonal_system_crank_nicholson(asset_prices, dt, sigma, interest_rate, theta)
        rhs = rhs_crank_nicholson(np.copy(option_value[i-1, :]),
                                  len(asset_prices), dt, sigma, interest_rate, theta)
        a_bc, b_bc, c_bc, rhs_bc, option_value[i, :] = \
            apply_d2vds2_boundaries_implicit_euler(a, b, c, rhs, option_value[i, :], asset_prices,
                                                   i * dt, strike_price, interest_rate, divs_to_date,
                                                   option_type, option_style)

        # apply_dirichlet_boundaries_implicit_euler(a, b, c, rhs, option_value[i, :], asset_prices,
        #                                          i*dt, strike_price, interest_rate, divs_to_date,
        #                                          option_type, option_style)
        option_value[i, 1:-1] = method(a_bc, b_bc, c_bc, rhs_bc, payoff[1:-1])
        option_value[i, -1] = 2 * option_value[i, -2] - option_value[i, -3]
        if i in dividend_timenodes:
            option_value[i, :] = apply_dividends(asset_prices, option_value[i, :], dividend_function,
                                                 strike_price, option_type, option_style)
            divs_to_date += dividend_function(asset_prices[-1])
    return option_value
