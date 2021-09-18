import numpy as np


def get_delta(asset_mesh, option_values, spot_price):
    if spot_price < 0 or spot_price > asset_mesh[-1]:
        raise ValueError('Spot price is outside of solution range')
    idx = (np.abs(asset_mesh-spot_price)).argmin()

    if spot_price < asset_mesh[idx]:
        ds = asset_mesh[idx]-asset_mesh[idx-1]
        weight_left = (spot_price-asset_mesh[idx-1])/ds
        weight_right = 1-weight_left
        delta_left = 0.5*(option_values[idx] - option_values[idx-2])/ds
        delta_right = 0.5*(option_values[idx+1] - option_values[idx-1])/ds
        return weight_left*delta_right+weight_right*delta_left
    elif spot_price > asset_mesh[idx]:
        ds = asset_mesh[idx+1] - asset_mesh[idx]
        weight_left = (spot_price-asset_mesh[idx])/ds
        weight_right = 1-weight_left
        delta_left = 0.5*(option_values[idx+1] - option_values[idx-1])/ds
        delta_right = 0.5*(option_values[idx+2] - option_values[idx])/ds
        return weight_left*delta_right+weight_right*delta_left
    else:
        ds = asset_mesh[idx]-asset_mesh[idx-1]
        return 0.5*(option_values[idx+1]-option_values[idx-1])/ds


def get_gamma(asset_mesh, option_values, spot_price):
    if spot_price < 0 or spot_price > asset_mesh[-1]:
        raise ValueError('Spot price is outside of solution range')
    idx = (np.abs(asset_mesh-spot_price)).argmin()

    if spot_price < asset_mesh[idx]:
        ds = asset_mesh[idx]-asset_mesh[idx-1]
        weight_left = (spot_price-asset_mesh[idx-1])/ds
        weight_right = 1-weight_left
        gamma_left = (option_values[idx] - 2 * option_values[idx-1] + option_values[idx-2])/ds/ds
        gamma_right = (option_values[idx+1] - 2 * option_values[idx] + option_values[idx-1])/ds/ds
        return weight_left*gamma_right+weight_right*gamma_left
    elif spot_price > asset_mesh[idx]:
        ds = asset_mesh[idx+1] - asset_mesh[idx]
        weight_left = (spot_price-asset_mesh[idx])/ds
        weight_right = 1-weight_left
        gamma_left = (option_values[idx+1] - 2 * option_values[idx] + option_values[idx-1])/ds/ds
        gamma_right = (option_values[idx+2] - 2 * option_values[idx+1] + option_values[idx])/ds/ds
        return weight_left*gamma_right+weight_right*gamma_left
    else:
        ds = asset_mesh[idx]-asset_mesh[idx-1]
        return (option_values[idx+1] - 2 * option_values[idx] + option_values[idx-1])/ds/ds


def get_theta(asset_mesh, dt, option_values, spot_price):
    if spot_price < 0 or spot_price > asset_mesh[-1]:
        raise ValueError('Spot price is outside of solution range')
    idx = (np.abs(asset_mesh-spot_price)).argmin()

    if spot_price < asset_mesh[idx]:
        ds = asset_mesh[idx]-asset_mesh[idx-1]
        weight_left = (spot_price-asset_mesh[idx-1])/ds
        weight_right = 1-weight_left
        theta_left = (option_values[-1, idx-1] - option_values[-2, idx-1])/dt
        theta_right = (option_values[-1, idx] - option_values[-2, idx])/dt
        return weight_left*theta_right+weight_right*theta_left
    elif spot_price > asset_mesh[idx]:
        ds = asset_mesh[idx+1] - asset_mesh[idx]
        weight_left = (spot_price-asset_mesh[idx])/ds
        weight_right = 1-weight_left
        theta_left = (option_values[-1, idx] - option_values[-2, idx])/dt
        theta_right = (option_values[-1, idx+1] - option_values[-2, idx+1])/dt
        return weight_left*theta_right+weight_right*theta_left
    else:
        return (option_values[-1, idx]-option_values[-2, idx])/dt


def get_vega(asset_mesh, option_values, option_values_sigma_shift, sigma_shift, spot_price):
    if spot_price < 0 or spot_price > asset_mesh[-1]:
        raise ValueError('Spot price is outside of solution range')
    idx = (np.abs(asset_mesh-spot_price)).argmin()

    if spot_price < asset_mesh[idx]:
        ds = asset_mesh[idx]-asset_mesh[idx-1]
        weight_left = (spot_price-asset_mesh[idx-1])/ds
        weight_right = 1-weight_left
        vega_left = (option_values_sigma_shift[idx-1] - option_values[idx-1])/sigma_shift
        vega_right = (option_values_sigma_shift[idx] - option_values[idx])/sigma_shift
        return weight_left*vega_right+weight_right*vega_left
    elif spot_price > asset_mesh[idx]:
        ds = asset_mesh[idx+1] - asset_mesh[idx]
        weight_left = (spot_price-asset_mesh[idx])/ds
        weight_right = 1-weight_left
        vega_left = (option_values_sigma_shift[idx] - option_values[idx])/sigma_shift
        vega_right = (option_values_sigma_shift[idx+1] - option_values[idx+1])/sigma_shift
        return weight_left*vega_right+weight_right*vega_left
    else:
        return (option_values_sigma_shift[idx] - option_values[idx])/sigma_shift

def get_rho(asset_mesh, option_values, option_values_r_shift, r_shift, spot_price):
    if spot_price < 0 or spot_price > asset_mesh[-1]:
        raise ValueError('Spot price is outside of solution range')
    idx = (np.abs(asset_mesh-spot_price)).argmin()

    if spot_price < asset_mesh[idx]:
        ds = asset_mesh[idx]-asset_mesh[idx-1]
        weight_left = (spot_price-asset_mesh[idx-1])/ds
        weight_right = 1-weight_left
        rho_left = (option_values_r_shift[idx-1] - option_values[idx-1])/r_shift
        rho_right = (option_values_r_shift[idx] - option_values[idx])/r_shift
        return weight_left*rho_right+weight_right*rho_left
    elif spot_price > asset_mesh[idx]:
        ds = asset_mesh[idx+1] - asset_mesh[idx]
        weight_left = (spot_price-asset_mesh[idx])/ds
        weight_right = 1-weight_left
        rho_left = (option_values_r_shift[idx] - option_values[idx])/r_shift
        rho_right = (option_values_r_shift[idx+1] - option_values[idx+1])/r_shift
        return weight_left*rho_right+weight_right*rho_left
    else:
        return (option_values_r_shift[idx] - option_values[idx])/r_shift