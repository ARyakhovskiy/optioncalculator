import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import numpy as  np


def plot_v_of_sigma_T_dependency(asset_prices, tau, option_values_3d, sigma):
    fig, ax_t = plt.subplots(1, 1)
    title = 'Option price'
    ax_t.set_title(title)
    legend_list = []

    ax_t.plot(sigma, option_values_3d[0, 0, :])
    ax_t.set_ylabel('V')
    ax_t.set_xlabel('sigma')
    axcolor = 'lightgoldenrodyellow'
    ax_t.margins(x=0)
    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)
    # Make a horizontal slider to control the frequency.
    ax_time = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    ax_price = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)

    time_slider = Slider(
        ax=ax_time,
        label='T',
        valmin=0.0,
        valmax=max(tau),
        valinit=0.0,
    )

    price_slider = Slider(
        ax=ax_price,
        label='S',
        valmin=0.0,
        valmax=max(asset_prices),
        valinit=0.0,
    )

    def update_value(val):
        s = price_slider.val
        t = time_slider.val
        s_index = int(s/max(asset_prices)*(len(asset_prices)-1))
        t_index = int(t/max(tau)*(len(tau)-1))
        ax_t.cla()
        ax_t.plot(sigma, option_values_3d[t_index, s_index, :])
        ax_t.set_ylabel('V')
        ax_t.set_xlabel('sigma')
        ax_t.margins(x=0)
        fig.canvas.draw_idle()

    # register the update function with each slider
    time_slider.on_changed(update_value)
    price_slider.on_changed(update_value)

    plt.show()
    return


def plot_v_of_t_sigma_dependency(asset_prices, tau, option_values_3d, sigma):

    fig, ax_t = plt.subplots(1, 1)
    title = 'Option price [US call; div_value = 10; X = 100]'
    ax_t.set_title(title)
    legend_list = []

    for i in range(len(sigma)):
        ax_t.plot(tau, option_values_3d[:, 0, i])
        legend_list.append('volatility='+"{:.2f}".format(sigma[i]))
    ax_t.legend(legend_list)

    ax_t.set_xlabel('T')
    axcolor = 'lightgoldenrodyellow'
    ax_t.margins(x=0)
    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)
    # Make a horizontal slider to control the frequency.
    ax_time = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    price_slider_value = Slider(
        ax=ax_time,
        label='S',
        valmin=0.0,
        valmax=max(asset_prices),
        valinit=0.0,
    )

    def update_value(val):
        s_index = int(val/max(asset_prices)*(len(asset_prices)-1))
        #value_graph.set_ydata(option_value[timestep, :])
        ax_t.cla()
        ax_t.set_title(title)
        for i in range(len(sigma)):
            ax_t.plot(tau, option_values_3d[:, s_index, i])
            legend_list.append('volatility=' + "{:.2f}".format(sigma[i]))
        ax_t.legend(legend_list)

        ax_t.set_xlabel('T')
        axcolor = 'lightgoldenrodyellow'
        ax_t.margins(x=0)
        fig.canvas.draw_idle()

    # register the update function with each slider
    price_slider_value.on_changed(update_value)
    plt.show()
    return



def animation_view(asset_prices, option_value, delta, gamma, tau):

    fig, ax = plt.subplots(2, 2)
    for i in range(len(option_value[:, 0])):
        ax[0, 0].cla()
        ax[0, 0].set_ylim(-10, np.amax(option_value))
        ax[0, 0].plot(asset_prices, option_value[0, :])
        ax[0, 0].plot(asset_prices, option_value[i, :])
        ax[0, 0].legend(['Payoff', 'Time to maturity:'+"{:.2f}".format(tau[i])])
        ax[0, 0].set_title('Option Value')


        ax[0, 1].cla()
        ax[0, 1].plot(asset_prices, delta[i, :])
        ax[0, 1].set_ylim(np.amin(delta), np.amax(delta))
        ax[0, 1].set_title('Delta')

        ax[1, 0].cla()
        ax[1, 0].set_ylim(np.amin(gamma), np.amax(gamma)*1)
        ax[1, 0].plot(asset_prices, gamma[i, :])
        ax[1, 0].set_title('Gamma')


        ''''
        ax[1, 1].cla()
        if calc_iv == True:
        ax[1, 1].set_ylim(-0.1, np.amax(iv_array)*1.1)
        for j in range(len(iv_array[i, :])):
           if iv_converged[i, j]:
               ax[1, 1].scatter(itm_prices[j], iv_array[i, j])
        ax[1, 1].plot(itm_prices, iv_array[i, :])
        ax[1, 1].set_title('IV')
        '''
        plt.pause(0.05)
    plt.show()
    return


def slider_view(asset_prices, option_value, delta, gamma, tau):

    fig_value, ax_value = plt.subplots()
    ax_value.set_xlabel('S')
    axcolor = 'lightgoldenrodyellow'
    ax_value.margins(x=0)
    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)
    value_graph, = plt.plot(asset_prices, option_value[0, :], lw=2)
    payoff_line = plt.plot(asset_prices, option_value[0, :], lw=2)
    # Make a horizontal slider to control the frequency.
    ax_time = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    time_slider_value = Slider(
        ax=ax_time,
        label='Time',
        valmin=0.0,
        valmax=max(tau),
        valinit=0.0,
    )

    # The function to be called anytime a slider's value changes
    def update_value(val):
        timestep = int(val/max(tau)*(len(tau)-1))
        value_graph.set_ydata(option_value[timestep, :])
        fig_value.canvas.draw_idle()

    # register the update function with each slider
    time_slider_value.on_changed(update_value)

    fig_gamma, ax_gamma = plt.subplots()
    ax_gamma.set_xlabel('S')
    axcolor = 'blue'
    ax_gamma.margins(x=0)
    # adjust the main plot to make room for the sliders
    plt.subplots_adjust(left=0.25, bottom=0.25)
    gamma_graph, = plt.plot(asset_prices, gamma[0, :], lw=2)
    # Make a horizontal slider to control the frequency.
    ax_time = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)
    time_slider_gamma = Slider(
        ax=ax_time,
        label='Time',
        valmin=0.0,
        valmax=max(tau),
        valinit=0.0,
    )

    # The function to be called anytime a slider's value changes
    def update_gamma(val):
        timestep = int(val/max(tau)*(len(tau)-1))
        gamma_graph.set_ydata(gamma[timestep, :])
        fig_gamma.canvas.draw_idle()

    # register the update function with each slider
    time_slider_gamma.on_changed(update_gamma)

    plt.show()

    return
