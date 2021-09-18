from django.shortcuts import render
from . black_scholes_solver import black_scholes_formula
from . bs_solver import run_solver

def index(request):

    fair_value = 0.0
    spot_prices = []
    fair_prices = []
    payoff = []

    greeks = {
        'delta': 0.0,
        'gamma': 0.0,
        'theta': 0.0,
        'vega': 0.0,
        'rho': 0.0
    }

    context = {
        'fair_value': fair_value,
        'spot_prices': spot_prices,
        'fair_prices': fair_prices,
        'payoff': payoff,
        'greeks': greeks,
    }

    if request.method == 'POST':
        print(request.POST)
        if request.POST.get("calculate"):
            spot_price = float(request.POST['underlying-price'])
            strike_price = float(request.POST['strike-price'])
            volatility = float(request.POST['volatility'])
            time_till_maturity = float(request.POST['time-till-maturity'])
            interest_rate = float(request.POST['interest-rate'])
            dividend_yield = float(request.POST['dividend-yield'])
            option_style = request.POST['option-style']
            option_type = request.POST['option-type']
            method = request.POST['method']

            print('Analytical value', black_scholes_formula(spot_price, strike_price, time_till_maturity, volatility, interest_rate,
                                        option_style=option_style, option_type=option_type))
            if method == "Analytical":
                fair_value = black_scholes_formula(spot_price, strike_price, time_till_maturity, volatility, interest_rate,
                                               option_style=option_style, option_type=option_type)
                context = {
                    'fair_value': fair_value,
                    'spot_prices': spot_prices,
                    'fair_prices': fair_prices,
                    'payoff': payoff,
                    'greeks': greeks,
                }
            elif method == "FDM":
                print('Spot price: ', spot_price)
                fair_value, rendered_spot_prices_str, rendered_option_prices, rendered_payoff, greeks = run_solver(spot_price, strike_price, time_till_maturity, volatility, interest_rate,
                                               option_style=option_style, option_type=option_type)

                # print(rendered_spot_prices_str)
                # /print(rendered_option_prices)
                # print(rendered_payoff)

                print('Fair value FDM: ', fair_value)

                print('Context spot_prices: ',rendered_spot_prices_str)

                context = {
                    'fair_value': fair_value,
                    'spot_prices': rendered_spot_prices_str,
                    'fair_prices': rendered_option_prices,
                    'payoff': rendered_payoff,
                    'greeks': greeks
                }
                print(context)

            else:
                raise ValueError("Unknown method")
    return render(request, 'calculator/calculator.html', context)
# Create your views here.
