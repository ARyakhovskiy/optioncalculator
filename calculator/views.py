from django.shortcuts import render
from . black_scholes_solver import black_scholes_formula
from . bs_solver import run_solver

def index(request):

    fair_value = 0.0
    spot_prices = []
    fair_prices = []
    payoff = []

    context = {
        'fair_value': fair_value,
        'spot_prices': spot_prices,
        'fair_prices': fair_prices,
        'payoff': payoff
    }

    if request.method == 'POST':
        print(request.POST)
        if request.POST.get("calculate"):
            spot_price = float(request.POST['underlying-price'])
            strike_price = float(request.POST['strike-price'])
            volatility = float(request.POST['volatility'])
            time_till_maturity = float(request.POST['time-till-maturity'])
            interest_rate = float(request.POST['interest-rate'])
            option_style = request.POST['option-style']
            option_type = request.POST['option-type']
            method = request.POST['method']

            print(black_scholes_formula(spot_price, strike_price, time_till_maturity, volatility, interest_rate,
                                        option_style=option_style, option_type=option_type))
            if method == "Analytical":
                fair_value = black_scholes_formula(spot_price, strike_price, time_till_maturity, volatility, interest_rate,
                                               option_style=option_style, option_type=option_type)
                context = {
                    'fair_value': fair_value,
                    'spot_prices': spot_prices,
                    'fair_prices': fair_prices,
                    'payoff': payoff
                }
            elif method == "FDM":
                fair_value, rendered_spot_prices_str, rendered_option_prices, rendered_payoff = run_solver(spot_price, strike_price, time_till_maturity, volatility, interest_rate,
                                               option_style=option_style, option_type=option_type)

                print(rendered_spot_prices_str)
                print(rendered_option_prices)
                print(rendered_payoff)

                context = {
                    'fair_value': fair_value,
                    'spot_prices': rendered_spot_prices_str,
                    'fair_prices': rendered_option_prices,
                    'payoff': rendered_payoff
                }
                print(context)

            else:
                raise ValueError("Unknown method")
    return render(request, 'calculator/calculator.html', context)
# Create your views here.
