from django.shortcuts import render
from . black_scholes_solver import black_scholes_formula

def index(request):

    fair_value = 0.0

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
            context = {
                'underlying_price': spot_price,
                'strike_price': strike_price,
            }
            print(black_scholes_formula(spot_price, strike_price, time_till_maturity, volatility, interest_rate,
                                        option_style=option_style, option_type=option_type))
            fair_value = black_scholes_formula(spot_price, strike_price, time_till_maturity, volatility, interest_rate,
                                               option_style=option_style, option_type=option_type)
    return render(request, 'calculator/calculator.html', {'fair_value': fair_value})
# Create your views here.
