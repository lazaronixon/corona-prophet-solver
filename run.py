import os
import numpy  as np
import pandas as pd
import scipy.optimize as optim

from flask     import Flask, Response, request
from fbprophet import Prophet

app = Flask(__name__)

##########################LOGISTIC FORECAST FOR CONFIRMED#####################################
@app.route('/confirmed', methods=['POST'])
def confirmed():
    # Define funcion with the coefficients to estimate
    def func_logistic(t, a, b, c):
        return c / (1 + a * np.exp(-b*t))

    def set_capacity():
        # Randomly initialize the coefficients
        p0 = np.random.exponential(size=3)

        # Set min bound 0 on all coefficients, and set different max bounds for each coefficient
        bounds = (0, [100000., 1000., 1000000000.])

        # Convert pd.Series to np.Array and use Scipy's curve fit to find the best Nonlinear Least Squares coefficients
        x = np.array(dataset['ts']) + 1
        y = np.array(dataset['y'])

        (a,b,c),cov = optim.curve_fit(func_logistic, x, y, bounds=bounds, p0=p0, maxfev=1000000)

        # The time step at which the growth is fastest
        t_fastest = np.log(a) / b
        i_fastest = func_logistic(t_fastest, a, b, c)

        if t_fastest <= x[-1]:
          dataset['cap'] = func_logistic(x[-1] + 10, a, b, c)
        else:
          dataset['cap'] = func_logistic(i_fastest + 10, a, b, c)

    periods = int(request.headers['periods'])
    body    = request.get_json()

    dataset = pd.DataFrame(body)
    dataset = dataset.reset_index(drop=False)
    dataset.columns = ['ts', 'ds', 'y']

    set_capacity()

    m = Prophet(growth='logistic')
    m.fit(dataset)

    future = m.make_future_dataframe(periods=periods)
    future['cap'] = dataset['cap'][0]

    forecast = m.predict(future)

    response = forecast[['ds', 'yhat', 'cap']].tail(periods)

    return Response(response.to_json(orient='records', date_format='iso'), mimetype='application/json')

##########################LOGISTIC FORECAST FOR CONFIRMED#####################################
@app.route('/deaths', methods=['POST'])
def deaths():
    periods = int(request.headers['periods'])
    body    = request.get_json()

    dataset = pd.DataFrame(body)

    m = Prophet(changepoint_prior_scale=0.1, changepoint_range=1, seasonality_mode='multiplicative', yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    m.fit(dataset)

    future = m.make_future_dataframe(periods=periods)

    forecast = m.predict(future)

    response = forecast[['ds', 'yhat']].tail(periods)

    return Response(response.to_json(orient='records', date_format='iso'), mimetype='application/json')

########################### WEB SERVICE RUN ###############################################
if __name__ == "__main__":
    app.run()
