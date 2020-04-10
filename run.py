import os
import numpy  as np
import pandas as pd
import scipy.optimize as optim

from flask     import Flask, Response, request
from fbprophet import Prophet

app = Flask(__name__)

##########################LOGISTIC FORECAST FOR CONFIRMED#####################################
@app.route('/prophet', methods=['POST'])
def prophet():
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
