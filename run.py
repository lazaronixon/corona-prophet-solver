import os
import numpy  as np
import pandas as pd
import scipy.optimize as optim

from flask     import Flask, Response, request
from fbprophet import Prophet

app = Flask(__name__)

##########################FORECAST FOR DEATHS#####################################
@app.route('/prophetize', methods=['POST'])
def prophetize():
    periods = int(request.headers['periods'])
    body    = request.get_json()

    dataset = pd.DataFrame(body)
    dataset = dataset.query('y > 0')

    m = Prophet()
    m.fit(dataset)

    future = m.make_future_dataframe(periods=periods)

    forecast = m.predict(future)

    response = forecast[['ds', 'yhat']].tail(periods)

    return Response(response.to_json(orient='records', date_format='iso'), mimetype='application/json')

########################### WEB SERVICE RUN ###############################################
if __name__ == "__main__":
    app.run()
