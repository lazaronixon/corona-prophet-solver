import os
import pandas as pd

from flask     import Flask, Response, request
from fbprophet import Prophet

app = Flask(__name__)

@app.route('/prophet', methods=['POST'])
def prophet():
    periods = int(request.headers['periods'])
    body    = request.get_json()

    dataset = pd.DataFrame(body)

    dataset.loc[1:, 'y'] = dataset['y'].diff()[1:]

    m = Prophet(yearly_seasonality=False, weekly_seasonality=False, daily_seasonality=False)
    m.fit(dataset)

    future = m.make_future_dataframe(periods=periods)

    forecast = m.predict(future)
    forecast['yhat'] = forecast['yhat'].cumsum()

    response = forecast[['ds', 'yhat']].tail(periods)

    return Response(response.to_json(orient='records', date_format='iso'), mimetype='application/json')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
