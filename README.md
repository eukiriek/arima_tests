
import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA

# Пример данных
df = pd.read_excel('turnover_drp.xlsx')
df['date'] = pd.to_datetime(df['date'])
df = df.set_index('date')

# Делим ряд
train = df['turnover'][:-12]
test = df['turnover'][-12:]

# 1. Экспоненциальное сглаживание
es_model = ExponentialSmoothing(train, trend='add', seasonal='add', seasonal_periods=12)
es_fit = es_model.fit()
es_forecast = es_fit.forecast(len(test))

# 2. Остатки
residuals = train - es_fit.fittedvalues

# 3. ARIMA на остатках
arima_model = ARIMA(residuals, order=(1,0,0))  # можно подобрать p,d,q
arima_fit = arima_model.fit()

# Прогноз ошибок
arima_forecast = arima_fit.forecast(len(test))

# 4. Комбинированный прогноз
hybrid_forecast = es_forecast + arima_forecast

# 5. Сравнение
comparison = pd.DataFrame({
    'Actual': test,
    'ES Forecast': es_forecast,
    'Hybrid Forecast': hybrid_forecast
})

from sklearn.metrics import mean_absolute_percentage_error
mape_es = mean_absolute_percentage_error(test, es_forecast) * 100
mape_hybrid = mean_absolute_percentage_error(test, hybrid_forecast) * 100

print(f"MAPE ES: {mape_es:.2f}% | MAPE Hybrid: {mape_hybrid:.2f}%")
