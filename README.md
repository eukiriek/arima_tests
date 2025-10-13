def mape(y_true, y_pred):
    # защита от деления на ноль
    denom = np.where(np.abs(y_true) < 1e-12, 1e-12, np.abs(y_true))
    return np.mean(np.abs((y_true - y_pred) / denom)) * 100

def evaluate_and_plot(model_name, fit_obj, forecast, color_actual='black'):
    mae = mean_absolute_error(test['value'], forecast)
    rmse = mean_squared_error(test['value'], forecast, squared=False)
    mp = mape(test['value'].values, forecast.values)
    print(f"{model_name} -> MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mp:.2f}%")

    plt.figure(figsize=(10,4))
    plt.plot(train.index, train['value'], label='train')
    plt.plot(test.index,  test['value'],  label='test', linewidth=2)
    plt.plot(test.index,  forecast,       label=f'{model_name} forecast', linestyle='--')
    plt.title(f'{model_name}: прогноз на {h} шагов')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Оценим SES
evaluate_and_plot("SES", ses_fit, ses_forecast)

# Оценим Holt
evaluate_and_plot("Holt", holt_fit, holt_forecast)

# (Опционально) Holt-Winters
# evaluate_and_plot("Holt-Winters", hw_fit, hw_forecast)
