# ==== 1. Библиотеки ====
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf

# ==== 2. Функция для проверки сезонности ====
def check_seasonality(series: pd.Series, period: int = 12, alpha: float = 0.1):
    """
    Проверка наличия сезонности во временном ряде turnover.

    series : pandas.Series
        Временной ряд turnover (индекс = даты).
    period : int
        Длина цикла (12 = месяцы, 4 = кварталы, 7 = недели и т.п.)
    alpha : float
        Порог силы сезонности (доля объяснённой дисперсии).

    Возвращает строку с заключением и строит ACF + STL-график.
    """

    # --- подготовка данных ---
    series = series.asfreq(pd.infer_freq(series.index) or "MS").interpolate()

    # --- ACF ---
    fig, ax = plt.subplots(figsize=(10, 4))
    plot_acf(series, lags=period * 3, ax=ax)
    plt.title("ACF для проверки сезонности")
    plt.show()

    # --- STL-декомпозиция ---
    stl = STL(series, period=period, robust=True).fit()
    season_strength = 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid)

    # --- Заключение ---
    if season_strength > alpha:
        result = (f"✔ Обнаружена сезонность с периодом {period}. "
                  f"Сезонная компонента объясняет ~{season_strength:.0%} дисперсии.")
    else:
        result = (f"✖ Явная сезонность с периодом {period} не выявлена. "
                  f"Сезонная компонента объясняет лишь ~{season_strength:.0%} дисперсии.")

    # --- График STL ---
    stl.plot()
    plt.suptitle("STL decomposition", fontsize=12)
    plt.show()

    return result


# ==== 3. Заведение данных ====
# Пример: загрузка из Excel
# df = pd.read_excel("turnover.xlsx")  

# Для примера сделаем искусственный ряд
dates = pd.date_range("2020-01-01", periods=48, freq="MS")  # 4 года, месяцы
np.random.seed(42)
data = 0.02 + 0.01 * np.sin(2 * np.pi * dates.month / 12) + 0.005 * np.random.randn(len(dates))
df = pd.DataFrame({"date": dates, "turnover": data})

# ==== 4. Подготовка индекса ====
df["date"] = pd.to_datetime(df["date"])
df = df.set_index("date")

# ==== 5. Вызов функции ====
conclusion = check_seasonality(df["turnover"], period=12)
print(conclusion)

