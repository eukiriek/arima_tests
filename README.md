import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose, STL

# --- подготовка данных ---
# df: колонки date, turnover (если имена другие – замените)
df = df.rename(columns=lambda c: c.strip().lower())
df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
df["turnover"] = pd.to_numeric(df["turnover"], errors="coerce")
df = (df.dropna(subset=["date","turnover"])
        .sort_values("date")
        .set_index("date"))

freq = pd.infer_freq(df.index) or "MS"      # месяцы (начало месяца)
series = df["turnover"].asfreq(freq).interpolate()

S = 12  # годовая


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL
from statsmodels.graphics.tsaplots import plot_acf

def check_seasonality(series: pd.Series, period: int = 12, alpha: float = 0.1):
    """
    Проверка наличия сезонности во временном ряде.
    
    series : pandas.Series
        Временной ряд turnover (индекс = даты).
    period : int
        Длина предполагаемого цикла (12 = месяцы, 4 = кварталы, 7 = недели и т.п.)
    alpha : float
        Порог для силы сезонности (доля дисперсии, объясняемая сезонной компонентой).
    
    Возвращает строку с заключением и показывает ACF + STL-график.
    """
    
    # Убираем пропуски и приводим к регулярной частоте
    series = series.asfreq(pd.infer_freq(series.index) or "MS").interpolate()

    # --- 1. ACF-график ---
    fig, ax = plt.subplots(figsize=(10,4))
    plot_acf(series, lags=period*3, ax=ax)
    plt.title("ACF для проверки сезонности")
    plt.show()
    
    # --- 2. STL-декомпозиция ---
    stl = STL(series, period=period, robust=True).fit()
    season_strength = 1 - np.var(stl.resid) / np.var(stl.seasonal + stl.resid)
    
    # --- 3. Заключение ---
    if season_strength > alpha:
        result = (f"✔ Обнаружена сезонность с периодом {period}. "
                  f"Сезонная компонента объясняет ~{season_strength:.0%} дисперсии.")
    else:
        result = (f"✖ Явная сезонность с периодом {period} не выявлена. "
                  f"Сезонная компонента объясняет лишь ~{season_strength:.0%} дисперсии.")
    
    # график STL
    stl.plot()
    plt.suptitle("STL decomposition", fontsize=12)
    plt.show()
    
    return result


