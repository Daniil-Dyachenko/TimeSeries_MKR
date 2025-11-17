"""
Модульна контрольна робота TimeSeries з навчальної дисципліни
«Аналіз та обробка часових рядів (Time Series)»
"""


import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
import os
import requests

matplotlib.use('Agg')
warnings.filterwarnings('ignore')
plt.rcParams['axes.unicode_minus'] = False

def parse_minfin_living_wage(url='https://index.minfin.com.ua/ua/labour/wagemin/',
                             use_backup=True, save_data=True):
    """Парсинг прожиткового мінімуму з сайту Minfin"""
    print("-" * 80)
    print(" ПАРСИНГ РЕАЛЬНИХ ДАНИХ З САЙТУ MINFIN.COM.UA")

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
        'Accept-Language': 'uk-UA,uk;q=0.9',
    }

    try:
        print(f"Спроба парсингу: {url}")
        response = requests.get(url, headers=headers, timeout=10)
        print(f"Статус відповіді: {response.status_code}")

        if response.status_code == 200:
            df_list = pd.read_html(response.text)
            if df_list and len(df_list) > 0:
                df = df_list[0]
                print(f" Знайдено таблицю з {len(df)} записами")
                parsed_data = process_minfin_data(df)

                if save_data:
                    save_parsed_data(parsed_data)
                return parsed_data
    except Exception as e:
        print(f"Помилка парсингу: {e}")

    if use_backup:
        print("\nВикористовуємо резервні реалістичні дані")
        return generate_realistic_minfin_data()
    else:
        raise Exception("Парсинг не вдався і резервні дані вимкнено")


def process_minfin_data(df):
    """Обробка даних після парсингу"""
    import re

    df = df.dropna()
    df.columns = ['period', 'total', 'children_under_6', 'children_6_18',
                  'working_age', 'disabled']

    dates, values_total = [], []

    for idx, row in df.iterrows():
        period_str = str(row['period'])
        total_val = row['total']

        match = re.search(r'з\s+(\d{2})\.(\d{2})\.(\d{4})', period_str)
        if match:
            day, month, year = match.group(1), int(match.group(2)), int(match.group(3))
            date_str = f"{day}.{month:02d}.{year}"

            try:
                date = pd.to_datetime(date_str, format='%d.%m.%Y')
                dates.append(date)
                values_total.append(float(total_val))
            except:
                continue

    if dates:
        df_processed = pd.DataFrame({
            'date': dates,
            'living_wage': values_total
        }).sort_values('date').reset_index(drop=True)

        print(f" Оброблено: {len(df_processed)} записів")
        print(f"  Період: {df_processed['date'].min().date()} → {df_processed['date'].max().date()}")
        print(f"  Діапазон: {df_processed['living_wage'].min():.0f} - {df_processed['living_wage'].max():.0f} грн")

        return df_processed
    else:
        raise Exception("Не вдалося обробити дати з таблиці")


def generate_realistic_minfin_data():
    """Генерація реалістичних даних як резерв"""
    dates = pd.date_range(start='2000-01-01', end='2025-01-01', freq='QS')
    n = len(dates)

    base_value = 270
    years_passed = np.arange(n) / 4
    growth_rate = 0.08
    living_wage = base_value * np.exp(growth_rate * years_passed)
    noise = np.random.normal(0, 20, n)
    living_wage = np.round(living_wage + noise).astype(int)

    df = pd.DataFrame({'date': dates, 'living_wage': living_wage})

    print(f"Згенеровано {len(df)} записів (РЕЗЕРВНІ ДАНІ)")
    print(f"Період: {dates[0].date()} → {dates[-1].date()}")
    print(f"Діапазон: {living_wage.min()} - {living_wage.max()} грн")

    return df


def save_parsed_data(df, directory='Parsed_Data'):
    """Збереження даних у CSV"""
    os.makedirs(directory, exist_ok=True)
    filepath = os.path.join(directory, "parsed_minfin_data.csv")
    df.to_csv(filepath, index=False, encoding='utf-8-sig')
    print(f" Файл збережено: {filepath}\n")
    return filepath

class AlphaBetaFilter:
    """α-β фільтр (скалярний фільтр Калмана для лінійної моделі)"""

    def __init__(self, T0=1.0):
        """
        Parameters:
        -----------
        T0 : float - період оновлення інформації
        """
        self.T0 = T0
        self.y_hat = None
        self.y_dot = None
        self.n = 0

    def predict(self):
        """Екстраполяція на наступний крок"""
        if self.y_hat is None:
            return None
        return self.y_hat + self.y_dot * self.T0

    def update(self, y_measured):
        """Оновлення оцінок за новим виміром"""
        self.n += 1

        if self.y_hat is None:
            self.y_hat = y_measured
            self.y_dot = 0.0
            return self.y_hat

        alpha_calc = (2 * (2 * self.n - 1)) / (self.n * (self.n + 1))
        beta_calc = (6 / (self.n * (self.n + 1)))

        alpha = max(0.15, alpha_calc)
        beta = max(0.08, beta_calc)

        y_extrap = self.y_hat + self.y_dot * self.T0

        residual = y_measured - y_extrap

        self.y_hat = y_extrap + alpha * residual
        self.y_dot = self.y_dot + (beta / self.T0) * residual

        return self.y_hat

    def get_state(self):
        """Поточний стан фільтра"""
        return {'position': self.y_hat, 'velocity': self.y_dot, 'n': self.n}


class AlphaBetaGammaFilter:
    """α-β-γ фільтр (поліном 2-го порядку)"""

    def __init__(self, T0=1.0):
        """
        Parameters:
        -----------
        T0 : float - період оновлення інформації
        """
        self.T0 = T0
        self.z0 = None
        self.z1 = None
        self.z2 = None
        self.n = 0

    def predict(self):
        """Екстраполяція"""
        if self.z0 is None:
            return None
        return self.z0 + self.z1 * self.T0 + 0.5 * self.z2 * (self.T0 ** 2)

    def update(self, z_measured):
        """Оновлення оцінок"""
        self.n += 1

        if self.z0 is None:
            self.z0 = z_measured
            self.z1 = 0.0
            self.z2 = 0.0
            return self.z0

        n = self.n
        alpha = (2 * (3 * n - 2)) / (n * (n + 1) * (2 * n + 1))
        beta = (6 * (2 * n - 1)) / (n * (n + 1) * (2 * n + 1)) / self.T0
        gamma = (60 / (n * (n + 1) * (2 * n + 1))) / (self.T0 ** 2)

        z_extrap = self.z0 + self.z1 * self.T0 + 0.5 * self.z2 * (self.T0 ** 2)

        residual = z_measured - z_extrap

        self.z0 = z_extrap + alpha * residual
        self.z1 = self.z1 + beta * residual
        self.z2 = self.z2 + gamma * residual

        return self.z0

    def get_state(self):
        """Поточний стан"""
        return {'position': self.z0, 'velocity': self.z1,
                'acceleration': self.z2, 'n': self.n}


class KalmanFilter:
    """Фільтр Калмана (повна матрична форма для лінійної моделі)"""

    def __init__(self, T0=1.0, sigma_measurement=50.0):
        """
        Parameters:
        -----------
        T0 : float - період оновлення
        sigma_measurement : float - СКВ вимірів
        """
        self.T0 = T0

        self.state = np.array([0.0, 0.0])

        self.F = np.array([
            [1, T0],
            [0, 1]
        ])

        self.H = np.array([[1, 0]])

        self.P = np.array([
            [1000.0, 0.0],
            [0.0, 100.0]
        ])

        self.R = np.array([[sigma_measurement ** 2]])

        q = 0.1
        self.Q = q * np.array([
            [T0 ** 4 / 4, T0 ** 3 / 2],
            [T0 ** 3 / 2, T0 ** 2]
        ])

        self.initialized = False

    def predict(self):
        """Крок передбачення"""
        self.state = self.F @ self.state

        self.P = self.F @ self.P @ self.F.T + self.Q

        return self.state[0]

    def update(self, z_measured):
        """Крок оновлення за виміром"""
        if not self.initialized:
            self.state[0] = z_measured
            self.initialized = True
            return z_measured

        self.predict()

        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        y = z_measured - (self.H @ self.state)[0]

        self.state = self.state + K.flatten() * y

        I = np.eye(2)
        self.P = (I - K @ self.H) @ self.P

        return self.state[0]

    def get_state(self):
        """Поточний стан"""
        return {'position': self.state[0], 'velocity': self.state[1]}

def apply_recursive_filters(data, T0=1.0, forecast_steps=10):
    """
    Застосування всіх трьох фільтрів до даних
    """
    print("\n" + "-" * 80)
    print("РЕКУРЕНТНЕ ЗГЛАДЖУВАННЯ")

    n = len(data)

    ab_filter = AlphaBetaFilter(T0=T0)
    abg_filter = AlphaBetaGammaFilter(T0=T0)
    kf = KalmanFilter(T0=T0, sigma_measurement=np.std(np.diff(data)))

    ab_smoothed = np.zeros(n)
    abg_smoothed = np.zeros(n)
    kf_smoothed = np.zeros(n)

    print(f"\nОбробка {n} вимірів...")
    for i in range(n):
        ab_smoothed[i] = ab_filter.update(data[i])
        abg_smoothed[i] = abg_filter.update(data[i])
        kf_smoothed[i] = kf.update(data[i])

    print(" Згладжування завершено")

    print(f"\nПрогнозування на {forecast_steps} кроків вперед...")

    ab_forecast = np.zeros(forecast_steps)
    abg_forecast = np.zeros(forecast_steps)
    kf_forecast = np.zeros(forecast_steps)

    # α-β прогноз
    ab_state = ab_filter.get_state()
    for i in range(forecast_steps):
        ab_forecast[i] = ab_state['position'] + ab_state['velocity'] * T0 * (i + 1)

    # α-β-γ прогноз
    abg_state = abg_filter.get_state()
    for i in range(forecast_steps):
        t = T0 * (i + 1)
        abg_forecast[i] = (abg_state['position'] +
                           abg_state['velocity'] * t +
                           0.5 * abg_state['acceleration'] * t ** 2)

    # Калман прогноз
    kf_state_backup = kf.state.copy()
    kf_P_backup = kf.P.copy()
    for i in range(forecast_steps):
        kf_forecast[i] = kf.predict()
    kf.state = kf_state_backup
    kf.P = kf_P_backup

    print(" Прогнозування завершено")

    metrics = {}
    for name, smoothed in [('α-β', ab_smoothed),
                           ('α-β-γ', abg_smoothed),
                           ('Kalman', kf_smoothed)]:
        mae = mean_absolute_error(data, smoothed)
        rmse = np.sqrt(mean_squared_error(data, smoothed))
        r2 = r2_score(data, smoothed)

        metrics[name] = {'MAE': mae, 'RMSE': rmse, 'R²': r2}

    return {
        'original': data,
        'ab_smoothed': ab_smoothed,
        'abg_smoothed': abg_smoothed,
        'kf_smoothed': kf_smoothed,
        'ab_forecast': ab_forecast,
        'abg_forecast': abg_forecast,
        'kf_forecast': kf_forecast,
        'metrics': metrics,
        'n': n,
        'forecast_steps': forecast_steps
    }


def print_metrics(metrics):
    """Виведення метрик якості"""
    print("\n" + "-" * 80)
    print(" МЕТРИКИ ЯКОСТІ ЗГЛАДЖУВАННЯ")
    print(f"\n{'Метод':<15} {'MAE':>12} {'RMSE':>12} {'R²':>12}")
    print("-" * 53)

    for name, m in metrics.items():
        print(f"{name:<15} {m['MAE']:>12.2f} {m['RMSE']:>12.2f} {m['R²']:>12.4f}")

    best_method = min(metrics.items(), key=lambda x: x[1]['RMSE'])
    print(f"\n Найкращий метод за RMSE: {best_method[0]}")


def analyze_filter_convergence(results):
    """Аналіз збіжності фільтрів """
    print("\n" + "-" * 80)
    print(" АНАЛІЗ ЗБІЖНОСТІ ФІЛЬТРІВ")

    data = results['original']
    n = results['n']

    part_size = n // 3

    print(f"\nАналіз розподілено на 3 періоди по ~{part_size} точок:")
    print("  • Початковий період (адаптація фільтра)")
    print("  • Середній період (стабілізація)")
    print("  • Кінцевий період (усталений режим)")
    print("\nВикористовується MAPE (Mean Absolute Percentage Error) для коректного аналізу")

    for name, smoothed in [('α-β', results['ab_smoothed']),
                           ('α-β-γ', results['abg_smoothed']),
                           ('Kalman', results['kf_smoothed'])]:

        def calc_mape(actual, predicted):
            mask = actual != 0
            return np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100

        mae_start = np.mean(np.abs(data[:part_size] - smoothed[:part_size]))
        mae_mid = np.mean(np.abs(data[part_size:2 * part_size] - smoothed[part_size:2 * part_size]))
        mae_end = np.mean(np.abs(data[2 * part_size:] - smoothed[2 * part_size:]))

        mape_start = calc_mape(data[:part_size], smoothed[:part_size])
        mape_mid = calc_mape(data[part_size:2 * part_size], smoothed[part_size:2 * part_size])
        mape_end = calc_mape(data[2 * part_size:], smoothed[2 * part_size:])

        print(f"\n{name}:")
        print(f"  Початковий період (точки 1-{part_size}):")
        print(f"    MAE = {mae_start:.2f} грн | MAPE = {mape_start:.2f}%")
        print(f"  Середній період (точки {part_size + 1}-{2 * part_size}):")
        print(f"    MAE = {mae_mid:.2f} грн | MAPE = {mape_mid:.2f}%")
        print(f"  Кінцевий період (точки {2 * part_size + 1}-{n}):")
        print(f"    MAE = {mae_end:.2f} грн | MAPE = {mape_end:.2f}%")

        mape_change = mape_end - mape_start

        print(f"\n  Динаміка відносної помилки:")
        if mape_end < mape_start:
            improvement = (mape_start - mape_end)
            print(f"Фільтр покращується: MAPE зменшилася на {improvement:.2f}%")
        elif abs(mape_change) < 0.5:
            print(f"Фільтр стабільний: MAPE практично незмінна (~{mape_start:.2f}%)")
        else:
            increase = mape_end - mape_start
            print(f"MAPE зросла на {increase:.2f}%")
            print(f"    (з {mape_start:.2f}% до {mape_end:.2f}%)")
            if increase < 2.0:
                print(f"Фільтр адекватно працює, зростання помірне")
            else:
                print(f"Фільтр має труднощі з адаптацією до тренду")

def plot_results(results, dates=None, save_dir='Graphs'):
    """Візуалізація результатів"""
    os.makedirs(save_dir, exist_ok=True)

    print("-" * 80)
    print(" ВІЗУАЛІЗАЦІЯ РЕЗУЛЬТАТІВ")
    print("-" * 80)

    n = results['n']
    forecast_steps = results['forecast_steps']

    if dates is not None:
        x_train = dates
        last_date = dates.iloc[-1]
        freq = pd.infer_freq(dates)
        if freq is None:
            freq = 'Q'
        x_forecast = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=freq)[1:]
    else:
        x_train = np.arange(n)
        x_forecast = np.arange(n, n + forecast_steps)

    fig, ax = plt.subplots(figsize=(14, 6))

    ax.plot(x_train, results['original'], 'k.', alpha=0.4, markersize=4,
            label='Вихідні дані', zorder=1)
    ax.plot(x_train, results['ab_smoothed'], 'b-', linewidth=2,
            label='α-β фільтр', alpha=0.8, zorder=3)
    ax.plot(x_train, results['abg_smoothed'], 'g-', linewidth=2,
            label='α-β-γ фільтр', alpha=0.8, zorder=4)
    ax.plot(x_train, results['kf_smoothed'], 'r-', linewidth=2,
            label='Фільтр Калмана', alpha=0.8, zorder=5)

    ax.set_title('Порівняння методів рекурентного згладжування',
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Час' if dates is not None else 'Номер виміру', fontsize=11)
    ax.set_ylabel('Значення (грн)', fontsize=11)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)

    if dates is not None:
        plt.xticks(rotation=45)

    plt.tight_layout()
    filename = '1_smoothing_comparison.png'
    fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f" Збережено: {save_dir}/{filename}")
    plt.close()

    fig, axes = plt.subplots(3, 1, figsize=(14, 12))

    methods = [
        ('α-β', results['ab_smoothed'], results['ab_forecast'], 'blue'),
        ('α-β-γ', results['abg_smoothed'], results['abg_forecast'], 'green'),
        ('Kalman', results['kf_smoothed'], results['kf_forecast'], 'red')
    ]

    for idx, (name, smoothed, forecast, color) in enumerate(methods):
        ax = axes[idx]

        ax.plot(x_train, results['original'], 'k.', alpha=0.3, markersize=4,
                label='Вихідні дані')
        ax.plot(x_train, smoothed, color=color, linewidth=2,
                label=f'{name} фільтр (згладжування)', alpha=0.8)

        if dates is not None:
            ax.plot(x_forecast, forecast, '--', color=color, linewidth=2.5,
                    label='Прогноз', alpha=0.9)
            ax.axvline(x=x_train.iloc[-1], color='orange', linestyle=':',
                       linewidth=2, alpha=0.7, label='Межа даних')
        else:
            ax.plot(x_forecast, forecast, '--', color=color, linewidth=2.5,
                    label='Прогноз', alpha=0.9)
            ax.axvline(x=n - 1, color='orange', linestyle=':', linewidth=2,
                       alpha=0.7, label='Межа даних')

        mae = results['metrics'][name]['MAE']
        r2 = results['metrics'][name]['R²']

        ax.set_title(f'{name} фільтр (MAE={mae:.2f}, R²={r2:.4f})',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('Час' if dates is not None else 'Номер виміру', fontsize=10)
        ax.set_ylabel('Значення (грн)', fontsize=10)
        ax.legend(loc='best', fontsize=9)
        ax.grid(True, alpha=0.3)

        if dates is not None:
            ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    filename = '2_forecasting_all_methods.png'
    fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f" Збережено: {save_dir}/{filename}")
    plt.close()

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))

    for idx, (name, smoothed, color) in enumerate([
        ('α-β', results['ab_smoothed'], 'blue'),
        ('α-β-γ', results['abg_smoothed'], 'green'),
        ('Kalman', results['kf_smoothed'], 'red')
    ]):
        residuals = results['original'] - smoothed

        axes[0, idx].plot(residuals, color=color, alpha=0.6, linewidth=1)
        axes[0, idx].axhline(y=0, color='black', linestyle='--', linewidth=1.5)
        axes[0, idx].set_title(f'{name}: Залишки у часі', fontsize=10, fontweight='bold')
        axes[0, idx].set_xlabel('Номер виміру')
        axes[0, idx].set_ylabel('Залишок')
        axes[0, idx].grid(True, alpha=0.3)

        axes[1, idx].hist(residuals, bins=20, color=color, alpha=0.7, edgecolor='black')
        axes[1, idx].axvline(x=0, color='red', linestyle='--', linewidth=2)
        axes[1, idx].set_title(f'{name}: Розподіл залишків', fontsize=10, fontweight='bold')
        axes[1, idx].set_xlabel('Залишок')
        axes[1, idx].set_ylabel('Частота')
        axes[1, idx].grid(True, alpha=0.3, axis='y')

        mean_res = np.mean(residuals)
        std_res = np.std(residuals)
        axes[1, idx].text(0.02, 0.98, f'μ = {mean_res:.2f}\nσ = {std_res:.2f}',
                          transform=axes[1, idx].transAxes,
                          verticalalignment='top',
                          bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                          fontsize=9)

    plt.tight_layout()
    filename = '3_residuals_analysis.png'
    fig.savefig(os.path.join(save_dir, filename), dpi=300, bbox_inches='tight')
    print(f" Збережено: {save_dir}/{filename}")
    plt.close()

    print(f"\n Усі графіки збережено в папці: {save_dir}/")

def main():
    """Головна функція"""
    print("\n" + "-" * 80)
    print(" Модульна контрольна робота : Рекурентне згладжування часових рядів")
    print("\nМетоди:")
    print("  1. α-β фільтр (скалярний фільтр Калмана)")
    print("  2. α-β-γ фільтр (поліном 2-го порядку)")
    print("  3. Фільтр Калмана (повна матрична форма)")
    print("\n" + "=" * 80)

    df = parse_minfin_living_wage(use_backup=True, save_data=True)

    data = df['living_wage'].values
    dates = df['date']

    if len(dates) > 1:
        time_diffs = dates.diff().dropna()
        avg_days = time_diffs.mean().days
        T0 = avg_days / 30.0
    else:
        T0 = 1.0

    print(f"\nПараметри:")
    print(f"  Кількість вимірів: {len(data)}")

    forecast_steps = max(5, len(data) // 4)
    print(f"  Кроків прогнозу: {forecast_steps}")

    results = apply_recursive_filters(data, T0=T0, forecast_steps=forecast_steps)

    print_metrics(results['metrics'])

    analyze_filter_convergence(results)

    print("\n" + "-" * 80)
    print(" АНАЛІЗ ПРОГНОЗІВ")

    last_value = data[-1]

    for name, forecast in [('α-β', results['ab_forecast']),
                           ('α-β-γ', results['abg_forecast']),
                           ('Kalman', results['kf_forecast'])]:

        forecast_change = (forecast[-1] - last_value) / last_value * 100
        avg_forecast = np.mean(forecast)

        print(f"\n{name}:")
        print(f"  Останнє значення: {last_value:.2f} грн")
        print(f"  Прогноз через {forecast_steps} кроків: {forecast[-1]:.2f} грн")
        print(f"  Зміна: {forecast_change:+.1f}%")
        print(f"  Середнє прогнозне: {avg_forecast:.2f} грн")

    print("\n" + "-" * 80)
    print(" ПОРІВНЯННЯ З ПРОСТОЮ ЛІНІЙНОЮ ЕКСТРАПОЛЯЦІЄЮ")

    train_size = int(len(data) * 0.7)
    x_simple = np.arange(len(data) - train_size)
    y_simple = data[train_size:]

    coeffs_simple = np.polyfit(x_simple, y_simple, 1)

    x_forecast_simple = np.arange(len(data) - train_size,
                                  len(data) - train_size + forecast_steps)
    simple_forecast = np.polyval(coeffs_simple, x_forecast_simple)

    simple_last = simple_forecast[-1]
    simple_change = (simple_last - last_value) / last_value * 100

    print(f"\nПроста лінійна екстраполяція:")
    print(f"  Прогноз через {forecast_steps} кроків: {simple_last:.2f} грн")
    print(f"  Зміна: {simple_change:+.1f}%")

    print(f"\nПорівняння з рекурентними методами:")
    for name, forecast in [('α-β', results['ab_forecast']),
                           ('α-β-γ', results['abg_forecast']),
                           ('Kalman', results['kf_forecast'])]:
        diff = abs(forecast[-1] - simple_last)
        print(f"  {name} vs проста: різниця {diff:.2f} грн ({diff / simple_last * 100:.1f}%)")

    plot_results(results, dates=dates, save_dir='Graphs')

    print("\n" + "-" * 80)
    print(" ЗБЕРЕЖЕННЯ ДАНИХ")
    print("-" * 80)

    results_df = pd.DataFrame({
        'date': dates,
        'original': results['original'],
        'ab_smoothed': results['ab_smoothed'],
        'abg_smoothed': results['abg_smoothed'],
        'kalman_smoothed': results['kf_smoothed']
    })

    if len(dates) > 1:
        freq = pd.infer_freq(dates)
        if freq is None:
            freq = 'Q'

        last_date = dates.iloc[-1]
        forecast_dates = pd.date_range(start=last_date, periods=forecast_steps + 1, freq=freq)[1:]

        forecast_df = pd.DataFrame({
            'date': forecast_dates,
            'ab_forecast': results['ab_forecast'],
            'abg_forecast': results['abg_forecast'],
            'kalman_forecast': results['kf_forecast']
        })

    os.makedirs('Data', exist_ok=True)

    results_df.to_csv('Data/smoothed_results.csv', index=False, encoding='utf-8-sig')
    print(" Збережено згладжені дані: Data/smoothed_results.csv")

    if len(dates) > 1:
        forecast_df.to_csv('Data/forecast_results.csv', index=False, encoding='utf-8-sig')
        print(" Збережено прогнози: Data/forecast_results.csv")

    print("\n" + "-" * 80)
    print(" Робота завершена успішно!")

if __name__ == "__main__":
    main()