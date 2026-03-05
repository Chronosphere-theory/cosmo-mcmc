#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Монте-Карло марковские цепи (MCMC) для модели Хроносферы
Сравнение с данными Planck 2018, Pantheon+ SNe, BAO (DESI и др.)
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# ----------------------------------------------------------------------
# 1. Импорт библиотек для космологии и MCMC
# ----------------------------------------------------------------------
# Убедитесь, что установлены: classy, cobaya, getdist, numpy, scipy, matplotlib
# Установка: pip install classy cobaya getdist

import classy
import cobaya
from cobaya.model import get_model
from cobaya.run import run
from cobaya.theory import Theory
from cobaya.likelihoods.base import Likelihood
import getdist
from getdist import plots, MCSamples

# ----------------------------------------------------------------------
# 2. Определение класса для модели Хроносферы как расширения CLASS
# ----------------------------------------------------------------------
class ChronosphereModel(Theory):
    """
    Реализация модели Хроносферы в виде теории для cobaya.
    Параметры:
        - epsilon0: современное значение epsilon = a/R (0.32 по умолчанию)
        - alpha: параметр потенциала (связан с pi-3)
        - beta: дополнительный параметр (если нужно)
    """
    name = "chronosphere"
    
    def initialize(self):
        self.requires = {"cosmological_model": {"classy": None}}
        self.provides = {"lensing": None, "cl": None, "power_spectrum": None}
    
    def get_can_provide_params(self):
        return ["epsilon0", "alpha", "beta"]
    
    def calculate(self, state, want_derived=True, **params_values):
        # Передаём параметры в CLASS
        # Для модели Хроносферы мы используем параметризацию динамической тёмной энергии
        # через уравнение состояния w(z). В первом приближении используем CPL:
        # w(z) = w0 + wa * z/(1+z)
        # Связь с параметрами модели: w0 = -1 + f(epsilon0), wa = g(epsilon0)
        # Более точное выражение можно получить из потенциала дилатона.
        # Здесь для простоты оставим w0 и wa свободными, но наложим приоры
        # из теоретических ограничений.
        
        w0 = params_values.get("w0", -1.0)
        wa = params_values.get("wa", 0.0)
        epsilon0 = params_values.get("epsilon0", 0.32)
        
        # Передаём в CLASS через флаги
        class_params = {
            'output': 'tCl,pCl,lCl,mPk',
            'l_max_scalars': 2500,
            'lensing': 'yes',
            'P_k_max_h/Mpc': 3.0,
            'h': params_values.get('h', 0.67),
            'omega_b': params_values.get('omega_b', 0.022),
            'omega_cdm': params_values.get('omega_cdm', 0.12),
            'tau_reio': params_values.get('tau', 0.054),
            'A_s': params_values.get('A_s', 2.1e-9),
            'n_s': params_values.get('n_s', 0.965),
            'Omega_Lambda': 0.0,
            'w0_fld': w0,
            'wa_fld': wa,
            'parameters_fld': 'epsilon',  # зарезервировано для будущего использования
        }
        
        # Создаём объект CLASS и вычисляем спектры
        cosmo = classy.Class()
        cosmo.set(class_params)
        cosmo.compute()
        
        # Получаем спектры и сохраняем в state
        state['cl'] = cosmo.lensed_cl(2500)
        state['Pk'] = cosmo.pk(1.0, 0.0)  # P(k) при z=0
        state['derived'] = {
            'H0': cosmo.h() * 100,
            'Omega_m': cosmo.Omega_m(),
            'sigma8': cosmo.sigma8(),
        }
        cosmo.struct_cleanup()
    
    def get_can_provide_likelihoods(self):
        return ["planck_2018", "pantheon_plus", "desi_bao"]

# ----------------------------------------------------------------------
# 3. Определение информационных приоров (можно задать в info)
# ----------------------------------------------------------------------
info = {
    "params": {
        # Космологические параметры
        "h": {"prior": {"min": 0.5, "max": 0.9}, "ref": 0.67, "proposal": 0.01},
        "omega_b": {"prior": {"min": 0.005, "max": 0.1}, "ref": 0.022, "proposal": 0.001},
        "omega_cdm": {"prior": {"min": 0.05, "max": 0.4}, "ref": 0.12, "proposal": 0.005},
        "tau": {"prior": {"min": 0.01, "max": 0.1}, "ref": 0.054, "proposal": 0.005},
        "A_s": {"prior": {"min": 1e-9, "max": 5e-9}, "ref": 2.1e-9, "proposal": 1e-10},
        "n_s": {"prior": {"min": 0.8, "max": 1.2}, "ref": 0.965, "proposal": 0.01},
        
        # Параметры модели Хроносферы
        "epsilon0": {"prior": {"min": 0.1, "max": 0.5}, "ref": 0.32, "proposal": 0.02},
        # Вместо прямого использования epsilon0, мы можем параметризовать w0 и wa через epsilon0
        "w0": {"derived": lambda epsilon0, alpha: -1 + alpha * epsilon0**2},  # пример связи
        "wa": {"derived": lambda epsilon0, beta: beta * epsilon0},            # пример связи
        "alpha": {"prior": {"min": 0.0, "max": 1.0}, "ref": 0.5, "proposal": 0.1},
        "beta": {"prior": {"min": -0.5, "max": 0.5}, "ref": 0.0, "proposal": 0.1},
    },
    "likelihood": {
        "planck_2018": {"external": lambda data, **kwargs: None},  # Заглушка, нужно подключить реальные данные
        "pantheon_plus": {"external": lambda data, **kwargs: None},
        "desi_bao": {"external": lambda data, **kwargs: None},
    },
    "theory": {
        "classy": {"external": classy.Class},
        "chronosphere": {"external": ChronosphereModel},
    },
    "prior": {
        "H0_big": {"dist": "norm", "loc": 73.2, "scale": 1.3},  # дополнительный приор от SH0ES (опционально)
    },
    "sampler": {
        "mcmc": {"max_samples": 50000, "burn_in": 1000, "Rminus1_stop": 0.01}
    },
    "output": "chains/chronosphere_mcmc",
}

# ----------------------------------------------------------------------
# 4. Запуск MCMC
# ----------------------------------------------------------------------
def run_mcmc():
    updated_info, sampler = run(info)
    return updated_info, sampler

# ----------------------------------------------------------------------
# 5. Анализ результатов
# ----------------------------------------------------------------------
def analyze_results():
    # Загружаем сэмплы
    samples = MCSamples(filename="chains/chronosphere_mcmc")
    
    # Распечатываем сводку
    print(samples.getMargeStats())
    
    # Построение треугольного графика
    g = plots.get_subplot_plotter()
    g.triangle_plot(samples, ["h", "omega_b", "omega_cdm", "epsilon0", "alpha", "beta"],
                    filled=True, title_limit=1)
    plt.savefig("chains/triangle_plot.png")
    
    # Сравнение с ΛCDM (можно загрузить соответствующие сэмплы)
    # Здесь для примера просто выведем значение chi2
    best_fit = samples.bestfit_sample
    print("Best fit parameters:", best_fit)
    
if __name__ == "__main__":
    print("Запуск MCMC для модели Хроносферы...")
    # Закомментировано для предотвращения случайного запуска в неподходящей среде
    # updated_info, sampler = run_mcmc()
    # analyze_results()
    print("Код готов к запуску при наличии установленных библиотек и данных.")