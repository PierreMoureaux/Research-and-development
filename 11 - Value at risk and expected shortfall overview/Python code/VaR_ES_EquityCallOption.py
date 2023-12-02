import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, chi2, kurtosis, skew
from math import exp, log, sqrt
import pandas as pd

def call_option_delta(S:float, K:float, T_opt:float, r:float, sigma_implied:float)->float:
    """
    Calculate the delta of a European call option using the Black-Scholes formula.

    Parameters:
    S (float): Current price of the underlying asset
    K (float): Strike price of the option+
    T_opt (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma_implied (float): Black-Scholes volatility

    Returns:
    float: Delta of the call option
    """
    d1 = (log(S / K) + (r + 0.5 * sigma_implied**2) * T_opt) / (sigma_implied * sqrt(T_opt))
    delta = norm.cdf(d1)

    return delta

def call_option_theta(S:float, K:float, T_opt:float, r:float, sigma_implied:float)->float:
    """
    Calculate the theta of a European call option using the Black-Scholes formula.

    Parameters:
    S (float): Current price of the underlying asset
    K (float): Strike price of the option
    T_opt (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma_implied (float): Black-Scholes volatility

    Returns:
    float: Theta of the call option
    """
    d1 = (log(S / K) + (r + 0.5 * sigma_implied**2) * T_opt) / (sigma_implied * sqrt(T_opt))
    d2 = d1 - sigma_implied * sqrt(T_opt)

    theta = -((S * norm.pdf(d1) * sigma_implied) / (2 * sqrt(T_opt))) - r * K * exp(-r * T_opt) * norm.cdf(d2)

    return theta

def call_option_gamma(S:float, K:float, T_opt:float, r:float, sigma_implied:float)->float:
    """
    Calculate the gamma of a European call option using the Black-Scholes formula.

    Parameters:
    S (float): Current price of the underlying asset
    K (float): Strike price of the option
    T_opt (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma_implied (float): Black-Scholes volatility

    Returns:
    float: Gamma of the call option
    """
    d1 = (log(S / K) + (r + 0.5 * sigma_implied**2) * T_opt) / (sigma_implied * sqrt(T_opt))
    gamma = norm.pdf(d1) / (S * sigma_implied * sqrt(T_opt))

    return gamma

# Function to calculate call option VaR using parametric method
def VaR_parametric(S:float, K:float, T:float, T_opt:float, r:float, sigma:float, sigma_implied:float, alpha:float)->float:
    """
    Calculate the value at risk of a European call option using the parametric method.

    Parameters:
    S (float): Current price of the underlying asset
    K (float): Strike price of the option
    T (float): VaR time interval
    T_opt (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma (float): Actual volatility of the underlying asset
    sigma_implied (float): Black-Scholes volatility
    alpha (float): VaR confidence level

    Returns:
    float: Parametric value at risk of the call option
    """
    call_delta = call_option_delta(S, K, T_opt, r, sigma_implied)
    var_alpha = -call_delta * sigma * S * np.sqrt(T) * norm.ppf(1-alpha)
    return var_alpha

# Function to calculate call option VaR using Monte-Carlo method with delta approximation
def VaR_ES_MC_delta_approximation(S:float, K:float, T:float, T_opt:float, r:float, sigma:float, sigma_implied:float, alpha:float, nbSimul:int)->float:
    """
    Calculate the value at risk and expected shortfall of a European call option using the Monte-Carlo delta approximation method.

    Parameters:
    S (float): Current price of the underlying asset
    K (float): Strike price of the option
    T (float): VaR time interval
    T_opt (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma (float): Actual volatility of the underlying asset
    sigma_implied (float): Black-Scholes volatility
    alpha (float): VaR confidence level
    nbSimul (float): number of simulation paths

    Returns:
    float: Monte-Carlo value at risk (delta approximation) of the call option
    float: Monte-Carlo expected shortfall (delta approximation) of the call option
    List: Monte-Carlo distribution (delta approximation) of the call option
    """
    phi = np.random.normal(0, 1, nbSimul)
    call_delta = call_option_delta(S, K, T_opt, r, sigma_implied)
    call_simul = [call_delta * sigma * S * phi_i * np.sqrt(T) for phi_i in phi]
    call_simul.sort()
    var_alpha = -np.quantile(call_simul,1-alpha)
    index = int(len(call_simul) * (1-alpha))
    values_up_to_alpha = call_simul[:index]
    es_alpha = -np.mean(values_up_to_alpha)
    return var_alpha, es_alpha, call_simul

# Function to calculate call option VaR using Monte-Carlo method with delta-theta-gamma approximation
def VaR_ES_MC_delta_theta_gamma_approximation(S:float, K:float, T:float, T_opt:float, r:float, sigma:float, sigma_implied:float, alpha:float, nbSimul:int)->float:
    """
    Calculate the value at risk and expected shortfall of a European call option using the Monte-Carlo delta-theta-gamma approximation method.

    Parameters:
    S (float): Current price of the underlying asset
    K (float): Strike price of the option
    T (float): VaR time interval
    T_opt (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma (float): Actual volatility of the underlying asset
    sigma_implied (float): Black-Scholes volatility
    alpha (float): VaR confidence level
    nbSimul (float): number of simulation paths

    Returns:
    float: Monte-Carlo value at risk (delta-theta-gamma approximation) of the call option
    float: Monte-Carlo expected shortfall (delta-theta-gamma approximation) of the call option
    List: Monte-Carlo distribution (delta-theta-gamma approximation) of the call option
    """
    phi = np.random.normal(0, 1, nbSimul)
    chi = [phi_i**2 for phi_i in phi]
    call_delta = call_option_delta(S, K, T_opt, r, sigma_implied)
    call_theta = call_option_theta(S, K, T_opt, r, sigma_implied)
    call_gamma = call_option_gamma(S, K, T_opt, r, sigma_implied)
    call_simul_chi = [(call_theta + call_delta * mu * S + 0.5 * call_gamma * (sigma*S)**2 * chi_i) * T for chi_i in chi]
    call_simul_brown = [call_delta * sigma * S * phi_i * np.sqrt(T) for phi_i in phi]
    call_simul = call_simul_chi + call_simul_brown
    call_simul.sort()
    var_alpha = -np.quantile(call_simul,1-alpha)
    index = int(len(call_simul) * (1-alpha))
    values_up_to_alpha = call_simul[:index]
    es_alpha = -np.mean(values_up_to_alpha)
    return var_alpha, es_alpha, call_simul

# Function to calculate hedged call option VaR using parametric method
def VaR_hedged_parametric(S:float, K:float, T:float, T_opt:float, r:float, sigma:float, sigma_implied:float, alpha:float)->float:
    """
    Calculate the value at risk of an hedged European call option position using the parametric method.

    Parameters:
    S (float): Current price of the underlying asset
    K (float): Strike price of the option
    T (float): VaR time interval
    T_opt (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma (float): Actual volatility of the underlying asset
    sigma_implied (float): Black-Scholes volatility
    alpha (float): VaR confidence level

    Returns:
    float: Parametric value at risk of the hedged call option position
    """
    call_theta = call_option_theta(S, K, T_opt, r, sigma_implied)
    call_gamma = call_option_gamma(S, K, T_opt, r, sigma_implied)
    var_alpha = -0.5 * call_gamma * (sigma*S)**2 * T * chi2.ppf(1-alpha,1) - call_theta * T
    return var_alpha

# Function to calculate hedged call option VaR using Monte-Carlo method with delta-theta-gamma approximation
def VaR_ES_MC_hedged_delta_theta_gamma_approximation(S:float, K:float, T:float, T_opt:float, r:float, sigma:float, sigma_implied:float, alpha:float, nbSimul:int)->float:
    """
    Calculate the value at risk and expected shortfall of an hedged European call option position using the Monte-Carlo delta-theta-gamma approximation method.

    Parameters:
    S (float): Current price of the underlying asset
    K (float): Strike price of the option
    T (float): VaR time interval
    T_opt (float): Time to expiration in years
    r (float): Risk-free interest rate
    sigma (float): Actual volatility of the underlying asset
    sigma_implied (float): Black-Scholes volatility
    alpha (float): VaR confidence level
    nbSimul (float): number of simulation paths

    Returns:
    float: Monte-Carlo value at risk (delta-theta-gamma approximation) of the hedged call option position
    float: Monte-Carlo expected shortfall (delta-theta-gamma approximation) of the hedged call option position
    List: Monte-Carlo distribution (delta-theta-gamma approximation) of the hedged call option position
    """
    chi = np.random.chisquare(1, nbSimul)
    call_theta = call_option_theta(S, K, T_opt, r, sigma_implied)
    call_gamma = call_option_gamma(S, K, T_opt, r, sigma_implied)
    hedged_call_simul = [(call_theta * T + 0.5 * call_gamma * (sigma*S)**2 * chi_i * T) for chi_i in chi]
    hedged_call_simul.sort()
    var_alpha = -np.quantile(hedged_call_simul,1-alpha)
    index = int(len(hedged_call_simul) * (1-alpha))
    values_up_to_alpha = hedged_call_simul[:index]
    es_alpha = -np.mean(values_up_to_alpha)
    return var_alpha, es_alpha, hedged_call_simul

# Parameters
S0 = 100  # Current stock price
K = 100   # Strike price
DCF = 365 # Day-count fraction
T = 1/DCF # VaR period of time
T_opt = 0.1 # Option maturity (in years)
r = 0.05  # Risk-free rate
mu = 0.05 # Real-world stock drift
sigma = 0.2  # Actual Volatility
sigma_implied = 0.2 # Implied Volatility

# Confidence level
alpha = 0.99

# Number of simulation paths
nbSimul = 100000

# Container for results storing
compile_dict = {'Metric definition':[], 'Metric value':[]}

#1 - Parametric vaR VS MC VaR using delta approximation
var_parametric = VaR_parametric(S0, K, T, T_opt, r, sigma, sigma_implied, alpha)
var_mc_d, es_mc_d, call_return_d = VaR_ES_MC_delta_approximation(S0, K, T, T_opt, r, sigma, sigma_implied, alpha, nbSimul)
label1 = f'Parametric VaR ({alpha * 100}%): {var_parametric:.2f}'
label2 = f'Delta approximation Monte Carlo VaR ({alpha * 100}%): {var_mc_d:.2f}'
label3 = f'Delta approximation Monte Carlo ES ({alpha * 100}%): {es_mc_d:.2f}'
compile_dict['Metric definition'].append(f'Parametric VaR ({alpha * 100}%)')
compile_dict['Metric value'].append(var_parametric)
compile_dict['Metric definition'].append(f'Delta approximation Monte Carlo VaR ({alpha * 100}%)')
compile_dict['Metric value'].append(var_mc_d)
compile_dict['Metric definition'].append(f'Delta approximation Monte Carlo ES ({alpha * 100}%)')
compile_dict['Metric value'].append(es_mc_d)

# Visualize results
plt.figure(1)
plt.hist(call_return_d, bins=50, density=True, alpha=0.6, color='g', label='Call option Returns using delta approximation (Monte Carlo)')
plt.axvline(x=-var_parametric, color='k', linestyle='--', label=label1)
plt.axvline(x=-var_mc_d, color='r', linestyle='--', label=label2)
plt.axvline(x=-es_mc_d, color='b', linestyle='--', label=label3)
plt.legend()

#2 - Parametric vaR VS MC VaR using delta-theta-gamma approximation
var_parametric = VaR_parametric(S0, K, T, T_opt, r, sigma, sigma_implied, alpha)
var_mc_dtg, es_mc_dtg, call_return_dtg = VaR_ES_MC_delta_theta_gamma_approximation(S0, K, T, T_opt, r, sigma, sigma_implied, alpha, nbSimul)
label1 = f'Parametric VaR ({alpha * 100}%): {var_parametric:.2f}'
label2 = f'Delta-theta-gamma approximation Monte Carlo VaR ({alpha * 100}%): {var_mc_dtg:.2f}'
label3 = f'Delta-theta-gamma approximation Monte Carlo ES ({alpha * 100}%): {es_mc_dtg:.2f}'
compile_dict['Metric definition'].append(f'Delta-theta-gamma approximation Monte Carlo VaR ({alpha * 100}%)')
compile_dict['Metric value'].append(var_mc_dtg)
compile_dict['Metric definition'].append(f'Delta-theta-gamma approximation Monte Carlo ES ({alpha * 100}%)')
compile_dict['Metric value'].append(es_mc_dtg)

# Visualize results
plt.figure(2)
plt.hist(call_return_dtg, bins=50, density=True, alpha=0.6, color='g', label='Call option Returns using delta-theta-gamma approximation (Monte Carlo)')
plt.axvline(x=-var_parametric, color='k', linestyle='--', label=label1)
plt.axvline(x=-var_mc_dtg, color='r', linestyle='--', label=label2)
plt.axvline(x=-es_mc_dtg, color='b', linestyle='--', label=label3)
plt.legend()

#Distribution moments
print(f'The skew of delta approximation MC call return is : {skew(call_return_d):.2f}')
print(f'The kurtosis of delta approximation MC call return is : {kurtosis(call_return_d):.2f}')
print(f'The skew of delta-theta-gamma approximation MC call return is : {skew(call_return_dtg):.2f}')
print(f'The kurtosis of delta-theta-gamma approximation MC call return is : {kurtosis(call_return_dtg):.2f}')

#3 - Parametric vaR VS MC VaR using delta-theta-gamma approximation with hedged call position
var_parametric_h = VaR_hedged_parametric(S0, K, T, T_opt, r, sigma, sigma_implied, alpha)
var_mc_h, es_mc_h, call_return_h = VaR_ES_MC_hedged_delta_theta_gamma_approximation(S0, K, T, T_opt, r, sigma, sigma_implied, alpha, nbSimul)
label1 = f'Hedged call position parametric VaR ({alpha * 100}%): {var_parametric_h:.2f}'
label2 = f'Hedged call position Delta-theta-gamma approximation Monte Carlo VaR ({alpha * 100}%): {var_mc_h:.2f}'
label3 = f'Hedged call position Delta-theta-gamma approximation Monte Carlo ES ({alpha * 100}%): {es_mc_h:.2f}'
compile_dict['Metric definition'].append(f'Hedged call position parametric VaR ({alpha * 100}%)')
compile_dict['Metric value'].append(var_parametric_h)
compile_dict['Metric definition'].append(f'Hedged call position Delta-theta-gamma approximation Monte Carlo VaR ({alpha * 100}%)')
compile_dict['Metric value'].append(var_mc_h)
compile_dict['Metric definition'].append(f'Hedged call position Delta-theta-gamma approximation Monte Carlo ES ({alpha * 100}%)')
compile_dict['Metric value'].append(es_mc_h)

# Visualize results
plt.figure(3)
plt.hist(call_return_h, bins=50, density=True, alpha=0.6, color='g', label='Hedged call option Returns using delta-theta-gamma approximation (Monte Carlo)')
plt.axvline(x=-var_parametric_h, color='k', linestyle='--', label=label1)
plt.axvline(x=-var_mc_h, color='r', linestyle='--', label=label2)
plt.axvline(x=-es_mc_h, color='b', linestyle='--', label=label3)
plt.legend()

#4 - Results comparison
compile_results = pd.DataFrame(compile_dict)
pd.set_option('display.max_colwidth', None)
print(compile_results)