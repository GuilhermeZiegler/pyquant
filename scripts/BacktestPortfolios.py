import pandas as pd
import numpy as np
import yfinance  as yf
import plotly.express as px
import plotly.graph_objects as go
from tqdm import tqdm
import random
import joblib
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
from pypfopt import expected_returns
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro,  gaussian_kde
from scipy.integrate import quad
import pickle
from scipy import stats
import time
from tqdm import tqdm
import os
import glob

class Portfolio:
    """
    A class to manage investment portfolios.

    Attributes:
    - data (DataFrame): A DataFrame containing historical prices of assets.
    - invested_capital (float): The amount of capital invested in the portfolio.
    - window_size (int): The size of the window for processing data.
    - chunk_dfs (list): A list of DataFrames representing data chunks.
    - prices (np.array): An array containing the latest prices of assets.
    - names (list): A list of asset names.
    - dates (list): A list of last dates for each chunk.
    - sharpe_weights (np.array): Optimal weights for maximum Sharpe ratio.
    - min_risk_weights (np.array): Optimal weights for minimum risk.
    - max_sharpe_control (list): Portfolio optimized for maximum Sharpe ratio with rebalancing.
    - max_sharpe (list): Portfolio optimized for maximum Sharpe ratio without rebalancing.
    - min_risk_control (list): Portfolio optimized for minimum risk with rebalancing.
    - min_risk (list): Portfolio optimized for minimum risk without rebalancing.
    - even_weights (list): Portfolio with even weights.
    - smart_max_sharpe_control (list): Smartly rebalanced portfolio optimized for maximum Sharpe ratio.
    - smart_max_sharpe (list): Smartly rebalanced portfolio optimized for maximum Sharpe ratio without rebalancing.
    - smart_min_risk_control (list): Smartly rebalanced portfolio optimized for minimum risk.
    - smart_min_risk (list): Smartly rebalanced portfolio optimized for minimum risk without rebalancing.
    - smart_even_weights (list): Smartly rebalanced portfolio with even weights.
    - portfolio_ROI (list): Evaluation of portfolios' portfolio_portfolio_portfolio_ROI considering each round of optimization and the final ROI.
        
    """
    def __init__(self, data, invested_capital = 100000,  window_size=30):
        self.invested_capital = invested_capital
        self.data = data
        self.window_size = window_size
        
        self.chunk_dfs = self.process_data(window_size)
        self.prices = self.get_prices()
        self.names = self.get_asset_name()
        self.dates  = self.get_last_dates()
        self.sharpe_weights, self.max_sharpe_risk  = self.get_opt_values()
        self.min_risk_weights, self.min_risk = self.get_opt_values(objective = 'risk')
        self.max_sharpe_control= self.dummy_balancer(method='sharpe_control')
        self.max_sharpe = self.dummy_balancer(method = 'sharpe')
        self.min_risk_control = self.dummy_balancer(method = 'risk_control')
        self.min_risk = self.dummy_balancer(method = 'risk') 
        self.even_weights = self.dummy_balancer(method = 'even')
        self.smart_max_sharpe_control = self.smart_balancer(method='sharpe_control')
        self.smart_max_sharpe = self.smart_balancer(method = 'sharpe')
        self.smart_min_risk_control = self.smart_balancer(method = 'risk_control')
        self.smart_min_risk = self.smart_balancer(method = 'risk')
        self.smart_even_weights = self.smart_balancer(method = 'even')
        self.portfolio_ROI  = self.get_portfolio_ROI()
    
        self.plot_ROI = self.plot_ROI_over_time()
    
    
    def process_data(self, window_size):    
        """
        Processes data into chunks.portfolio_ROI

        Parameters:
        - window_size: Size of the window.

        Returns:
        - List of DataFrames representing data chunks.
        """
        chunk_dfs = []
        for i in range(0, len(self.data), window_size):
            if i + window_size <= len(self.data):
                chunk_df = self.data.iloc[i:i+window_size].copy()
            else:
                chunk_df = self.data.iloc[i:].copy()
        
            chunk_dfs.append(chunk_df)
            
        return chunk_dfs
    
    
    def simulate_frontier(self, chunk_df, risk_free_rate=0, trading_days=252, simulations=10000, objective='sharpe'):
        """
        Simulates optimal asset allocation based on the Sharpe ratio within a specified tolerance range,
        and selects the one with the highest Sharpe ratio.

        Parameters:
        - chunk_df: DataFrame containing asset prices.
        - risk_free_rate: Risk-free rate.
        - trading_days: Number of trading days.
        - simulations: Number of simulations.
        - sharpe_tolerance: Tolerance range for the Sharpe ratio.
        - objective: 'sharpe' or 'risk' to specify whether to maximize Sharpe ratio or minimize risk.

        Returns:
        - Numpy array representing optimal asset allocation with the highest Sharpe ratio or lowest positive standard deviation.
        """
        max_sharpe_ratio = -np.inf
        min_positive_sd = np.inf
        optimal_weights = None
        num_assets = len(chunk_df.columns)
        simple_returns = chunk_df.pct_change().dropna().mean() * trading_days
        cov_matrix = chunk_df.pct_change().dropna().cov() * trading_days

        for _ in range(simulations):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            returns = np.dot(weights, simple_returns)
            var = np.dot(weights.T, np.dot(cov_matrix, weights))
            sd = np.sqrt(var)

            if objective == 'sharpe':
                sharpe_ratio = (returns - risk_free_rate) / sd
                if sharpe_ratio > max_sharpe_ratio:
                    max_sharpe_ratio = sharpe_ratio
                    optimal_weights = weights
            elif objective == 'risk':
                if sd > 0 and sd < min_positive_sd:
                    min_positive_sd = sd
                    optimal_weights = weights
                    
            else:
                raise ValueError("Objective must be either 'sharpe' or 'risk'.")
        
        return optimal_weights, sd

    def get_opt_values(self, lower_bound=0.00, upper_bound=1, objective='sharpe'):
        
        """
        Calculate optimal portfolio weights for each chunk of data using mean-variance optimization.

        Parameters:
        lower_bound : float, optional
            The minimum bound for the weights. Default is 0.00.
        upper_bound : float, optional
            The maximum bound for the weights. Default is 1.
        objective : str, optional
            The objective for optimization. Can be 'sharpe' for maximizing the Sharpe ratio
            or 'min_volatility' for minimizing portfolio volatility. Default is 'sharpe'.
        Returns:
        np.array
            A numpy array of optimal weights for each chunk of data.
        """ 
        
        opt_weights = []
        pfl_risk = []
        
        for i, chunk_df in enumerate(self.chunk_dfs):
            try: 
                mu = expected_returns.mean_historical_return(chunk_df)
                S = risk_models.sample_cov(chunk_df)
                ef = EfficientFrontier(mu, S, weight_bounds=(lower_bound, upper_bound))
                if objective == 'sharpe':
                    ef.max_sharpe(risk_free_rate=0)
                else:
                    ef.min_volatility()
                cleaned_weights = ef.clean_weights()
                weights = [weight for _, weight in cleaned_weights.items()]
                

                risk = ef.port_volatility()
                opt_weights.append(weights)
                pfl_risk.append(risk)

            except Exception as e:
                
                #print(f"Error calculating optimal weights for chunk {i} \n Trying to simulate closest weights")
                if objective == 'sharpe':
                    simulated_max_sharpe_w, simulated_max_sharpe_risk  = self.simulate_frontier(chunk_df, risk_free_rate=0, objective=objective)
                    opt_weights.append(simulated_max_sharpe_w)
                    pfl_risk.append(simulated_max_sharpe_risk)
                else:
                    simulated_lowest_risk_w, simulated_lowest_risk = self.simulate_frontier(chunk_df, objective=objective)
                    opt_weights.append(simulated_lowest_risk_w)
                    pfl_risk.append(simulated_lowest_risk)
 
        optimal_weights = [weights for weights in opt_weights if weights is not None]
        optimal_risk = [risk for risk in pfl_risk if risk is not None]
        
        return  np.array(optimal_weights),  np.array(optimal_risk)

    def get_prices(self):
        prices = []
        for chunk_df in self.chunk_dfs:
            last_values = chunk_df.iloc[-1].values
            prices.append(last_values)
        return np.array(prices)

    def get_last_dates(self):
        """
        Gets the last index for each chunk.

        Returns:
        - List of last indices for each chunk.
        """
        last_indices = []
        for chunk_df in self.chunk_dfs[1:]:
            last_index = chunk_df.index[-1]
            last_index_date_only = str(last_index).split()[0]
            last_indices.append(last_index_date_only)
        return last_indices

    def get_asset_name(self, string ='_Close' ):
        """
        Gets the names of assets.

        Returns:
        - List of asset names.
        """
        names = []
        df = self.chunk_dfs[0]
        for column in df.columns:
            if  string in column:
                names.append(column.replace(string, ''))
            else:
                names.append(column)
        return names


    def smart_balancer(self, method='sharpe_control'):
        prices = self.prices
        if method == 'sharpe':
            weights = self.sharpe_weights
        elif method == 'risk':
            weights = self.min_risk_weights
        elif method == 'even':
            weights = np.full_like(prices, 1 / prices.shape[1])
        elif method == 'risk_control':
            weights = np.full_like(prices, self.min_risk_weights[0])
        else:
            weights = np.full_like(prices, self.sharpe_weights[0])
        
        funds = np.zeros_like(prices)
        total_capital = np.zeros_like(prices) 
        quantities = np.zeros_like(prices)
        smart_weights_balanced_portfolio = []
        for i in range(0, len(weights)):
            df = pd.DataFrame()
            if i == 0:
               quantities[0] = self.invested_capital * weights[i] / prices[i] 
               total_capital[0] = self.invested_capital * weights[i]
            else:
                for j in range(1, len(prices[i])):
                    if prices[i][j] > prices [i-1][j]:
                        funds[i][j] = (prices[i][j] - prices[i-1][j]) * quantities[i-1][j]
                   
                quantities[i] =  np.sum(funds[i]) * weights[i] / prices[i] + quantities[i-1]           
                weights[i] =  quantities[i] / np.sum(quantities[i]*prices[i])
                
                df[f'prices_t{i-1}'] = prices[i - 1]
                df[f'prices_t{i}'] = prices[i]
                df[f'weights_t{i-1}'] = weights[i-1]
                df[f'weights_t{i}'] = weights[i]
                df[f'funds_t{i}'] = funds[i]
                df[f'quantities_t{i-1}'] = quantities[i-1]
                df[f'quantities_bought_t{i}'] = quantities[i] - quantities[i-1]
                df[f'quantities_t{i}'] = quantities[i]
                df[f'invested_capital_t{i-1}'] = quantities[i-1] * prices[i-1]
                df[f'invested_capital_t{i}'] = quantities[i] * prices[i]
                df[f'cumulative_capital_t{i-1}'] = df[f'invested_capital_t{i-1}'].cumsum()
                df[f'cumulative_capital_t{i}'] = df[f'invested_capital_t{i}'].cumsum()
                df.index = self.names
                smart_weights_balanced_portfolio.append(df)

        return smart_weights_balanced_portfolio


    def dummy_balancer(self, method='sharpe_control'):
        """
        Adjusts the weights of a portfolio based on the specified method.
        
        Parameters:
        - self: Reference to the instance of the object containing the portfolio data, including prices (`prices`), Sharpe weights (`sharpe_weights`), minimum risk weights (`min_risk_weights`), and invested capital (`invested_capital`).
        - method (str, optional): The balancing method to use. Possible values are:
        - 'sharpe': Uses the Sharpe weights.
        - 'risk': Uses the minimum risk weights.
        - 'even': Distributes the weights equally among the assets.
        - 'risk_control': Uses a risk control strategy based on the first minimum risk weight.
        - Default is 'sharpe_control': Uses the first Sharpe weight.
        
        Returns:
        - simple_weights_balanced_portfolio (list of pandas.DataFrame): A list of DataFrames, each containing the prices, weights, quantities, and capital invested for each asset in each period.
        """
            
        prices = self.prices
        if method == 'sharpe':
            weights = self.sharpe_weights
        elif method == 'risk':
            weights = self.min_risk_weights
        elif method == 'even':
            weights = np.full_like(prices, 1 / prices.shape[1])
        elif method == 'risk_control':
            weights = np.full_like(prices, self.min_risk_weights[0])
        else:
            weights = np.full_like(prices, self.sharpe_weights[0])
            
        total_capital = np.zeros_like(prices) 
        quantities = np.zeros_like(prices)
        simple_weights_balanced_portfolio = []

        for i in range(1, len(weights)):
            df = pd.DataFrame()

            quantities[i-1] = self.invested_capital * weights[i-1] / prices[i-1] 
            total_capital[i] = quantities[i-1] * prices[i]   
            quantities[i] = np.sum(total_capital[i]) * weights[i] / prices[i]
 
            df[f'prices_t{i-1}'] = prices[i - 1]
            df[f'prices_t{i}'] = prices[i]
            df[f'weights_t{i-1}'] = weights[i-1]
            df[f'weights_t{i}'] = weights[i]
            df[f'funds_t{i}'] = prices[i] * quantities[i]
            df[f'quantities_t{i-1}'] = quantities[i-1]
            df[f'quantities_bought_t{i}'] = quantities[i] - quantities[i-1]
            df[f'quantities_t{i}'] = quantities[i]
            df[f'invested_capital_t{i-1}'] = quantities[i-1] * prices[i-1]
            df[f'invested_capital_t{i}'] = total_capital[i]
            df[f'cumulative_capital_t{i-1}'] = df[f'invested_capital_t{i-1}'].cumsum()
            df[f'cumulative_capital_t{i}'] = df[f'invested_capital_t{i}'].cumsum()
            df.index = self.names
            simple_weights_balanced_portfolio.append(df)
            
        return simple_weights_balanced_portfolio

    def get_portfolio_ROI(self):
        optimal_ROI = []
        portfolios = [
            ("sharpe_control", self.max_sharpe_control),
            ("sharpe", self.max_sharpe),
            ("min_risk_control", self.min_risk_control),
            ("min_risk", self.min_risk),
            ("even_weights", self.even_weights),
            ("smart_sharpe_control", self.smart_max_sharpe_control),
            ("smart_max_sharpe", self.smart_max_sharpe),
            ("smart_min_risk_control", self.smart_min_risk_control),
            ("smart_min_risk", self.smart_min_risk),
            ("smart_even_weights", self.smart_even_weights)
        ]
        num_periods = len(portfolios[0][1])
        
        for i in range(num_periods):
            opt_ROI = {} 
            for name, portfolio in portfolios:
                this_opt = portfolio[i][f'cumulative_capital_t{i+1}'].iloc[-1]
                previous_opt = portfolio[i][f'cumulative_capital_t{i}'].iloc[-1] if i > 0 else self.invested_capital
                previous_ROI = this_opt / previous_opt
                latest_ROI = this_opt / self.invested_capital
                opt_ROI[name] = {
                    "previous_ROI_df": round(previous_ROI, 4),
                    "lastest_ROI_df": round(latest_ROI, 4)
                }
            optimal_ROI.append(opt_ROI)


        return optimal_ROI 

    def plot_ROI_over_time(self, return_type='portfolio_ROI', method='smart'):
        # Seleciona o DataFrame com base no return_type
        if return_type == 'portfolio_ROI':
            df = self.portfolio_ROI
        elif return_type == 'optimization_ROI':
            df = self.optimization_ROI
        else:
            raise ValueError("Invalid return_type. Use 'portfolio_ROI' or 'optimization_ROI'.")

        # Filtra as colunas com base no método
        columns_to_plot = [col for col in df.columns if method in col]

        fig, ax = plt.subplots(figsize=(10, 6))
        for column in columns_to_plot:
            ax.plot(df.index, df[column], label=column)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('ROI')
        ax.set_title('ROI Over Time for Different Strategies')
        ax.legend()
        ax.grid(True)
        ax.set_xticks(df.index)
        ax.set_xticklabels(self.dates, rotation=45)
        
        return fig
    
class BacktestPortfolios:
    def __init__(self, assets, num_combinations, num_assets, window_size=30, seed=52):
        self.assets = assets
        self.window_size = window_size
        self.num_assets = num_assets
        self.num_combinations = num_combinations
        self.seed = seed
        self.combinations = self.generate_combinations()
        self.portfolios = self.analyze_portfolios()
        self.smart_strategies, self.dummy_strategies = self.normalize_strategies()

    def generate_combinations(self):
        random.seed(self.seed)
        combinations = []
        for _ in range(self.num_combinations):
            shuffled_assets = self.assets[:]
            random.shuffle(shuffled_assets)
            combination = random.sample(shuffled_assets, self.num_assets)
            concatenated_df = pd.concat(combination, axis=1, join='inner')
            if not concatenated_df.empty:
                combinations.append(concatenated_df)
        return combinations

    def analyze_portfolios(self):
        portfolios = {}
        for i, df in tqdm(enumerate(self.combinations), total=self.num_combinations):
            portfolio_name = f'n_assets_{self.num_assets}_window_size_{self.window_size}_{i}'
            try:
                portfolios[portfolio_name] = Portfolio(data=df, invested_capital=100000, window_size=self.window_size)
            except Exception as e:
                print(f"Error analyzing portfolio {portfolio_name}: {e}")
        return portfolios

    def normalize_strategies(self, return_type='portfolio_ROI', roi_type='lastest_ROI_df', period='last'):
        smart_normalized_strategies = []
        dummy_normalized_strategies = []
        for _, plf in self.portfolios.items():
            roi_data = getattr(plf, return_type)

            if not roi_data: 
                continue

            if period == 'last':
                roi_data = [roi_data[-1]]

            for period_data in roi_data:
                smart_denominator = period_data.get('smart_even_weights', {}).get(roi_type, 1)
                dummy_denominator = period_data.get('even_weights', {}).get(roi_type, 1)

                smart_normalized = {key: value.get(roi_type, 0) / smart_denominator
                                    for key, value in period_data.items()
                                    if 'smart' in key and 'smart_even_weights' not in key
                                    }
                dummy_normalized = {key: value.get(roi_type, 0) / dummy_denominator
                                    for key, value in period_data.items()
                                    if 'smart' not in key and 'even_weights' not in key
                                    }

                smart_normalized_strategies.append(smart_normalized)
                dummy_normalized_strategies.append(dummy_normalized)

        return smart_normalized_strategies, dummy_normalized_strategies
    
class BacktestAnalysis:
    def __init__(self, smart_strategies=None, dummy_strategies=None, load: bool = False, window_size: list = None):
        self.smart_strategies = smart_strategies if smart_strategies else []
        self.dummy_strategies = dummy_strategies if dummy_strategies else []
        self.load = load
        self.window_size = window_size if window_size else [30, 60, 90, 120, 150, 180]

        if self.load:
            self.load_optimizations()
    
    def load_optimizations(self):
        base_dir = '..\\pkl'
        all_smart_strategies = {ws: [] for ws in self.window_size}
        all_dummy_strategies = {ws: [] for ws in self.window_size}
        
        for filename in os.listdir(base_dir):
            if filename.endswith('.pkl'):
                for ws in self.window_size:
                    if f'window_{str(ws)}' in filename:
                        file_path = os.path.join(base_dir, filename)
                        try:
                            with open(file_path, 'rb') as file:
                                backtest_portfolio = pickle.load(file)
                                all_smart_strategies[ws].extend(backtest_portfolio.smart_strategies)
                                all_dummy_strategies[ws].extend(backtest_portfolio.dummy_strategies)
                        except (pickle.UnpicklingError, EOFError, KeyError) as e:
                            print(f"Error processing file {filename}: {e}")

        self.smart_strategies = all_smart_strategies
        self.dummy_strategies = all_dummy_strategies
    
    def test_distribution(self, strategy_type='smart', keys='all'):
        """
        Tests the distribution of data from a specified type of strategy (smart or dummy) for normality.

        Parameters:
        strategy_type (str): The type of strategy to test. Either 'smart' or 'dummy'.
        keys (str or list, optional): 
        The keys to test. If 'all', tests all keys found in the strategies. 
        If a list of keys, only tests the specified keys. Default is 'all'.

        Returns:
        list of dict: A list of dictionaries containing the results of the normality test for each key.
            Each dictionary has the following keys:
            - 'key': The key being tested.
            - 'normal': A boolean indicating if the distribution is normal (p-value > 0.05).
            - 'p_value': The p-value from the Shapiro-Wilk test for normality.
        """
        distribution_results = []
        
        if strategy_type == 'smart':
            strategies = []
            for ws in self.window_size:
                strategies.extend(self.smart_strategies.get(ws, []))
        elif strategy_type == 'dummy':
            strategies = []
            for ws in self.window_size:
                strategies.extend(self.dummy_strategies.get(ws, []))
        else:
            raise ValueError("Invalid strategy_type. Must be 'smart' or 'dummy'.")


        if keys == 'all':
            all_keys = set().union(*(strategy.keys() for strategy in strategies))
        else:
            all_keys = keys
                
        all_data = {key: [] for key in all_keys}
        
        for strategy in strategies:
            for key in all_data.keys():
                value = strategy.get(key, None)
                if value is not None:
                    if isinstance(value, (list, np.ndarray)):
                        all_data[key].extend(value)
                    else:
                        all_data[key].append(value)
                        
        for key in all_data.keys():
            if len(all_data[key]) >= 3:
                _, p_value = shapiro(all_data[key])
                normal = p_value > 0.05

                distribution_results.append({
                    'key': key,
                    'normal': normal,
                    'p_value': p_value
                })
                

                if normal:
                    self.plot_histogram(all_data[key], f'{strategy_type}_{key}')
                else:
                    self.plot_kde(all_data[key], f'{strategy_type}_{key}')
            else:
                print(f"Insufficient data for key '{key}' to perform normality test.")
        
        return distribution_results

    
    def plot_histogram(self, data, strategy, prob=1, auto_save=True):
        """
        Plots a histogram of initial returns for a specific strategy and adjusts the bar colors based on data quartiles.

        Args:
            data (list or array-like): List or array containing the initial return data to be plotted.
            strategy (str): The name of the strategy to be included in the chart title and legend.
            prob (float, optional): Probability value to highlight with a vertical line on the chart. Default is 1.
            auto_save (bool, optional): If True, automatically saves the generated chart as a PNG file. Default is True.

        Returns:
            None

        Note:
            - The generated chart includes:
            - A histogram of the data with bars colored according to quartiles.
            - A black line representing the normal distribution fitted to the data.
            - A red vertical line indicating the value specified by `prob`.
            - The probability of returns greater than `prob` displayed in the legend.

        Example:
            >>> plot_histogram(data=[0.1, 0.5, 0.3, 0.7], strategy='StrategyA', prob=0.2)
        """
        fig, ax = plt.subplots(figsize=(12, 8))  # Create a figure and axis object

        q1, q2, q3 = np.percentile(data, [25, 50, 75])
        n, bins, patches = ax.hist(data, bins=30, alpha=0.7, label=strategy, edgecolor='black')

        for patch, bin_left in zip(patches, bins[:-1]):
            if bin_left < q1:
                patch.set_facecolor('red')
            elif bin_left < q2:
                transition_ratio = (bin_left - q1) / (q2 - q1)
                patch.set_facecolor(plt.cm.Blues(transition_ratio))
            elif bin_left < q3:
                transition_ratio = (bin_left - q2) / (q3 - q2)
                patch.set_facecolor(plt.cm.Blues(transition_ratio))
            else:
                patch.set_facecolor('blue')

            mu, std = norm.fit(data)
            bin_centers = 0.5 * (bins[:-1] + bins[1:])
            p = norm.pdf(bin_centers, mu, std)
            p_scaled = p * np.max(n) / np.max(p)
            ax.plot(bin_centers, p_scaled, 'k', linewidth=2, label='Normal Distribution', alpha=0.5)
            ax.axvline(x=1, color='r', linestyle='--', label='Value 1')
            probability = np.sum(np.array(data) > prob) / len(data)
            ax.legend([f'Probability of returns > 1: {probability:.2f}'])
            ax.set_title(f'Histogram of Initial Returns for: {strategy}  for {len(data)} combinations')
            ax.set_xlabel('Last_ROI')
            ax.set_ylabel('Frequency')
            ax.set_xlim(min(data) - 0.1 * abs(min(data)), max(data) + 0.1 * abs(max(data)))
            ax.set_ylim(0, max(n) + 0.25 * max(n))
            ax.grid(True)

            if auto_save:
                fig_name = f'backtest_for_{len(data)}_combinations_{self.window_size}_for_strategy_{strategy}.png'
                fig.savefig(fig_name, bbox_inches='tight')  # Save the figure with tight bounding box

            plt.show()

    def plot_kde(self, data, strategy, prob=1, auto_save=True):
        """
        Plots a Kernel Density Estimate (KDE) plot of initial returns for a specific strategy and highlights different data quartiles.

        Args:
            data (list or array-like): List or array containing the initial return data to be plotted.
            strategy (str): The name of the strategy to be included in the chart title and legend.
            prob (float, optional): Probability value to highlight with a vertical line on the chart. Default is 1.
            auto_save (bool, optional): If True, automatically saves the generated chart as a PNG file. Default is True.

        Returns:
            None

        Note:
            - The generated chart includes:
            - A KDE plot with different shades of blue for different data quartiles.
            - A red vertical line indicating the value specified by `prob` and displaying the probability of data greater than `prob`.
            - The KDE plot is displayed with appropriate x and y limits, and the chart is saved if `auto_save` is True.

        Example:
            >>> plot_kde(data=[0.1, 0.5, 0.3, 0.7], strategy='StrategyA', prob=0.2)
        """
        fig, ax = plt.subplots(figsize=(12, 8))  # Create a figure and axis object

        kde = gaussian_kde(data)
        x_values = np.linspace(min(data) - 0.1 * abs(min(data)), max(data) + 0.1 * abs(max(data)), 1000)
        y_values = kde(x_values)
        q1, q2, q3 = np.percentile(data, [25, 50, 75])
        ax.fill_between(x_values, y_values, where=(x_values < q1), color=plt.cm.Blues(0.2), alpha=0.5)
        ax.fill_between(x_values, y_values, where=(x_values >= q1) & (x_values < q2), color=plt.cm.Blues(0.4), alpha=0.5)
        ax.fill_between(x_values, y_values, where=(x_values >= q2) & (x_values < q3), color=plt.cm.Blues(0.6), alpha=0.5)
        ax.fill_between(x_values, y_values, where=(x_values >= q3), color=plt.cm.Blues(0.8), alpha=0.5)
        prob_greater_than = np.sum(np.array(data) > prob) / len(data)
        ax.axvline(x=1, color='r', linestyle='--', label=f'Probability > {prob}: {prob_greater_than:.2f} for {len(data)} combinations')
        ax.set_title(f'KDE Plot - {strategy}')
        ax.set_xlabel('ROI_initial_period')
        ax.set_ylabel('Density')
        ax.legend()
        ax.set_xlim(min(data) - 0.1 * abs(min(data)), max(data) + 0.1 * abs(max(data)))
        ax.set_ylim(0, np.max(y_values) + 0.25 * np.max(y_values))
        ax.grid(False)

        if auto_save:
            fig_name = f'backtest_for_{len(data)}_combinations_window_size_{self.window_size[0]}_for_strategy_{strategy}.png'
            fig.savefig(fig_name, bbox_inches='tight')  # Save the figure with tight bounding box

        plt.show()

        
    def get_results(self, strategy='smart', prob=1, auto_save=True):
        """
        Calculates and returns the probability of each metric for the given strategy.

        Args:
            strategy (str, optional): Strategy type, either 'smart' or 'dummy'. Default is 'smart'.
            prob (float, optional): Threshold probability value. Default is 1.
            auto_save (bool, optional): If True, automatically saves the resulting DataFrame as a CSV file. Default is True.

        Returns:
            pd.DataFrame: A DataFrame containing the probabilities of each metric for the specified strategy.
        """
        if strategy == 'smart':
            all_strategies = self.smart_strategies
        else:
            all_strategies = self.dummy_strategies

        probabilities_dict = {}

        for key, strategies in all_strategies.items():
            metrics = {}
            metric_count = {}

            for strategy in strategies:
                for metric, value in strategy.items():
                    if metric not in metrics:
                        metrics[metric] = 0
                        metric_count[metric] = 0

                    if value > prob:
                        metrics[metric] += 1
                    metric_count[metric] += 1
            
            probabilities = {metric: (metrics[metric] / metric_count[metric] if metric_count[metric] > 0 else 0)
                            for metric in metric_count}
            
            probabilities_dict[key] = probabilities

        df_probabilities = pd.DataFrame(probabilities_dict).T
        df_probabilities = df_probabilities.round(3)

        if auto_save:
            csv_file_name = f'probabilities_for_{strategy}_strategies.csv'
            df_probabilities.to_csv(csv_file_name, index=True)

        return df_probabilities
    
