import yfinance as yf
import pandas as pd
import numpy as np
import os
import pickle
import joblib
import plotly.graph_objects as go
import re
pd.set_option('display.max_columns', None)


class CandleFit:
    def __init__(self, ticker: str, period: str = '5y'):
        """
        Initializes the CandleFit object with a specified ticker and period.

        Parameters:
        ticker (str): The ticker symbol of the stock.
        period (str): The period for which to download historical data.
        """
        self.ticker = ticker
        self.period = period
        self.data = self.get_ticker()
        self.features = self.get_price_features()
        self.threshold_dict: dict = None
      

    def get_ticker(self):
        """
        Downloads historical data for the specified ticker and period.

        Returns:
        pd.DataFrame: A DataFrame containing the historical data with formatted dates and column names.
        """
        try:
            ticker_obj = yf.Ticker(self.ticker)
            hist = ticker_obj.history(period=self.period)
            hist.index = pd.to_datetime(hist.index).strftime('%Y-%m-%d')
            hist.columns = [col.lower() for col in hist.columns]
            return hist
        except Exception as e:
            print(f"Error downloading aux: {e}")
            return pd.DataFrame()
    
    @staticmethod
    def load_dict(key: str) -> dict:
        """
        Loads a dictionary from a pickle file based on the provided key.

        Parameters:
        key (str): The key to search for in the dictionary.

        Returns:
        dict: The dictionary associated with the provided key.

        Raises:
        FileNotFoundError: If the pickle file is not found.
        KeyError: If the key is not found in the dictionary.
        ValueError: If there is an error loading the pickle file.
        """
        filepath = os.path.join('..', 'pkl', 'threshold_dicts.pkl')
        try:
            with open(filepath, 'rb') as file:
                data = pickle.load(file)
            
            for item in data:
                if key in item:
                    return item[key]
            raise KeyError(f"Key '{key}' not found in the threshold dictionary.")
        except FileNotFoundError:
            raise FileNotFoundError(f"The file '{filepath}' was not found.")
        except pickle.PickleError:
            raise ValueError("Error occurred while loading the pickle file.")
        
    def get_price_features(self):
        """
        Calculates various price and volume features from historical data.

        Returns:
        pd.DataFrame: A DataFrame containing the calculated features.
        """
        aux = self.data.copy()
        df = pd.DataFrame(index=aux.index)
        df.index = pd.to_datetime(df.index).strftime('%Y-%m-%d')
        df['return_rate'] = aux['close'].pct_change()
        df['volume_change'] = aux['volume'].diff()
        df['volume_var'] = aux['volume'].pct_change() + 1
        df['price_range'] = aux['high'] - aux['low']
        df['price_var'] = df['price_range'] / aux['low']
        df['price_change'] = aux['close'] - aux['open']
        df['close_vol'] = aux['close'].expanding().std()
        df['low_vol'] = aux['low'].expanding().std()
        df['high_vol'] = aux['high'].expanding().std()
        df['open_vol'] = aux['open'].expanding().std()
        df['upper_wick'] = aux['high'] - aux[['open', 'close']].max(axis=1)
        df['lower_wick'] = aux[['open', 'close']].min(axis=1) - aux['low']
        df['wick_change'] = df['upper_wick'] - df['lower_wick']
        df['wick_var'] = df['wick_change'] / df['lower_wick']
        df['std_wick'] = df['wick_change'].abs().expanding().std()
        df = df.apply(pd.to_numeric, errors='coerce')
        df = pd.concat([aux, df], axis=1)
        
        return df

    def get_candle_features(self, 
                            doji_threshold: float =  0.002, 
                            bullish_threshold : float =  0.005,
                            bearish_threshold: float =  0.005,
                            volatility_window: int = 7):
                            
 
        """
        Calculates various candlestick features based on price data and a threshold dictionary.

        Parameters:
        threshold_dict (dict): A dictionary containing thresholds for calculating features. If None, it loads the default dictionary.

        Returns:
        pd.DataFrame: A DataFrame containing the calculated candlestick features.
        """
        
        aux = self.features.copy()
        aux = aux.loc[:, ~aux.columns.duplicated()]
        df = pd.DataFrame(index=aux.index)
        df[f'std_volatility_window'] = aux['price_change'].rolling(window=volatility_window).std().abs()
        df['bearish_threshold'] = pd.to_numeric(bearish_threshold * df[f'std_volatility_window'], errors='coerce').fillna(0)
        df['bullish_threshold'] = pd.to_numeric(bullish_threshold * df[f'std_volatility_window'], errors='coerce').fillna(0)
        df['is_bearish'] = (aux['close'] <= (aux['open'] - df['bearish_threshold'])).astype(int)
        df['is_bullish'] = (aux['close'] >= (aux['open'] + df['bullish_threshold'])).astype(int)
        df['is_doji'] = (abs(aux['close'] - aux['open']) <= doji_threshold).astype(int)
        df['is_bearish_open_gap'] = (aux['open'] < aux['close'].shift(1)).astype(int)
        df['is_bullish_open_gap'] = (aux['open'] > aux['close'].shift(1)).astype(int)
    
        df = pd.concat([aux, df], axis=1)
        self.features = df
        return  self.features

    def fit_morning_star(self,
                         doji_threshold: float = 0.002,
                         bullish_to_bearish_ratio : float = 0.33,
                         bearish_threshold: float = 0.2):
                  
        """
        Identifies the morning star candlestick pattern in the historical data.

        Parameters:
        doji_threshold (float): The threshold for the price change to identify a doji candle. Default is 0.002.
        bullish_threshold (float): The ratio of the price change on the third day to the price change on the first day to identify a bullish candle. Default is 0.33.
        bearish_threshold (float): The threshold to identify a bearish candle on the first day. Default is 0.2.
        volatility_window (int): The window size (in days) for calculating the volatility of the historical data. Default is 7.

        Returns:
        pd.DataFrame: A DataFrame with a column indicating the presence of the morning star pattern.
        """
        df = self.get_candle_features()
        df = df.loc[:, ~df.columns.duplicated(keep='last')]

        # Condition 1: Two days ago was a bearish candle and the close of that day is lower than the open of that day adjusted by the threshold
        df['is_bearish_morning_star'] = ((df['is_bearish'].shift(2) == 1) & 
                                        (df['close'].shift(2) + bearish_threshold * df['price_change'].shift(2) 
                                        <= df['open'].shift(2))).astype(int)

        # Condition 2: The previous day was a doji candle and had a bearish open gap
        df['is_bearish_open_gap_morning_star'] = (df['is_bearish_open_gap'].shift(1) == 1).astype(int)
        df['is_doji_morning_star'] = (df['price_change'].shift(1).abs() <= doji_threshold ).astype(int)

        # Condition 3: Today is a bullish candle and the price change from two days ago to today is significant
        df['is_bullish_morning_star'] = ((df['is_bullish'] == 1) & \
                                        (df['close'] - df['close'].shift(2) >= bullish_to_bearish_ratio)
                                        * df['price_change'].shift(2).abs()).astype(int)

        # Combine all conditions to determine the morning star pattern
        df['is_morning_star'] = df[['is_bearish_morning_star', 'is_bearish_open_gap_morning_star', 'is_doji_morning_star', 
                                    'is_bullish_morning_star']].all(axis=1).astype(int)

        self.features = df
        return self.features
    
    
    def fit_hammer(self,
                   lower_wick_to_price_change_ratio: float = 3,
                   upper_wick_to_price_change_ratio: float = 0.02,
                   volatility_window: int = None):
        """
        Identifies the hammer candlestick pattern in the historical data.

        Parameters:
        lower_wick_to_price_change_ratio (float): The minimum ratio of the lower wick length to the price change to identify a hammer handle. Default is 4.
        upper_wick_to_price_change_ratio (float): The maximum ratio of the upper wick length to the price change to identify a hammer head. Default is 0.05.

        Returns:
        pd.DataFrame: A DataFrame with columns indicating the presence of the hammer pattern and its components.
        """
        df = self.get_candle_features(volatility_window).copy()
        df = df.loc[:, ~df.columns.duplicated(keep='last')]

        # Condition 1: Two days ago was a bearish candle and the close of that day is lower than the open of that day adjusted by the threshold
        df['is_hammer_head'] =  (df['upper_wick'] <= upper_wick_to_price_change_ratio * df['std_volatility_window']).astype(int)
        df['is_hammer_handle'] =  (df['lower_wick'] / df['price_change'].abs() >= df['std_volatility_window'] * lower_wick_to_price_change_ratio).astype(int)
        df['is_hammer'] = df[['is_hammer_head', 'is_hammer_handle']].all(axis=1).astype(int)

        self.features = df
        return self.features

    
    def get_movings(self, short:int = None, long:int = None, strategy: str = 'test'):

        """
        Calculates buy and sell signals based on moving averages for different strategies and parameters.

        Parameters:
        - threshold_dict (dict, optional): Dictionary containing moving average settings for different strategies. If any of parameters is None,
        it will be loaded with `self.load_dict(key='rolling_cross_dict')` and available strategies will be measured.  
        - short (int, optional): Time window for the short moving average. Not used directly in the function.
        - long_ (int, optional): Time window for the long moving average. Not used directly in the function.
        - signal_window (int, optional): Time window for signal calculation. Not used directly in the function.

        Returns:
        - pd.DataFrame: DataFrame with additional columns for moving averages and buy/sell signals.
        """
        df = self.get_candle_features()  
        df = df.loc[:, ~df.columns.duplicated(keep='last')]
        
        if (short is None) != (long is None):
            raise ValueError("Please set both short and long to valid int or set both to None.")
        elif all(x is not None for x in (short, long)):
            threshold_dict = {f'{strategy}_{short}_{long}': {'short': short, 'long': long}}
        else:
            display('Loading standard strategies')
            threshold_dict = self.load_dict(key='rolling_cross_dict')

        tolerance = 0.015 * df['close_vol']  
        for key, value in threshold_dict.items():
            strategy = key.split('_')[0] if '_' in key else key
            short = value['short']
            long = value['long']
            
            df.loc[:, f'{strategy}_short_{short}'] = df['close'].rolling(window=short, min_periods=1).mean()
            df.loc[:, f'{strategy}_long_{long}'] = df['close'].rolling(window=long, min_periods=1).mean()
            
            
            buy = (df[f'{strategy}_short_{short}'] > df[f'{strategy}_long_{long}'] + tolerance) & \
                (df[f'{strategy}_short_{short}'].shift(1) < df[f'{strategy}_long_{long}'].shift(1) + tolerance)
            
            sell = (df[f'{strategy}_short_{short}'] < df[f'{strategy}_long_{long}'] - tolerance) & \
                (df[f'{strategy}_short_{short}'].shift(1) > df[f'{strategy}_long_{long}'].shift(1) - tolerance)
            
            df.loc[:, f'{strategy}_{short}_{long}'] = np.where(buy, 1, np.where(sell, -1, 0))

                        
        self.features = df
        self.threshold_dict = threshold_dict  
        return self.features          

    def candlestick_chart(self, 
                      key: str = 'is_morning_star', 
                      plot_type: str = 'pattern',
                      height: int = 1200, 
                      offset: float = 12):
        """
        Generates a candlestick chart with optional pattern or price action markers.

        Parameters:
        - key (str): Key for identifying the pattern or price action indicators. Defaults to 'is_morning_star'.
        - plot_type (str): Type of plot to generate. Options are 'pattern' or 'price_action'. Defaults to 'pattern'.
        - height (int): Height of the plot in pixels. Defaults to 900.
        - offset (float): Vertical offset for pattern markers. Defaults to 12.

        Returns:
        - go.Figure: A Plotly Figure object with the candlestick chart.
        """
        
        aux = self.features.copy()
        aux = aux.loc[:, ~aux.columns.duplicated(keep='last')]
        
        if key not in aux.columns:
            print(f"Key '{key}' not found in aux columns.")
            return None

        trace = go.Candlestick(
            x=aux.index,
            open=aux["open"],
            high=aux["high"],
            low=aux["low"],
            close=aux["close"],
            name=self.ticker,
            yaxis="y"
        )
        
        volume_colors = ['green' if aux['volume'][i] > aux['volume'][i-1] else 'red' for i in range(1, len(aux))]
        volume_colors.insert(0, 'green')
        volume_trace = go.Bar(
                x=aux.index,
                y=aux['volume'],
                marker_color=volume_colors,
                name='Volume',
                yaxis="y2"
            )
            
        
        if plot_type == 'pattern':     
            markers = aux[aux[key] == 1].copy()
            markers['close'] = markers['low'] - offset
            markers = markers.dropna(subset=['close'])
            marker_trace = go.Scatter(
                x=markers.index,
                y=markers['close'],
                mode='markers',
                marker=dict(
                    color='blue',
                    size=8,
                    symbol='triangle-up'
                ),
                name=key,
                yaxis="y")
            
            data = [trace, marker_trace, volume_trace]
            
        elif plot_type == 'moving_cross':
            strategy, short, long = key.split('_')
            short_col = f'{strategy}_short_{short}'
            long_col = f'{strategy}_long_{long}'
            signal_col = f'{strategy}_{short}_{long}'
            short_trace = go.Scatter(
                x=aux.index,
                y=aux[short_col],
                mode='lines',
                name=f'{strategy.capitalize()} Short {short}',
                line=dict(color='purple')  
            )
            long_trace = go.Scatter(
                x=aux.index,
                y=aux[long_col],
                mode='lines',
                name=f'{strategy.capitalize()} Long {long}',
                line=dict(color='orange')  
            )

            markers = aux[aux[signal_col] != 0].copy()  
            markers['close'] = markers.apply(
                lambda row: row['low'] - offset if row[signal_col] == 1 else row['high'] + offset,
                axis=1
            )
            markers['color'] = markers[signal_col].apply(lambda x: 'green' if x == 1 else 'red')
            markers['symbol'] = markers[signal_col].apply(lambda x: 'arrow-up' if x == 1 else 'arrow-down')
            
            marker_trace = go.Scatter(
                x=markers.index,
                y=markers['close'],
                mode='markers',
                marker=dict(
                    color=markers['color'],
                    size=8,
                    symbol=markers['symbol']
                ),
                name='Signal',
                yaxis="y"
            )
            
           
            data = [trace, short_trace, long_trace, marker_trace, volume_trace]
        
        layout = go.Layout(
            title=f"{self.ticker} Candlestick Chart markers: {key.capitalize()}",
            xaxis=dict(title="Date"),
            yaxis=dict(title="Price", domain=[0.3, 1]),
            yaxis2=dict(title="Volume", domain=[0, 0.2]),
            height=height,
            barmode='relative'
        )
        
        fig = go.Figure(data=data, layout=layout)
        return fig
        
    def get_trades(self,  
                   strategy: str = None,
                   reward_risk_ratio: list = None, 
                   price_col: str = 'close',
                   trade_period=7,
                   target_return=0.01):
        """
        Calculates potential trades based on a given strategy and parameters.

        Parameters:
        df (DataFrame): The DataFrame containing price data and strategy signals.
        strategy (str): The column name of the strategy signals (1 for buy, -1 for sell).
        reward_risk_ratio (list): List of reward to risk ratios to consider.
        price_col (str): The column name to use as the price for entering trades.
        trade_period (int): The number of periods to look ahead for the trade.
        target_return (float): The target return for each trade.

        Returns:
        DataFrame: The DataFrame with trade information, results, and positions.
        """
        
        df = self.features.copy()
        
        cols = ['open', 'high', 'low', 'close']
        shifted_cols = []
        stop_loss_cols = []
        
        for col in cols:
            for i in range(2, trade_period + 1):
                df[f'{col}_{i}'] = df[col].shift(-i)
                shifted_cols.append(f'{col}_{i}')
        
        has_signals = (df[strategy] == 1) | (df[strategy] == -1)
        if not has_signals.any():
            return "The strategy did not generate buy or sell signals."
        
        df = df[cols + [strategy] + shifted_cols][has_signals].copy()
        df['side'] = df[strategy].apply(lambda x: 'long' if x == 1 else 'short')
        df['tp'] = df.apply(lambda row: (1 + target_return) * row[price_col] if row[strategy] == 1 else (1 - target_return) * row[price_col], axis=1)
        if reward_risk_ratio is None:      
            reward_risk_ratio = np.arange(0.5, 2.50, 0.5)
        
        for rr in reward_risk_ratio:
            df[f'sl_{rr}'] = np.where(df['side'] == 'long', df[price_col] * (1 - target_return * rr),df[price_col] * (1 + target_return * rr))
            stop_loss_cols.append(f'sl_{rr}')

    
        for loss_col in stop_loss_cols:
            df[f'out_day_{loss_col}'] = np.nan
            df[f'result_{loss_col}'] = np.nan
            df[f'result_{loss_col}_value'] = np.nan
            
            for index, row in df.iterrows():
                stop_loss_val = row[loss_col]
                target_price = row['tp']
                entry_price = row[price_col] 
                
                for i in range(2, trade_period + 1):
                    next_open = row[f'open_{i}']
                    next_high = row[f'high_{i}']
                    next_low = row[f'low_{i}']
                    next_close = row[f'close_{i}']
                    
                    if row['side'] == 'long':
                        price_vars = [next_open, next_low, next_high, next_close]
                        for price_var in price_vars:
                            if price_var > target_price:
                                df.at[index, f'out_day_{loss_col}'] = int(i)
                                df.at[index, f'result_{loss_col}'] = 'profit'
                                df.at[index, f'result_{loss_col}_value'] = price_var - entry_price
                                break
                            elif price_var < stop_loss_val:
                                df.at[index, f'out_day_{loss_col}'] = int(i)
                                df.at[index, f'result_{loss_col}'] = 'loss'
                                df.at[index, f'result_{loss_col}_value'] = price_var - entry_price
                                break
                    else:  # 'short'
                        price_vars = [next_open, next_high, next_low, next_close]
                        for price_var in price_vars:
                            if price_var < target_price:
                                df.at[index, f'out_day_{loss_col}'] = int(i)
                                df.at[index, f'result_{loss_col}'] = 'profit'
                                df.at[index, f'result_{loss_col}_value'] = entry_price - price_var
                                break
                            elif price_var > stop_loss_val:
                                df.at[index, f'out_day_{loss_col}'] = int(i)
                                df.at[index, f'result_{loss_col}'] = 'loss'
                                df.at[index, f'result_{loss_col}_value'] = entry_price - price_var
                                break
                    
                    if not np.isnan(df.at[index, f'out_day_{loss_col}']):
                        break
        
        return df
    
    @staticmethod
    def summarize_results(df):
        """
        Summarize profit and loss results for different stop loss levels.

        This function calculates the total profit and loss for each stop loss level present
        in the DataFrame. The stop loss levels are identified dynamically based on column names
        that follow the pattern 'result_sl_X.Y'.

        Args:
            df (pd.DataFrame): The DataFrame containing the trade results. The DataFrame must
                            have columns with names in the format 'result_sl_X.Y' and
                            'result_sl_X.Y_value', where X.Y represents the stop loss level.

        Returns:
            pd.DataFrame: A DataFrame with columns 'Level', 'Profit', and 'Loss', summarizing
                        the total profit and loss for each stop loss level.
        """
        summary_df = pd.DataFrame(columns=['Risk Reward Ratio', 'Profit', 'Loss', 'Net', 'Profits Count', 'Losses Count', 'Success Rate'])
        
        levels = set()
        for col in df.columns:
            match = re.match(r'result_sl_(\d+\.\d+)', col)
            if match:
                levels.add(match.group(1))
        
        levels = sorted(levels, key=lambda x: float(x))        
        summary_list = []
        for level in levels:
            results_col = f'result_sl_{level}'
            values_col = f'result_sl_{level}_value'
            
            if results_col in df.columns and values_col in df.columns:
                df[values_col] = pd.to_numeric(df[values_col], errors='coerce')
                
                profits_series = df[df[values_col] > 0][values_col]
                losses_series = df[df[values_col] < 0][values_col]
                
                profits = profits_series.sum()
                profits_count = profits_series.count()
                
                losses = losses_series.sum()
                losses_count = losses_series.count()
                
                total_trades = profits_count + losses_count
                success_rate = profits_count / total_trades if total_trades > 0 else 0
                
                summary_list.append({
                    'Risk Reward Ratio': level,
                    'Profit': profits,
                    'Loss': losses,
                    'Net': profits + losses,
                    'Profits Count': profits_count,
                    'Losses Count': losses_count,
                    'Success Rate': success_rate
                })
        
        summary_df = pd.concat([summary_df, pd.DataFrame(summary_list)], ignore_index=True)
        
        return summary_df
