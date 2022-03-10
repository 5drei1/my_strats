import freqtrade.vendor.qtpylib.indicators as qtpylib
from typing import Dict, List, Optional
import numpy as np
import talib.abstract as ta
from freqtrade.strategy import IStrategy, informative
from freqtrade.strategy import (merge_informative_pair,
                                DecimalParameter, IntParameter, BooleanParameter, timeframe_to_minutes, stoploss_from_open, RealParameter)
from pandas import DataFrame, Series
from functools import reduce
from freqtrade.persistence import Trade
from datetime import datetime, timedelta, timezone
from freqtrade.exchange import timeframe_to_prev_date
from technical.indicators import RMI, zema, ichimoku
import math
import pandas_ta as pta
import technical.indicators as ftt
import logging
logger = logging.getLogger(__name__)
# --------------------------------



########################################################################################################
#                                                                                                      #
#                                                                                                      # 
#            ██████   ██████  ███    ██ ████████     ████████ ██████  ██    ██ ███████ ████████        #
#            ██   ██ ██    ██ ████   ██    ██           ██    ██   ██ ██    ██ ██         ██           # 
#            ██   ██ ██    ██ ██ ██  ██    ██           ██    ██████  ██    ██ ███████    ██           # 
#            ██   ██ ██    ██ ██  ██ ██    ██           ██    ██   ██ ██    ██      ██    ██           # 
#            ██████   ██████  ██   ████    ██           ██    ██   ██  ██████  ███████    ██           # 
#                                                                                                      #                                                                                    
#                        ███    ███ ███████     ██      █████  ███    ███      █████                   #         
#                        ████  ████ ██          ██     ██   ██ ████  ████     ██   ██                  #         
#                        ██ ████ ██ █████       ██     ███████ ██ ████ ██     ███████                  #         
#                        ██  ██  ██ ██          ██     ██   ██ ██  ██  ██     ██   ██                  #         
#                        ██      ██ ███████     ██     ██   ██ ██      ██     ██   ██                  #         
#                                                                                                      #
#                                ███    ██  ██████   ██████  ██████                                    #                 
#                                ████   ██ ██    ██ ██    ██ ██   ██                                   #                 
#                                ██ ██  ██ ██    ██ ██    ██ ██████                                    #                 
#                                ██  ██ ██ ██    ██ ██    ██ ██   ██                                   #                 
#                                ██   ████  ██████   ██████  ██████                                    #     
#                                                                                                      # 
# Based on ....                                                                                        #
########################################################################################################
## Apollo 11, MultiMA_TSL, modded by stash86, based on SMAOffsetProtectOptV1 (modded by Perkmeister)  ##
## Obelisk Ichimoku                                                                                   ##
## Strategy for Freqtrade https://github.com/freqtrade/freqtrade                                      ##                                                    
##                                                                                                    ##
##                                                                         .... and many others       ##
########################################################################################################

def to_minutes(**timdelta_kwargs):
    return int(timedelta(**timdelta_kwargs).total_seconds() / 60)


class Apollo11HO(IStrategy):
    timeframe = "15m"

    # Stoploss¢
    stoploss = -0.20
    startup_candle_count: int = 480
    trailing_stop = False
    use_custom_stoploss = False
    use_sell_signal = True
    use_custom_sell = True


    # Indicator values:
    buy_params = {

        "buy_rsx": 55,
        # Signal 1
        "s1_ema_xs" : 3,
        "s1_ema_sm" : 5,
        "s1_ema_md" : 10,
        "s1_ema_xl" : 50,
        "s1_ema_xxl" : 200,

        # Signal 2
        "s2_ema_input" : 50,
        "s2_ema_offset_input" : -1,

        "s2_bb_sma_length" : 49,
        "s2_bb_std_dev_length" : 64,
        "s2_bb_lower_offset" : 3,

        "s2_fib_sma_len" : 50,
        "s2_fib_atr_len" : 14,

        "s2_fib_lower_value" : 4.236,
    }
    sell_params = {
        "pHSL": -0.185,
        "pPF_1": 0.013,
        "pPF_2": 0.061,
        "pSL_1": 0.007,
        "pSL_2": 0.014,

        ##
        "sell_btc_safe": -389,
        "sell_cmf": -0.046,
        "sell_ema": 0.988,
        "sell_ema_close_delta": 0.022,

        "buy_ada_cti": -0.715,
        ##
        "sell_deadfish_profit": -0.063,
        "sell_deadfish_bb_factor": 0.954,
        "sell_deadfish_bb_width": 0.043,
        "sell_deadfish_volume_factor": 2.37,
        ##
        "sell_cmf_div_1_cmf": 0.442,
        "sell_cmf_div_1_profit": 0.02,
    }
    rsi_buy_optimize = False
    buy_rsx = IntParameter(15, 75, default=buy_params['buy_rsx'], space='buy', optimize = rsi_buy_optimize)
    # signal controls
    buy_signal_1 = True

    optimize_sig1 = True
    # Signal 1
    s1_ema_xs = IntParameter(1, 4, default=buy_params['s1_ema_xs'], space='buy', optimize=optimize_sig1)
    s1_ema_sm = IntParameter(5, 9, default=buy_params['s1_ema_sm'], space='buy', optimize=optimize_sig1)
    s1_ema_md = IntParameter(10, 25, default=buy_params['s1_ema_md'], space='buy', optimize=optimize_sig1)
    s1_ema_xl = IntParameter(35, 75, default=buy_params['s1_ema_xl'], space='buy', optimize=optimize_sig1)
    s1_ema_xxl = IntParameter(150, 250, default=buy_params['s1_ema_xxl'], space='buy', optimize=optimize_sig1)

    # Signal 2
    optimize_sig2 = True
    buy_signal_2 = True
    s2_ema_input = IntParameter(30, 80, default=buy_params['s1_ema_xxl'], space='buy', optimize=optimize_sig2)
    
    s2_ema_offset_input = DecimalParameter(-2.0, 2.0, default=buy_params['s2_ema_offset_input'], decimals=1, space='buy', optimize=optimize_sig2)

    s2_bb_sma_length = IntParameter(30, 75, default=buy_params['s2_bb_sma_length'], space='buy', optimize=optimize_sig2)
    s2_bb_std_dev_length = IntParameter(50, 100, default=buy_params['s2_bb_std_dev_length'], space='buy', optimize=optimize_sig2)
    s2_bb_lower_offset = IntParameter(1, 5, default=buy_params['s2_bb_lower_offset'], space='buy', optimize=optimize_sig2)

    s2_fib_sma_len = IntParameter(30, 75, default=buy_params['s2_fib_sma_len'], space='buy', optimize=optimize_sig2)
    s2_fib_atr_len = IntParameter(10, 30, default=buy_params['s2_fib_atr_len'], space='buy', optimize=optimize_sig2)

    s2_fib_lower_value = DecimalParameter(3.000, 6.999, default=buy_params['s2_fib_lower_value'], decimals=3, space='buy', optimize=optimize_sig2)

    ############### Custom Sell
    is_optimize_sell_stoploss = False
    sell_cmf = DecimalParameter(-0.4, 0.0, default=0.0, optimize = is_optimize_sell_stoploss)
    sell_ema_close_delta = DecimalParameter(0.022, 0.027, default= 0.024, optimize = is_optimize_sell_stoploss)
    sell_ema = DecimalParameter(0.97, 0.99, default=0.987 , optimize = is_optimize_sell_stoploss)
    is_optimize_deadfish = False
    sell_deadfish_bb_width = DecimalParameter(0.010, 0.025, default=0.05 , optimize = is_optimize_deadfish)
    sell_deadfish_profit = DecimalParameter(-0.10, -0.05, default=-0.05 , optimize = is_optimize_deadfish)
    sell_deadfish_bb_factor = DecimalParameter(0.90, 1.20, default=1.0 , optimize = is_optimize_deadfish)
    sell_deadfish_volume_factor = DecimalParameter(1.5, 3, default=1.5 , optimize = is_optimize_deadfish)

    is_optimize_cti_r = False
    sell_cti_r_cti = DecimalParameter(0.55, 1, default=0.5 , optimize = is_optimize_cti_r)
    sell_cti_r_r = DecimalParameter(-15, 0, default=-20 , optimize = is_optimize_cti_r)

    is_optimize_cmf_div = False
    sell_cmf_div_1_profit = DecimalParameter(0.005, 0.02, default=0.005 , optimize = is_optimize_cmf_div)
    sell_cmf_div_1_cmf = DecimalParameter(0.0, 0.5, default=0.0 , optimize = is_optimize_cmf_div)
    sell_cmf_div_2_profit = DecimalParameter(0.005, 0.02, default=0.005 , optimize = is_optimize_cmf_div)
    sell_cmf_div_2_cmf = DecimalParameter(0.0, 0.5, default=0.0 , optimize = is_optimize_cmf_div)
      

    # Trailing stop:
    trailing_stop = True
    trailing_stop_positive = 0.007
    trailing_stop_positive_offset = 0.014
    trailing_only_offset_is_reached = True


    # ROI table:
    minimal_roi = {
        "120": 0.066
    }
    inf_1h = '1h'
    def informative_pairs(self):

        pairs = self.dp.current_whitelist()
        informative_pairs = [(pair, '1h') for pair in pairs]


        return informative_pairs
    @property
    def protections(self):
        return [
            {
                # Don't enter a trade right after selling a trade.
                "method": "CooldownPeriod",
                "stop_duration": to_minutes(minutes=0),
            },
            {
                # Stop trading if max-drawdown is reached.
                "method": "MaxDrawdown",
                "lookback_period": to_minutes(hours=12),
                "trade_limit": 20,  # Considering all pairs that have a minimum of 20 trades
                "stop_duration": to_minutes(hours=1),
                "max_allowed_drawdown": 0.2,  # If max-drawdown is > 20% this will activate
            },
            {
                # Stop trading if a certain amount of stoploss occurred within a certain time window.
                "method": "StoplossGuard",
                "lookback_period": to_minutes(hours=6),
                "trade_limit": 4,  # Considering all pairs that have a minimum of 4 trades
                "stop_duration": to_minutes(minutes=30),
                "only_per_pair": False,  # Looks at all pairs
            },
            {
                # Lock pairs with low profits
                "method": "LowProfitPairs",
                "lookback_period": to_minutes(hours=1, minutes=30),
                "trade_limit": 2,  # Considering all pairs that have a minimum of 2 trades
                "stop_duration": to_minutes(hours=15),
                "required_profit": 0.02,  # If profit < 2% this will activate for a pair
            },
            {
                # Lock pairs with low profits
                "method": "LowProfitPairs",
                "lookback_period": to_minutes(hours=6),
                "trade_limit": 4,  # Considering all pairs that have a minimum of 4 trades
                "stop_duration": to_minutes(minutes=30),
                "required_profit": 0.01,  # If profit < 1% this will activate for a pair
            },
        ]
    def informative_1h_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        assert self.dp, "DataProvider is required for multiple timeframes."
        # Get the informative pair
        informative_1h = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.inf_1h)


######################################################################################################################## ICHIMOKU ####################################################################

        # # Heikin Ashi Candles
        heikinashi = qtpylib.heikinashi(informative_1h)
        informative_1h['ha_open'] = heikinashi['open']
        informative_1h['ha_close'] = heikinashi['close']
        informative_1h['ha_high'] = heikinashi['high']
        informative_1h['ha_low'] = heikinashi['low']

        displacement = 30
        ichimoku = ftt.ichimoku(informative_1h, 
            conversion_line_period=20, 
            base_line_periods=60,
            laggin_span=120, 
            displacement=displacement
            )


        informative_1h['chikou_span'] = ichimoku['chikou_span']

        # cross indicators
        informative_1h['tenkan_sen'] = ichimoku['tenkan_sen']
        informative_1h['kijun_sen'] = ichimoku['kijun_sen']

        # cloud, green a > b, red a < b
        informative_1h['senkou_a'] = ichimoku['senkou_span_a']
        informative_1h['senkou_b'] = ichimoku['senkou_span_b']
        informative_1h['leading_senkou_span_a'] = ichimoku['leading_senkou_span_a']
        informative_1h['leading_senkou_span_b'] = ichimoku['leading_senkou_span_b']
        informative_1h['cloud_green'] = ichimoku['cloud_green'] * 1
        informative_1h['cloud_red'] = ichimoku['cloud_red'] * -1

        informative_1h.loc[:, 'cloud_top'] = informative_1h.loc[:, ['senkou_a', 'senkou_b']].max(axis=1)
        informative_1h.loc[:, 'cloud_bottom'] = informative_1h.loc[:, ['senkou_a', 'senkou_b']].min(axis=1)

        # DANGER ZONE START

        # NOTE: Not actually the future, present data that is normally shifted forward for display as the cloud
        informative_1h['future_green'] = (informative_1h['leading_senkou_span_a'] > informative_1h['leading_senkou_span_b']).astype('int') * 2
        informative_1h['future_red'] = (informative_1h['leading_senkou_span_a'] < informative_1h['leading_senkou_span_b']).astype('int') * 2

        # The chikou_span is shifted into the past, so we need to be careful not to read the
        # current value.  But if we shift it forward again by displacement it should be safe to use.
        # We're effectively "looking back" at where it normally appears on the chart.
        informative_1h['chikou_high'] = (
                (informative_1h['chikou_span'] > informative_1h['cloud_top'])
            ).shift(displacement).fillna(0).astype('int')

        informative_1h['chikou_low'] = (
                (informative_1h['chikou_span'] < informative_1h['cloud_bottom'])
            ).shift(displacement).fillna(0).astype('int')

        # DANGER ZONE END

        informative_1h['atr'] = ta.ATR(informative_1h, timeperiod=14)
        ssl_down, ssl_up = ssl_atr(informative_1h, 10)
        informative_1h['ssl_down'] = ssl_down
        informative_1h['ssl_up'] = ssl_up
        informative_1h['ssl_ok'] = (
                (ssl_up > ssl_down) 
            ).astype('int') * 3
        informative_1h['ssl_bear'] = (
                (ssl_up < ssl_down) 
            ).astype('int') * 3

        informative_1h['ichimoku_ok'] = (
                (informative_1h['tenkan_sen'] > informative_1h['kijun_sen'])
                & (informative_1h['close'] > informative_1h['cloud_top'])
                & (informative_1h['future_green'] > 0) 
                & (informative_1h['chikou_high'] > 0) 
            ).astype('int') * 4

        informative_1h['ichimoku_bear'] = (
                (informative_1h['tenkan_sen'] < informative_1h['kijun_sen'])
                & (informative_1h['close'] < informative_1h['cloud_bottom'])
                & (informative_1h['future_red'] > 0) 
                & (informative_1h['chikou_low'] > 0) 
            ).astype('int') * 4

        informative_1h['ichimoku_valid'] = (
                (informative_1h['leading_senkou_span_b'] == informative_1h['leading_senkou_span_b']) # not NaN
            ).astype('int') * 1

        informative_1h['bear_trend_pulse'] = (
                (informative_1h['ichimoku_bear'] > 0) 
                & (informative_1h['ssl_bear'] > 0)
            ).astype('int') * 2

        informative_1h['bear_trend_over'] = (
                (informative_1h['ssl_bear'] == 0)
                | (informative_1h['close'] > informative_1h['cloud_bottom'])
            ).astype('int') * 1

        informative_1h.loc[ (informative_1h['bear_trend_pulse'] > 0), 'bear_trending'] = 3
        informative_1h.loc[ (informative_1h['bear_trend_over'] > 0) , 'bear_trending'] = 0
        informative_1h['bear_trending'].fillna(method='ffill', inplace=True)

        return informative_1h
######################################################################################################################## ICHIMOKU END ################################################################

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        # Adding EMA's into the dataframe
        dataframe["s1_ema_xs"] = ta.EMA(dataframe, timeperiod=self.s1_ema_xs.value)
        dataframe["s1_ema_sm"] = ta.EMA(dataframe, timeperiod=self.s1_ema_sm.value)
        dataframe["s1_ema_md"] = ta.EMA(dataframe, timeperiod=self.s1_ema_md.value)
        dataframe["s1_ema_xl"] = ta.EMA(dataframe, timeperiod=self.s1_ema_xl.value)
        dataframe["s1_ema_xxl"] = ta.EMA(dataframe, timeperiod=self.s1_ema_xxl.value)

        s2_ema_value = ta.EMA(dataframe, timeperiod=self.s2_ema_input.value)
        s2_ema_xxl_value = ta.EMA(dataframe, timeperiod=200)

        dataframe["s2_ema"] = ((s2_ema_value - s2_ema_value * self.s2_ema_offset_input.value)/2)
        dataframe["s2_ema_xxl_off"] = s2_ema_xxl_value - s2_ema_xxl_value * self.s2_fib_lower_value.value
        dataframe["s2_ema_xxl"] = ta.EMA(dataframe, timeperiod=200)

        s2_bb_sma_value = ta.SMA(dataframe, timeperiod=self.s2_bb_sma_length.value)
        s2_bb_std_dev_value = ta.STDDEV(dataframe, self.s2_bb_std_dev_length.value)
        dataframe["s2_bb_std_dev_value"] = s2_bb_std_dev_value
        dataframe["s2_bb_lower_band"] = s2_bb_sma_value - (s2_bb_std_dev_value * self.s2_bb_lower_offset.value)

        s2_fib_atr_value = ta.ATR(dataframe, timeframe=self.s2_fib_atr_len.value)
        s2_fib_sma_value = ta.SMA(dataframe, timeperiod=self.s2_fib_sma_len.value)

        dataframe["s2_fib_lower_band"] = s2_fib_sma_value - s2_fib_atr_value * self.s2_fib_lower_value.value
        dataframe['rsx'] = pta.rsx(dataframe['close'], timeperiod=14)
        #CUSTOM SELL INDICATORS
        # Heiken Ashis
        heikinashi = qtpylib.heikinashi(dataframe)
        heikinashi["volume"] = dataframe["volume"]
        # Profit Maximizer - PMAX
        dataframe['pm'], dataframe['pmx'] = pmax(heikinashi, MAtype=1, length=9, multiplier=27, period=10, src=3)
        dataframe['source'] = (dataframe['high'] + dataframe['low'] + dataframe['open'] + dataframe['close'])/4
        dataframe['pmax_thresh'] = ta.EMA(dataframe['source'], timeperiod=9)
       # CTI
        dataframe['cti'] = pta.cti(dataframe["close"], length=20)
        dataframe['ema_200'] = ta.EMA(dataframe, timeperiod=200)
        dataframe['r_14'] = williams_r(dataframe, period=14)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=14)
        dataframe['volume_mean_12'] = dataframe['volume'].rolling(12).mean().shift(1)
        dataframe['volume_mean_24'] = dataframe['volume'].rolling(24).mean().shift(1)
        # CMF
        dataframe['cmf'] = chaikin_money_flow(dataframe, 20)
        # Bollinger bands (hyperopt hard to implement)
        bollinger2 = qtpylib.bollinger_bands(qtpylib.typical_price(dataframe), window=20, stds=2)
        dataframe['bb_lowerband2'] = bollinger2['lower']
        dataframe['bb_middleband2'] = bollinger2['mid']
        dataframe['bb_upperband2'] = bollinger2['upper']
        dataframe['bb_width'] = ((dataframe['bb_upperband2'] - dataframe['bb_lowerband2']) / dataframe['bb_middleband2'])
        #CUSTOM SELL INDICATORS

        informative_1h = self.informative_1h_indicators(dataframe, metadata)
        dataframe = merge_informative_pair(dataframe, informative_1h, self.timeframe, self.inf_1h, ffill=True)
        return dataframe

    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        max_slip = 0.983

        if(len(dataframe) < 1):
            return False

        dataframe = dataframe.iloc[-1].squeeze()
        if ((rate > dataframe['close'])) :

            slippage = ( (rate / dataframe['close']) - 1 ) * 100

            if slippage < max_slip:
                return True
            else:
                return False

    def populate_buy_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # basic buy methods to keep the strategy simple
        buy_check = ((dataframe['rsx'] < self.buy_rsx.value))
        if self.buy_signal_1:
            conditions = [
                (dataframe['ichimoku_valid_1h']>0),
                (dataframe['bear_trending_1h'] <= 0),
                (dataframe['rsx'] < self.buy_rsx.value),
                dataframe["close"] < dataframe["s1_ema_xxl"],
                qtpylib.crossed_above(dataframe["s1_ema_sm"], dataframe["s1_ema_md"]),
                dataframe["s1_ema_xs"] < dataframe["s1_ema_xl"],
                dataframe["close"] < dataframe["s1_ema_xl"],
                dataframe["volume"] > 0,
            ]
            dataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_1")

        if self.buy_signal_2:
            conditions = [
                (dataframe['ichimoku_valid_1h']>0),
                (dataframe['bear_trending_1h'] <= 0),
                (dataframe['rsx'] < self.buy_rsx.value),
                qtpylib.crossed_above(dataframe["s2_fib_lower_band"], dataframe["s2_bb_lower_band"]),
                dataframe["close"] < dataframe["s2_ema"],
                dataframe["volume"] > 0,
            ]
            dataframe.loc[reduce(lambda x, y: x & y, conditions), ["buy", "buy_tag"]] = (1, "buy_signal_2")

        if not self.buy_signal_1 and not self.buy_signal_2:
            dataframe.loc[(), "buy"] = 0

        return dataframe

    def populate_sell_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # This is essentailly ignored as we're using strict ROI / Stoploss / TTP sale scenarios
        dataframe.loc[(), "sell"] = 0
        return dataframe

    def custom_sell(self, pair: str, trade: 'Trade', current_time: 'datetime', current_rate: float,
                    current_profit: float, **kwargs):

        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        last_candle = dataframe.iloc[-1]
        previous_candle_1 = dataframe.iloc[-2]

        max_profit = ((trade.max_rate - trade.open_rate) / trade.open_rate)
        max_loss = ((trade.open_rate - trade.min_rate) / trade.min_rate)

        buy_tag = 'empty'
        if hasattr(trade, 'buy_tag') and trade.buy_tag is not None:
            buy_tag = trade.buy_tag
        buy_tags = buy_tag.split()

        pump_tags = ['adaptive ']

        # sell cti_r
        if 0.012 > current_profit >= 0.0 :
            if (last_candle['cti'] > self.sell_cti_r_cti.value) and (last_candle['r_14'] > self.sell_cti_r_r.value):
                return f"sell_profit_cti_r_1( {buy_tag})"

        # sell over 200
        if last_candle['close'] > last_candle['ema_200']:
            if (current_profit > 0.01) and (last_candle['rsi'] > 75):
                return f"sell_profit_o_1 ( {buy_tag})"

        # sell quick
        if (0.06 > current_profit > 0.02) and (last_candle['rsi'] > 80.0):
            return f"signal_profit_q_1( {buy_tag})"

        if (0.06 > current_profit > 0.02) and (last_candle['cti'] > 0.95):
            return f"signal_profit_q_2( {buy_tag})"

        # sell recover
        if (max_loss > 0.06) and (0.05 > current_profit > 0.01) and (last_candle['rsi'] < 40):
            return f"signal_profit_r_1( {buy_tag})"

        # stoploss - deadfish
        if (    (current_profit < self.sell_deadfish_profit.value)
                and (last_candle['close'] < last_candle['ema_200'])
                and (last_candle['bb_width'] < self.sell_deadfish_bb_width.value)
                and (last_candle['close'] > last_candle['bb_middleband2'] * self.sell_deadfish_bb_factor.value)
                and (last_candle['volume_mean_12'] < last_candle['volume_mean_24'] * self.sell_deadfish_volume_factor.value)
                and (last_candle['cmf'] < 0.0)
        ):
            return f"sell_stoploss_deadfish( {buy_tag})"




##################### CI #############################################################################################################

def EWO(dataframe, sma1_length=5, sma2_length=35):
    df = dataframe.copy()
    sma1 = ta.SMA(df, timeperiod=sma1_length)
    sma2 = ta.SMA(df, timeperiod=sma2_length)
    smadif = (sma1 - sma2) / df['close'] * 100
    return smadif

def bollinger_bands(stock_price, window_size, num_of_std):
    rolling_mean = stock_price.rolling(window=window_size).mean()
    rolling_std = stock_price.rolling(window=window_size).std()
    lower_band = rolling_mean - (rolling_std * num_of_std)
    return np.nan_to_num(rolling_mean), np.nan_to_num(lower_band)


def ha_typical_price(bars):
    res = (bars['ha_high'] + bars['ha_low'] + bars['ha_close']) / 3.
    return Series(index=bars.index, data=res)


def pmax(df, period, multiplier, length, MAtype, src):

    period = int(period)
    multiplier = int(multiplier)
    length = int(length)
    MAtype = int(MAtype)
    src = int(src)

    mavalue = f'MA_{MAtype}_{length}'
    atr = f'ATR_{period}'
    pm = f'pm_{period}_{multiplier}_{length}_{MAtype}'
    pmx = f'pmX_{period}_{multiplier}_{length}_{MAtype}'

    # MAtype==1 --> EMA
    # MAtype==2 --> DEMA
    # MAtype==3 --> T3
    # MAtype==4 --> SMA
    # MAtype==5 --> VIDYA
    # MAtype==6 --> TEMA
    # MAtype==7 --> WMA
    # MAtype==8 --> VWMA
    # MAtype==9 --> zema
    if src == 1:
        masrc = df["close"]
    elif src == 2:
        masrc = (df["high"] + df["low"]) / 2
    elif src == 3:
        masrc = (df["high"] + df["low"] + df["close"] + df["open"]) / 4

    if MAtype == 1:
        mavalue = ta.EMA(masrc, timeperiod=length)
    elif MAtype == 2:
        mavalue = ta.DEMA(masrc, timeperiod=length)
    elif MAtype == 3:
        mavalue = ta.T3(masrc, timeperiod=length)
    elif MAtype == 4:
        mavalue = ta.SMA(masrc, timeperiod=length)
    elif MAtype == 5:
        mavalue = VIDYA(df, length=length)
    elif MAtype == 6:
        mavalue = ta.TEMA(masrc, timeperiod=length)
    elif MAtype == 7:
        mavalue = ta.WMA(df, timeperiod=length)
    elif MAtype == 8:
        mavalue = vwma(df, length)
    elif MAtype == 9:
        mavalue = zema(df, period=length)

    df[atr] = ta.ATR(df, timeperiod=period)
    df['basic_ub'] = mavalue + ((multiplier/10) * df[atr])
    df['basic_lb'] = mavalue - ((multiplier/10) * df[atr])


    basic_ub = df['basic_ub'].values
    final_ub = np.full(len(df), 0.00)
    basic_lb = df['basic_lb'].values
    final_lb = np.full(len(df), 0.00)

    for i in range(period, len(df)):
        final_ub[i] = basic_ub[i] if (
            basic_ub[i] < final_ub[i - 1]
            or mavalue[i - 1] > final_ub[i - 1]) else final_ub[i - 1]
        final_lb[i] = basic_lb[i] if (
            basic_lb[i] > final_lb[i - 1]
            or mavalue[i - 1] < final_lb[i - 1]) else final_lb[i - 1]

    df['final_ub'] = final_ub
    df['final_lb'] = final_lb

    pm_arr = np.full(len(df), 0.00)
    for i in range(period, len(df)):
        pm_arr[i] = (
            final_ub[i] if (pm_arr[i - 1] == final_ub[i - 1]
                                    and mavalue[i] <= final_ub[i])
        else final_lb[i] if (
            pm_arr[i - 1] == final_ub[i - 1]
            and mavalue[i] > final_ub[i]) else final_lb[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] >= final_lb[i]) else final_ub[i]
        if (pm_arr[i - 1] == final_lb[i - 1]
            and mavalue[i] < final_lb[i]) else 0.00)

    pm = Series(pm_arr)

    # Mark the trend direction up/down
    pmx = np.where((pm_arr > 0.00), np.where((mavalue < pm_arr), 'down',  'up'), np.NaN)

    return pm, pmx


def HA(dataframe, smoothing=None):
    df = dataframe.copy()

    df['HA_Close']=(df['open'] + df['high'] + df['low'] + df['close'])/4

    df.reset_index(inplace=True)

    ha_open = [ (df['open'][0] + df['close'][0]) / 2 ]
    [ ha_open.append((ha_open[i] + df['HA_Close'].values[i]) / 2) for i in range(0, len(df)-1) ]
    df['HA_Open'] = ha_open

    df.set_index('index', inplace=True)

    df['HA_High']=df[['HA_Open','HA_Close','high']].max(axis=1)
    df['HA_Low']=df[['HA_Open','HA_Close','low']].min(axis=1)

    if smoothing is not None:
        sml = abs(int(smoothing))
        if sml > 0:
            df['Smooth_HA_O']=ta.EMA(df['HA_Open'], sml)
            df['Smooth_HA_C']=ta.EMA(df['HA_Close'], sml)
            df['Smooth_HA_H']=ta.EMA(df['HA_High'], sml)
            df['Smooth_HA_L']=ta.EMA(df['HA_Low'], sml)
            
    return df

def pump_warning(dataframe, perc=15):
    df = dataframe.copy()    
    df["change"] = df["high"] - df["low"]
    df["test1"] = (df["close"] > df["open"])
    df["test2"] = ((df["change"]/df["low"]) > (perc/100))
    df["result"] = (df["test1"] & df["test2"]).astype('int')
    return df['result']

def tv_wma(dataframe, length = 9, field="close") -> DataFrame:
    """
    Source: Tradingview "Moving Average Weighted"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : WMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_wma'
    """

    norm = 0
    sum = 0

    for i in range(1, length - 1):
        weight = (length - i) * length
        norm = norm + weight
        sum = sum + dataframe[field].shift(i) * weight

    dataframe["tv_wma"] = (sum / norm) if norm > 0 else 0
    return dataframe["tv_wma"]

def tv_hma(dataframe, length = 9, field="close") -> DataFrame:
    """
    Source: Tradingview "Hull Moving Average"
    Pinescript Author: Unknown
    Args :
        dataframe : Pandas Dataframe
        length : HMA length
        field : Field to use for the calculation
    Returns :
        dataframe : Pandas DataFrame with new columns 'tv_hma'
    """

    dataframe["h"] = 2 * tv_wma(dataframe, math.floor(length / 2), field) - tv_wma(dataframe, length, field)

    dataframe["tv_hma"] = tv_wma(dataframe, math.floor(math.sqrt(length)), "h")
    # dataframe.drop("h", inplace=True, axis=1)

    return dataframe["tv_hma"]

def SSLChannels(dataframe, length = 7):
    df = dataframe.copy()
    df['ATR'] = ta.ATR(df, timeperiod=14)
    df['smaHigh'] = df['high'].rolling(length).mean() + df['ATR']
    df['smaLow'] = df['low'].rolling(length).mean() - df['ATR']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']

def ssl_atr(dataframe, length = 7):
    df = dataframe.copy()
    df['smaHigh'] = df['high'].rolling(length).mean() + df['atr']
    df['smaLow'] = df['low'].rolling(length).mean() - df['atr']
    df['hlv'] = np.where(df['close'] > df['smaHigh'], 1, np.where(df['close'] < df['smaLow'], -1, np.NAN))
    df['hlv'] = df['hlv'].ffill()
    df['sslDown'] = np.where(df['hlv'] < 0, df['smaHigh'], df['smaLow'])
    df['sslUp'] = np.where(df['hlv'] < 0, df['smaLow'], df['smaHigh'])
    return df['sslDown'], df['sslUp']
    
def momdiv(dataframe: DataFrame, mom_length: int = 10, bb_length: int = 20, bb_dev: float = 2.0, lookback: int = 30) -> DataFrame:
    mom: Series = ta.MOM(dataframe, timeperiod=mom_length)
    upperband, middleband, lowerband = ta.BBANDS(mom, timeperiod=bb_length, nbdevup=bb_dev, nbdevdn=bb_dev, matype=0)
    buy = qtpylib.crossed_below(mom, lowerband)
    sell = qtpylib.crossed_above(mom, upperband)
    hh = dataframe['high'].rolling(lookback).max()
    ll = dataframe['low'].rolling(lookback).min()
    coh = dataframe['high'] >= hh
    col = dataframe['low'] <= ll
    df = DataFrame({
            "momdiv_mom": mom,
            "momdiv_upperb": upperband,
            "momdiv_lowerb": lowerband,
            "momdiv_buy": buy,
            "momdiv_sell": sell,
            "momdiv_coh": coh,
            "momdiv_col": col,
        }, index=dataframe['close'].index)
    return df

# Williams %R
def williams_r(dataframe: DataFrame, period: int = 14) -> Series:
    """Williams %R, or just %R, is a technical analysis oscillator showing the current closing price in relation to the high and low
        of the past N days (for a given N). It was developed by a publisher and promoter of trading materials, Larry Williams.
        Its purpose is to tell whether a stock or commodity market is trading near the high or the low, or somewhere in between,
        of its recent trading range.
        The oscillator is on a negative scale, from −100 (lowest) up to 0 (highest).
    """

    highest_high = dataframe["high"].rolling(center=False, window=period).max()
    lowest_low = dataframe["low"].rolling(center=False, window=period).min()

    WR = Series(
        (highest_high - dataframe["close"]) / (highest_high - lowest_low),
        name=f"{period} Williams %R",
        )

    return WR * -100

# Chaikin Money Flow
def chaikin_money_flow(dataframe, n=20, fillna=False) -> Series:
    """Chaikin Money Flow (CMF)
    It measures the amount of Money Flow Volume over a specific period.
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf
    Args:
        dataframe(pandas.Dataframe): dataframe containing ohlcv
        n(int): n period.
        fillna(bool): if True, fill nan values.
    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((dataframe['close'] - dataframe['low']) - (dataframe['high'] - dataframe['close'])) / (dataframe['high'] - dataframe['low'])
    mfv = mfv.fillna(0.0)  # float division by zero
    mfv *= dataframe['volume']
    cmf = (mfv.rolling(n, min_periods=0).sum()
           / dataframe['volume'].rolling(n, min_periods=0).sum())
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return Series(cmf, name='cmf')
###################################################################################################################################