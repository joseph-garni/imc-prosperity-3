import numpy as np
import math
import pandas as pd
import statistics
import jsonpickle
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple

class Trader:
    def __init__(self):
        # Initialize position limits for all products
        self.position_limits = {
            'CROISSANTS': 250,
            'JAMS': 350,
            'DJEMBES': 60,
            'PICNIC_BASKET1': 60,
            'PICNIC_BASKET2': 100,
            'RAINFOREST_RESIN': 50,
            'KELP': 50,
            'SQUID_INK': 50,
            'VOLCANIC_ROCK': 400,
            'VOLCANIC_ROCK_VOUCHER_9500': 200,
            'VOLCANIC_ROCK_VOUCHER_9750': 200,
            'VOLCANIC_ROCK_VOUCHER_10000': 200,
            'VOLCANIC_ROCK_VOUCHER_10250': 200,
            'VOLCANIC_ROCK_VOUCHER_10500': 200
        }
        
        # State variables for LSTM-like memory
        self.price_history = {}
        self.ma_windows = {
            'SQUID_INK': [10, 20, 50],  # Different windows for mean reversion on SQUID_INK
            'default': [5, 10, 20]       # Default moving average windows for other products
        }
        self.initialized = False
    
    # appends price_history to dictionary for each product using price and timestamp, then sets price_history of produce using the best mid_price at each time interval over the last 200 data points.
    def update_price_history(self, product, mid_price, timestamp):
        """Update price history with new mid price"""
        if product not in self.price_history:
            self.price_history[product] = {'price': [], 'timestamp': []}
        
        self.price_history[product]['price'].append(mid_price)
        self.price_history[product]['timestamp'].append(timestamp)
        
        # Keep only the last 200 prices (memory management)
        if len(self.price_history[product]['price']) > 200:
            self.price_history[product]['price'].pop(0)
            self.price_history[product]['timestamp'].pop(0)
    
    # calculate moving average for each product if there are more than 5 price points to calculate ma from
    def calculate_moving_averages(self, product):
        """Calculate moving averages for a product"""
        if product not in self.price_history or len(self.price_history[product]['price']) < 5:
            return {}
        
        prices = self.price_history[product]['price']
        windows = self.ma_windows.get(product, self.ma_windows['default'])
        
        ma_values = {}
        for window in windows: #for default [5, 10, 20] it will run 5, 10, 20 as window
            if len(prices) >= window: # if len of prices in price history for product is greater than or equal to 
                ma_values[f'ma_{window}'] = sum(prices[-window:]) / window
                #if we have 200 price values for JAMS
                # len(prices) = 200
                # for 5, 10, 20
                #if 200 >= 5:
                #    setting ma_values[f'ma_5'] = sum['prices[last 5 items in price history:]'] / window (5)
                #if len(prices)
        
        return ma_values # returns dictionary of moving averages 
    
    # mid price checks if there are both buy orders and sell orders on the orderbook, then calculates mid price from the best values (max value on buy orders -> top value in list, and min value on sell orders -> lowest value in sell list)
    def calculate_mid_price(self, order_depth):
        """Calculate mid price from order depth"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            return None
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2
    
    # current position either returns 0 or our algorithms currently traded position in product
    def get_current_position(self, state, product):
        """Get current position for a product"""
        if product in state.position:
            return state.position[product]
        return 0

    # position in the form:
    # position = {"PRODUCT1": 3} or {"PRODUCT1": -10}
    
    def calculate_basket_values(self, mid_prices):
        # checks if there are values for mid_price for each value in each basket, 'CROISSANTS', 'JAMS', 'DJEMBES', if they are not values for all of them, return empty dictionary
        """Calculate theoretical values for picnic baskets"""
        if not all(p in mid_prices for p in ['CROISSANTS', 'JAMS', 'DJEMBES']):
            return {}
        
        # PICNIC_BASKET1: 6 CROISSANTS, 3 JAMS, 1 DJEMBE
        basket1_value = (
            6 * mid_prices['CROISSANTS'] + 
            3 * mid_prices['JAMS'] + 
            1 * mid_prices['DJEMBES']
        )
        
        # PICNIC_BASKET2: 4 CROISSANTS, 2 JAMS
        basket2_value = (
            4 * mid_prices['CROISSANTS'] + 
            2 * mid_prices['JAMS']
        )
        
        return {
            'PICNIC_BASKET1': basket1_value,
            'PICNIC_BASKET2': basket2_value
        }
    
    def trade_based_on_moving_average(self, product, order_depth, state, mid_price):
        """Implement trading logic based on moving averages"""
        orders = []
        ma_values = self.calculate_moving_averages(product)
        current_position = self.get_current_position(state, product)
        
        # Not enough history for this product
        if not ma_values:
            return []
        
        # Calculate signal strength based on deviation from moving averages
        signal = 0
        for ma_key, ma_value in ma_values.items():
            deviation = (mid_price - ma_value) / ma_value
            
            # For SQUID_INK, use stronger mean reversion due to high volatility
            if product == 'SQUID_INK':
                # Negative correlation means price tends to revert to mean
                signal -= deviation * 3  # Stronger mean reversion factor
            else:
                #here products, use standard trend following
                signal += deviation
                # For ot
        
        # Normalize signal
        signal = signal / len(ma_values)
        
        # Determine position target based on signal
        max_position = self.position_limits[product]
        position_target = int(signal * max_position)

        #max position is the position limits for the product
        # for position target we either use position target or max position
        # Limit position target to position limits
        position_target = max(-max_position, min(max_position, position_target))
        
        # Calculate quantity needed to reach target
        # quantity needed is signal value multiplied by max position (i.e 200) i.e 200*0.1  minus current positions in quantity
        quantity_needed = position_target - current_position
        
        # If no action needed
        if quantity_needed == 0:
            return []
        
        # Place orders to reach the target position
        if quantity_needed > 0:  # Need to buy
            for price, volume in sorted(order_depth.sell_orders.items()): # sorted list of price and volume, with sell orders in lowest to highest so we send buy orders to match with the lowest possible sell orders on the order book
                # Don't exceed quantity needed
                executable_volume = min(abs(quantity_needed), abs(volume))
                if executable_volume > 0:
                    orders.append(Order(product, price, executable_volume))
                    quantity_needed -= executable_volume # subtract executable volume from quantity needed
                    if quantity_needed <= 0:
                        break
        else:  # Need to sell # if quantity needed is negative these are sell orders
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True): # sorted list of price and volume, with buy order items set highest to lowest -> as we are selling to highest bidder
                # Don't exceed quantity needed
                executable_volume = min(abs(quantity_needed), abs(volume))
                if executable_volume > 0:
                    orders.append(Order(product, price, -executable_volume))
                    quantity_needed += executable_volume
                    if quantity_needed >= 0:
                        break
        
        return orders
    
    def trade_basket_arbitrage(self, product, order_depth, state, mid_price, basket_values):
        """Trade basket products based on arbitrage opportunities"""
        orders = []
        current_position = self.get_current_position(state, product)
        max_position = self.position_limits[product]
        
        # Skip if no basket value is available
        if product not in basket_values:
            return []
        
        theoretical_value = basket_values[product]
        
        # Calculate price difference (absolute)
        price_diff = mid_price - theoretical_value
        
        # Determine if basket is overpriced or underpriced
        # Use threshold to account for transaction costs
        threshold = 10
        
        # If the basket is significantly overpriced, sell it
        if price_diff > threshold and current_position > -max_position:
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                sell_volume = min(abs(volume), max_position + current_position)
                if sell_volume > 0:
                    orders.append(Order(product, price, -sell_volume))
                    current_position -= sell_volume
                    if current_position <= -max_position:
                        break
        
        # If the basket is significantly underpriced, buy it
        elif price_diff < -threshold and current_position < max_position:
            for price, volume in sorted(order_depth.sell_orders.items()):
                buy_volume = min(abs(volume), max_position - current_position)
                if buy_volume > 0:
                    orders.append(Order(product, price, buy_volume))
                    current_position += buy_volume
                    if current_position >= max_position:
                        break
        
        return orders
    
    def run(self, state: TradingState):
        """Main trading method"""
        # Initialize trader data if needed
        if not state.traderData:
            state.traderData = "LSTM_TRADER_V1"
        
        # Store current timestamp
        timestamp = state.timestamp
        
        # Dictionary to store all orders
        result = {}
        
        # Dictionary to store all mid prices
        mid_prices = {}
        
        # First pass: Calculate mid prices and update price history
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            
            # Calculate mid price
            mid_price = self.calculate_mid_price(order_depth)
            if mid_price:
                mid_prices[product] = mid_price
                self.update_price_history(product, mid_price, timestamp)
        
        # Calculate theoretical basket values
        basket_values = self.calculate_basket_values(mid_prices)
        
        # Second pass: Execute trading strategies for each product
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []
            
            # Skip if no mid price is available
            if product not in mid_prices:
                result[product] = []
                continue
            
            mid_price = mid_prices[product]
            
            # Apply appropriate strategy based on product type
            if product in ['PICNIC_BASKET1', 'PICNIC_BASKET2']:
                # Use basket arbitrage strategy
                orders = self.trade_basket_arbitrage(product, order_depth, state, mid_price, basket_values)
            elif product == 'SQUID_INK':
                # Use mean reversion strategy for SQUID_INK
                orders = self.trade_based_on_moving_average(product, order_depth, state, mid_price)
            else:
                # Use standard moving average strategy for other products
                orders = self.trade_based_on_moving_average(product, order_depth, state, mid_price)
            
            result[product] = orders
        
        # Always return 0 conversions since we're not converting products
        conversions = 0
        
        # Save state data
        traderData = state.traderData
        
        return result, conversions, traderData