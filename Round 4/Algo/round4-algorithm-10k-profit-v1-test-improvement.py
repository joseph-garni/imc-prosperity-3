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
            'CROISSANTS': 250, 'JAMS': 350, 'DJEMBES': 60,
            'PICNIC_BASKET1': 60, 'PICNIC_BASKET2': 100,
            'RAINFOREST_RESIN': 50, 'KELP': 50, 'SQUID_INK': 50,
            'VOLCANIC_ROCK': 400,
            'VOLCANIC_ROCK_VOUCHER_9500': 200, 'VOLCANIC_ROCK_VOUCHER_9750': 200,
            'VOLCANIC_ROCK_VOUCHER_10000': 200, 'VOLCANIC_ROCK_VOUCHER_10250': 200,
            'VOLCANIC_ROCK_VOUCHER_10500': 200,
            'MAGNIFICENT_MACARONS': 75  # New product
        }
        
        # State variables for trading
        self.price_history = {}
        self.ma_windows = {
            'SQUID_INK': [10, 20, 50],  # Different windows for SQUID_INK
            'default': [5, 10, 20]       # Default moving average windows
        }
        
        # Voucher details
        self.voucher_details = {
            'VOLCANIC_ROCK_VOUCHER_9500': {'strike': 9500, 'initial_days': 7},
            'VOLCANIC_ROCK_VOUCHER_9750': {'strike': 9750, 'initial_days': 7},
            'VOLCANIC_ROCK_VOUCHER_10000': {'strike': 10000, 'initial_days': 7},
            'VOLCANIC_ROCK_VOUCHER_10250': {'strike': 10250, 'initial_days': 7},
            'VOLCANIC_ROCK_VOUCHER_10500': {'strike': 10500, 'initial_days': 7}
        }
        
        # Volatility surface data
        self.iv_data = {}
        self.volatility_params = {'a': 0, 'b': 0, 'c': 0.2}
        self.base_iv_history = []
        
        # Round tracking for option expiry calculation
        self.current_round = 0
        self.last_timestamp = 0
        
        # MAGNIFICENT_MACARONS specific variables
        self.sunlight_history = []
        self.sugar_price_history = []
        self.macaron_price_history = []
        self.macaron_pnl_history = []  # Track profit/loss history
        self.last_conversion_timestamp = 0
        self.conversion_cooldown = 5
        self.trade_aggressiveness = 0.8  # Default aggressiveness factor (0.0-1.0)
        
        self.initialized = False
    
    def update_price_history(self, product, mid_price, timestamp):
        if product not in self.price_history:
            self.price_history[product] = {'price': [], 'timestamp': []}    
        
        self.price_history[product]['price'].append(mid_price)
        self.price_history[product]['timestamp'].append(timestamp)
        
        # Keep only recent history
        if len(self.price_history[product]['price']) > 200:
            self.price_history[product]['price'].pop(0)
            self.price_history[product]['timestamp'].pop(0)
    
    def calculate_moving_averages(self, product):
        if product not in self.price_history or len(self.price_history[product]['price']) < 5:
            return {}
        
        prices = self.price_history[product]['price']
        windows = self.ma_windows.get(product, self.ma_windows['default'])
        
        ma_values = {}
        for window in windows:
            if len(prices) >= window:
                ma_values[f'ma_{window}'] = sum(prices[-window:]) / window
        
        return ma_values
    
    def calculate_mid_price(self, order_depth):
        if not order_depth.buy_orders or not order_depth.sell_orders:
            if order_depth.buy_orders:
                return max(order_depth.buy_orders.keys())
            elif order_depth.sell_orders:
                return min(order_depth.sell_orders.keys())
            return None
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2
    
    def get_current_position(self, state, product):
        if product in state.position:
            return state.position[product]
        return 0
    
    def calculate_basket_values(self, mid_prices):
        if not all(p in mid_prices for p in ['CROISSANTS', 'JAMS', 'DJEMBES']):
            return {}
        
        basket1_value = (
            6 * mid_prices['CROISSANTS'] + 
            3 * mid_prices['JAMS'] + 
            1 * mid_prices['DJEMBES']
        )
        
        basket2_value = (
            4 * mid_prices['CROISSANTS'] + 
            2 * mid_prices['JAMS']
        )
        
        return {
            'PICNIC_BASKET1': basket1_value,
            'PICNIC_BASKET2': basket2_value
        }
    
    def normal_cdf(self, x):
        a1, a2, a3, a4, a5 = 0.254829592, -0.284496736, 1.421413741, -1.453152027, 1.061405429
        p = 0.3275911
        sign = 1 if x >= 0 else -1
        x = abs(x)
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        return 0.5 * (1.0 + sign * y)
    
    def normal_pdf(self, x):
        return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

    def black_scholes_call(self, S, K, T, r, sigma):
        if T <= 0 or sigma <= 0:
            return max(0, S - K)
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        call_price = S * self.normal_cdf(d1) - K * math.exp(-r * T) * self.normal_cdf(d2)
        return max(0, call_price)
    
    def implied_volatility(self, S, K, T, r, market_price, precision=0.0001, max_iterations=100):
        if market_price <= 0 or T <= 0:
            return 0.0
        
        intrinsic = max(0, S - K)
        if market_price <= intrinsic:
            return 0.01
        
        sigma_low, sigma_high = 0.001, 5.0
        
        for i in range(max_iterations):
            sigma_mid = (sigma_low + sigma_high) / 2
            price = self.black_scholes_call(S, K, T, r, sigma_mid)
            
            if abs(price - market_price) < precision:
                return sigma_mid
            
            if price < market_price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
            
            if sigma_high - sigma_low < precision:
                return sigma_mid
        
        return (sigma_low + sigma_high) / 2
    
    def calculate_moneyness(self, S, K, T):
        if T <= 0 or S <= 0 or K <= 0:
            return 0
        return math.log(K / S) / math.sqrt(T)
    
    def calculate_volatility(self, moneyness):
        a, b, c = self.volatility_params['a'], self.volatility_params['b'], self.volatility_params['c']
        vol = a * moneyness**2 + b * moneyness + c
        return max(0.05, min(2.0, vol))
    
    def calculate_time_to_expiry(self):
        days_remaining = max(2, 7 - (self.current_round - 1))
        return days_remaining / 252
    
    def calculate_theoretical_voucher_prices(self, volcanic_rock_price):
        r = 0.01
        T = self.calculate_time_to_expiry()
        theoretical_prices = {}
        
        for voucher, details in self.voucher_details.items():
            strike = details['strike']
            moneyness = self.calculate_moneyness(volcanic_rock_price, strike, T)
            sigma = self.calculate_volatility(moneyness)
            price = self.black_scholes_call(volcanic_rock_price, strike, T, r, sigma)
            theoretical_prices[voucher] = price
        
        return theoretical_prices
    
    def update_implied_vols(self, volcanic_rock_price, voucher_prices, timestamp):
        r = 0.01
        T = self.calculate_time_to_expiry()
        
        if T <= 0:
            return
        
        self.iv_data[timestamp] = []
        
        for voucher, details in self.voucher_details.items():
            if voucher not in voucher_prices:
                continue
            
            try:
                strike = details['strike']
                market_price = voucher_prices[voucher]
                iv = self.implied_volatility(volcanic_rock_price, strike, T, r, market_price)
                moneyness = self.calculate_moneyness(volcanic_rock_price, strike, T)
                self.iv_data[timestamp].append((moneyness, iv))
                
                if abs(moneyness) < 0.1:
                    self.base_iv_history.append((timestamp, iv))
            except:
                continue
    
    def trade_volcanic_rock_vouchers(self, product, order_depth, state, mid_price, theoretical_prices):
        orders = []
        current_position = self.get_current_position(state, product)
        max_position = self.position_limits[product]
        
        if product not in theoretical_prices or mid_price <= 0 or theoretical_prices[product] <= 0:
            return []
        
        theoretical_value = theoretical_prices[product]
        price_diff_pct = (mid_price - theoretical_value) / theoretical_value
        threshold = 0.03
        position_ratio = current_position / max_position if max_position > 0 else 0
        
        # If voucher is overpriced, sell it
        if price_diff_pct > threshold and current_position > -max_position:
            aggression_factor = 1.0 - abs(position_ratio)
            order_size = max(1, int(max_position * aggression_factor * min(1, price_diff_pct * 10)))
            order_size = min(order_size, max_position + current_position)
            
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                sell_volume = min(abs(volume), order_size)
                if sell_volume > 0:
                    orders.append(Order(product, price, -sell_volume))
                    current_position -= sell_volume
                    order_size -= sell_volume
                    if order_size <= 0 or current_position <= -max_position:
                        break
        
        # If voucher is underpriced, buy it
        elif price_diff_pct < -threshold and current_position < max_position:
            aggression_factor = 1.0 - abs(position_ratio)
            order_size = max(1, int(max_position * aggression_factor * min(1, abs(price_diff_pct) * 10)))
            order_size = min(order_size, max_position - current_position)
            
            for price, volume in sorted(order_depth.sell_orders.items()):
                buy_volume = min(abs(volume), order_size)
                if buy_volume > 0:
                    orders.append(Order(product, price, buy_volume))
                    current_position += buy_volume
                    order_size -= buy_volume
                    if order_size <= 0 or current_position >= max_position:
                        break
        
        return orders
    
    def trade_volcanic_rock(self, product, order_depth, state, mid_price):
        orders = []
        current_position = self.get_current_position(state, product)
        max_position = self.position_limits[product]
        
        # Calculate aggregate voucher position
        voucher_position = 0
        for voucher in self.voucher_details.keys():
            voucher_position += self.get_current_position(state, voucher)
        
        # Simple delta hedge
        T = self.calculate_time_to_expiry()
        hedge_ratio = 1.0 if T <= 0 else 0.5
        
        target_position = -int(voucher_position * hedge_ratio)
        target_position = max(-max_position, min(max_position, target_position))
        quantity_needed = target_position - current_position
        
        if quantity_needed == 0:
            return []
        
        # Execute orders
        if quantity_needed > 0:  # Buy
            for price, volume in sorted(order_depth.sell_orders.items()):
                executable_volume = min(abs(quantity_needed), abs(volume))
                if executable_volume > 0:
                    orders.append(Order(product, price, executable_volume))
                    quantity_needed -= executable_volume
                    if quantity_needed <= 0:
                        break
        else:  # Sell
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                executable_volume = min(abs(quantity_needed), abs(volume))
                if executable_volume > 0:
                    orders.append(Order(product, price, -executable_volume))
                    quantity_needed += executable_volume
                    if quantity_needed >= 0:
                        break
        
        return orders
    
    def trade_magnificent_macarons(self, order_depth, state, sunlight_index, sugar_price):
        orders = []
        product = 'MAGNIFICENT_MACARONS'
        current_position = self.get_current_position(state, product)
        max_position = self.position_limits[product]
        
        # Update history
        self.sunlight_history.append(sunlight_index)
        self.sugar_price_history.append(sugar_price)
        
        # Calculate mid price if possible
        mid_price = None
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            self.macaron_price_history.append(mid_price)
        
        # Keep history at a larger size (up to 500 data points)
        max_history = 500
        if len(self.sunlight_history) > max_history:
            self.sunlight_history.pop(0)
            self.sugar_price_history.pop(0)
            if len(self.macaron_price_history) > max_history:
                self.macaron_price_history.pop(0)
        
        # Need enough history to make decisions
        if len(self.sunlight_history) < 20:
            return []  # Not enough data yet
        
        # Calculate sunlight statistics
        recent_sunlight = self.sunlight_history[-20:]  # Last 20 points
        all_sunlight = self.sunlight_history  # All available history points
        
        avg_sunlight = sum(all_sunlight) / len(all_sunlight)
        std_sunlight = math.sqrt(sum((x - avg_sunlight) ** 2 for x in all_sunlight) / len(all_sunlight))
        
        # Define thresholds for significant deviation (z-score)
        low_threshold = -0.8  # Below this is considered "significantly low"
        high_threshold = 0.8  # Above this is considered "significantly high"
        
        # Calculate z-score for current sunlight
        if std_sunlight > 0:
            z_score = (sunlight_index - avg_sunlight) / std_sunlight
        else:
            z_score = 0
        
        # Aggressive trading based on sunlight z-score
        if z_score < low_threshold:
            # Significantly low sunlight - prices likely to rise steeply
            # Cover short positions first
            if current_position < 0:
                for price, volume in sorted(order_depth.sell_orders.items()):
                    buy_volume = min(abs(volume), abs(current_position))
                    if buy_volume > 0:
                        orders.append(Order(product, price, buy_volume))
                        current_position += buy_volume
                        if current_position >= 0:
                            break
            
            # Then go long aggressively
            if current_position < max_position:
                # Calculate aggression factor based on how extreme the z-score is
                aggression = min(1.0, abs(z_score / low_threshold))
                target_position = int(max_position * aggression)
                
                # Ensure we don't exceed max position
                target_position = min(target_position, max_position)
                
                # Calculate quantity needed
                quantity_needed = target_position - current_position
                
                if quantity_needed > 0:
                    for price, volume in sorted(order_depth.sell_orders.items()):
                        buy_volume = min(abs(volume), quantity_needed)
                        if buy_volume > 0:
                            orders.append(Order(product, price, buy_volume))
                            current_position += buy_volume
                            quantity_needed -= buy_volume
                            if quantity_needed <= 0 or current_position >= max_position:
                                break
        
        elif z_score > high_threshold:
            # Significantly high sunlight - prices likely to fall steeply
            # Reduce long positions first
            if current_position > 0:
                for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    sell_volume = min(abs(volume), current_position)
                    if sell_volume > 0:
                        orders.append(Order(product, price, -sell_volume))
                        current_position -= sell_volume
                        if current_position <= 0:
                            break
            
            # Then go short aggressively
            if current_position > -max_position:
                # Calculate aggression factor based on how extreme the z-score is
                aggression = min(1.0, abs(z_score / high_threshold))
                target_position = -int(max_position * aggression)
                
                # Ensure we don't exceed max position
                target_position = max(target_position, -max_position)
                
                # Calculate quantity needed
                quantity_needed = target_position - current_position
                
                if quantity_needed < 0:
                    for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                        sell_volume = min(abs(volume), abs(quantity_needed))
                        if sell_volume > 0:
                            orders.append(Order(product, price, -sell_volume))
                            current_position -= sell_volume
                            quantity_needed += sell_volume
                            if quantity_needed >= 0 or current_position <= -max_position:
                                break
        else:
            # Sunlight not significantly different - move toward neutral position
            # If we have a significant position, gradually reduce it
            if current_position > 20:
                # Reduce long position
                for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    sell_volume = min(abs(volume), current_position - 10)  # Leave a small position
                    if sell_volume > 0:
                        orders.append(Order(product, price, -sell_volume))
                        current_position -= sell_volume
                        if current_position <= 10:
                            break
            elif current_position < -20:
                # Reduce short position
                for price, volume in sorted(order_depth.sell_orders.items()):
                    buy_volume = min(abs(volume), abs(current_position) - 10)  # Leave a small position
                    if buy_volume > 0:
                        orders.append(Order(product, price, buy_volume))
                        current_position += buy_volume
                        if current_position >= -10:
                            break
        
        return orders
    
    def should_convert_macarons(self, state, current_position):
        # Skip if cooldown period hasn't elapsed
        if state.timestamp - self.last_conversion_timestamp < self.conversion_cooldown:
            return 0
        
        # More aggressive conversion strategy
        # For very large positions, always convert the maximum allowed
        if current_position > 50:
            self.last_conversion_timestamp = state.timestamp
            return 10  # Maximum conversion limit is 10
        
        # For medium positions, convert if profitable or we need to reduce risk
        elif current_position > 25:
            self.last_conversion_timestamp = state.timestamp
            return min(current_position // 2, 10)  # Convert up to half of position, max 10
        
        # For very large short positions, always convert the maximum allowed
        elif current_position < -50:
            self.last_conversion_timestamp = state.timestamp
            return -10  # Maximum conversion limit is -10
        
        # For medium short positions, convert if profitable or we need to reduce risk
        elif current_position < -25:
            self.last_conversion_timestamp = state.timestamp
            return max(current_position // 2, -10)  # Convert up to half of position, max -10
        
        return 0
    
    def trade_based_on_moving_average(self, product, order_depth, state, mid_price):
        orders = []
        ma_values = self.calculate_moving_averages(product)
        current_position = self.get_current_position(state, product)
        
        if not ma_values:
            return []
        
        # Calculate signal from moving averages
        signal = 0
        for ma_key, ma_value in ma_values.items():
            deviation = (mid_price - ma_value) / ma_value
            
            if product == 'SQUID_INK':
                # Mean reversion for SQUID_INK
                signal -= deviation * 3
            else:
                # Trend following for others
                signal += deviation
        
        # Normalize and calculate target position
        signal = signal / len(ma_values)
        max_position = self.position_limits[product]
        position_target = int(signal * max_position)
        position_target = max(-max_position, min(max_position, position_target))
        quantity_needed = position_target - current_position
        
        if quantity_needed == 0:
            return []
        
        # Execute orders to reach target
        if quantity_needed > 0:  # Buy
            for price, volume in sorted(order_depth.sell_orders.items()):
                executable_volume = min(abs(quantity_needed), abs(volume))
                if executable_volume > 0:
                    orders.append(Order(product, price, executable_volume))
                    quantity_needed -= executable_volume
                    if quantity_needed <= 0:
                        break
        else:  # Sell
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                executable_volume = min(abs(quantity_needed), abs(volume))
                if executable_volume > 0:
                    orders.append(Order(product, price, -executable_volume))
                    quantity_needed += executable_volume
                    if quantity_needed >= 0:
                        break
        
        return orders
    
    def trade_basket_arbitrage(self, product, order_depth, state, mid_price, basket_values):
        if product not in basket_values:
            return []
        
        orders = []
        current_position = self.get_current_position(state, product)
        max_position = self.position_limits[product]
        theoretical_value = basket_values[product]
        price_diff = mid_price - theoretical_value
        threshold = 10
        
        # If basket is overpriced, sell it
        if price_diff > threshold and current_position > -max_position:
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                sell_volume = min(abs(volume), max_position + current_position)
                if sell_volume > 0:
                    orders.append(Order(product, price, -sell_volume))
                    current_position -= sell_volume
                    if current_position <= -max_position:
                        break
        
        # If basket is underpriced, buy it
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
        # Initialize result dictionary to store orders for each product
        result = {}
        
        # Parse trader data if available
        if state.traderData:
            try:
                trader_data = jsonpickle.decode(state.traderData)
                if isinstance(trader_data, dict) and "round" in trader_data:
                    self.current_round = trader_data["round"]
            except:
                self.current_round = 1
        
        # Update round counter if new day detected
        if hasattr(self, 'last_timestamp') and state.timestamp - self.last_timestamp > 5000:
            self.current_round += 1
        self.last_timestamp = state.timestamp
        
        # Dictionary to store all mid prices
        mid_prices = {}
        
        # Variable to track conversions for MAGNIFICENT_MACARONS
        conversions = 0
        
        # First pass: Calculate mid prices and update price history
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            mid_price = self.calculate_mid_price(order_depth)
            if mid_price:
                mid_prices[product] = mid_price
                self.update_price_history(product, mid_price, state.timestamp)
        
        # Calculate theoretical basket values
        basket_values = self.calculate_basket_values(mid_prices)
        
        # Calculate theoretical voucher prices if VOLCANIC_ROCK price exists
        theoretical_voucher_prices = {}
        if 'VOLCANIC_ROCK' in mid_prices:
            volcanic_rock_price = mid_prices['VOLCANIC_ROCK']
            theoretical_voucher_prices = self.calculate_theoretical_voucher_prices(volcanic_rock_price)
            
            # Update implied volatilities based on market prices
            voucher_mid_prices = {product: price for product, price in mid_prices.items() 
                                 if product.startswith('VOLCANIC_ROCK_VOUCHER')}
            self.update_implied_vols(volcanic_rock_price, voucher_mid_prices, state.timestamp)
        
        # Get MAGNIFICENT_MACARONS observation if available
        macaron_sunlight = 0
        macaron_sugar_price = 0
        if hasattr(state, 'observations') and state.observations is not None:
            if hasattr(state.observations, 'MAGNIFICENT_MACARONS') and state.observations.MAGNIFICENT_MACARONS is not None:
                macaron_obs = state.observations.MAGNIFICENT_MACARONS
                if hasattr(macaron_obs, 'sunlightIndex'):
                    macaron_sunlight = macaron_obs.sunlightIndex
                if hasattr(macaron_obs, 'sugarPrice'):
                    macaron_sugar_price = macaron_obs.sugarPrice
        
        # Second pass: Execute trading strategies for each product
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []
            
            # Handle MAGNIFICENT_MACARONS specially
            if product == 'MAGNIFICENT_MACARONS':
                orders = self.trade_magnificent_macarons(order_depth, state, macaron_sunlight, macaron_sugar_price)
                
                # Check if we should convert macarons
                current_position = self.get_current_position(state, product)
                conversions = self.should_convert_macarons(state, current_position)
            # Skip if no mid price is available
            elif product not in mid_prices:
                result[product] = []
                continue
            # Apply appropriate strategy based on product type
            elif product in ['PICNIC_BASKET1', 'PICNIC_BASKET2']:
                # Use basket arbitrage strategy
                mid_price = mid_prices[product]
                orders = self.trade_basket_arbitrage(product, order_depth, state, mid_price, basket_values)
            elif product.startswith('VOLCANIC_ROCK_VOUCHER'):
                # Use options pricing strategy for volcanic rock vouchers
                mid_price = mid_prices[product]
                orders = self.trade_volcanic_rock_vouchers(product, order_depth, state, mid_price, theoretical_voucher_prices)
            elif product == 'VOLCANIC_ROCK':
                # Use delta hedging strategy for volcanic rock
                mid_price = mid_prices[product]
                orders = self.trade_volcanic_rock(product, order_depth, state, mid_price)
            elif product == 'SQUID_INK':
                # Use mean reversion strategy for SQUID_INK
                mid_price = mid_prices[product]
                orders = self.trade_based_on_moving_average(product, order_depth, state, mid_price)
            else:
                # Use standard moving average strategy for other products
                mid_price = mid_prices[product]
                orders = self.trade_based_on_moving_average(product, order_depth, state, mid_price)
            
            result[product] = orders
        
        # Save state data with updated round
        traderData = jsonpickle.encode({"round": self.current_round})
        
        return result, conversions, traderData