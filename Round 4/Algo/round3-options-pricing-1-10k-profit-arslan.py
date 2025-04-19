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
        
        # Options parameters for volcanic rock vouchers
        self.voucher_details = {
            'VOLCANIC_ROCK_VOUCHER_9500': {'strike': 9500, 'initial_days': 7},
            'VOLCANIC_ROCK_VOUCHER_9750': {'strike': 9750, 'initial_days': 7},
            'VOLCANIC_ROCK_VOUCHER_10000': {'strike': 10000, 'initial_days': 7},
            'VOLCANIC_ROCK_VOUCHER_10250': {'strike': 10250, 'initial_days': 7},
            'VOLCANIC_ROCK_VOUCHER_10500': {'strike': 10500, 'initial_days': 7}
        }
        
        # Store volatility surface data
        self.iv_data = {}  # {timestamp: {moneyness: implied_vol}}
        self.volatility_params = {'a': 0, 'b': 0, 'c': 0.2}  # Initial parameters for vol smile
        self.base_iv_history = []  # Store history of base IV
        
        # Round tracking (for time to expiry calculation)
        self.current_round = 0
        self.initialized = False
    
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
    
    def calculate_moving_averages(self, product):
        """Calculate moving averages for a product"""
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
        """Calculate mid price from order depth"""
        if not order_depth.buy_orders or not order_depth.sell_orders:
            # If only one side has orders, estimate based on that side
            if order_depth.buy_orders:
                return max(order_depth.buy_orders.keys())
            elif order_depth.sell_orders:
                return min(order_depth.sell_orders.keys())
            return None
        
        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())
        return (best_bid + best_ask) / 2
    
    def get_current_position(self, state, product):
        """Get current position for a product"""
        if product in state.position:
            return state.position[product]
        return 0
    
    def calculate_basket_values(self, mid_prices):
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
    
    def normal_cdf(self, x):
        """Approximation of the cumulative distribution function for the standard normal distribution"""
        # Constants for the approximation
        a1 = 0.254829592
        a2 = -0.284496736
        a3 = 1.421413741
        a4 = -1.453152027
        a5 = 1.061405429
        p = 0.3275911
        
        # Save the sign of x
        sign = 1
        if x < 0:
            sign = -1
        x = abs(x)
        
        # A&S formula 7.1.26
        t = 1.0 / (1.0 + p * x)
        y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * math.exp(-x * x)
        
        return 0.5 * (1.0 + sign * y)
    
    def normal_pdf(self, x):
        """Probability density function for the standard normal distribution"""
        return (1.0 / math.sqrt(2 * math.pi)) * math.exp(-0.5 * x * x)

    def black_scholes_call(self, S, K, T, r, sigma):
        """Calculate Black-Scholes price for a call option"""
        # S: spot price
        # K: strike price
        # T: time to maturity (in years)
        # r: risk-free rate (annual)
        # sigma: volatility (annual)
        
        # Handle edge cases
        if T <= 0:
            return max(0, S - K)  # Intrinsic value at expiration
        
        if sigma <= 0:
            return max(0, S - K)  # No volatility means intrinsic value
        
        d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        
        call_price = S * self.normal_cdf(d1) - K * math.exp(-r * T) * self.normal_cdf(d2)
        return max(0, call_price)  # Ensure non-negative price
    
    def implied_volatility(self, S, K, T, r, market_price, precision=0.0001, max_iterations=100):
        """Calculate implied volatility using binary search method"""
        # S: spot price
        # K: strike price
        # T: time to maturity (in years)
        # r: risk-free rate (annual)
        # market_price: observed option price
        
        # Handle edge cases
        if market_price <= 0 or T <= 0:
            return 0.0
        
        # Intrinsic value check
        intrinsic = max(0, S - K)
        if market_price <= intrinsic:
            return 0.01  # Return minimal volatility
        
        # Binary search for implied volatility
        sigma_low = 0.001
        sigma_high = 5.0
        
        for i in range(max_iterations):
            sigma_mid = (sigma_low + sigma_high) / 2
            price = self.black_scholes_call(S, K, T, r, sigma_mid)
            
            # Check if we're close enough
            if abs(price - market_price) < precision:
                return sigma_mid
            
            # Adjust search range
            if price < market_price:
                sigma_low = sigma_mid
            else:
                sigma_high = sigma_mid
            
            # Check for convergence
            if sigma_high - sigma_low < precision:
                return sigma_mid
        
        # Return middle value if we've reached max iterations
        return (sigma_low + sigma_high) / 2
    
    def calculate_moneyness(self, S, K, T):
        """Calculate moneyness, as per the hint: m_t = log(K/St) / sqrt(TTE)"""
        if T <= 0 or S <= 0 or K <= 0:
            return 0
        
        return math.log(K / S) / math.sqrt(T)
    
    def fit_volatility_surface(self, vols_data):
        """Fit a parabolic curve to the volatility smile data
        Fitting: v_t = a * m_t^2 + b * m_t + c
        Using least squares method manually since np.polyfit isn't available
        """
        if not vols_data or len(vols_data) < 3:
            # Not enough data points for fitting
            return {'a': 0, 'b': 0, 'c': 0.2}  # Default params
        
        # Extract moneyness and IV data
        moneyness_values = []
        iv_values = []
        
        for m, iv in vols_data:
            moneyness_values.append(m)
            iv_values.append(iv)
        
        if len(set(moneyness_values)) < 3:
            # Need at least 3 different points for a parabola
            return {'a': 0, 'b': 0, 'c': 0.2}  # Default params
        
        try:
            # Implement a basic least squares fit for a parabola
            # For y = ax^2 + bx + c
            n = len(moneyness_values)
            sum_x = sum(moneyness_values)
            sum_x2 = sum(x**2 for x in moneyness_values)
            sum_x3 = sum(x**3 for x in moneyness_values)
            sum_x4 = sum(x**4 for x in moneyness_values)
            sum_y = sum(iv_values)
            sum_xy = sum(x*y for x, y in zip(moneyness_values, iv_values))
            sum_x2y = sum(x**2*y for x, y in zip(moneyness_values, iv_values))
            
            # Solve the normal equations
            # [sum_x4 sum_x3 sum_x2] [a]   [sum_x2y]
            # [sum_x3 sum_x2 sum_x ] [b] = [sum_xy ]
            # [sum_x2 sum_x  n     ] [c]   [sum_y  ]
            
            # Using Cramer's rule for simplicity
            det_main = (sum_x4 * (sum_x2 * n - sum_x**2) - 
                        sum_x3 * (sum_x3 * n - sum_x * sum_x2) + 
                        sum_x2 * (sum_x3 * sum_x - sum_x2**2))
            
            if abs(det_main) < 1e-10:  # Near-zero determinant
                return {'a': 0, 'b': 0, 'c': 0.2}  # Default params
            
            det_a = (sum_x2y * (sum_x2 * n - sum_x**2) - 
                     sum_xy * (sum_x3 * n - sum_x * sum_x2) + 
                     sum_y * (sum_x3 * sum_x - sum_x2**2))
            
            det_b = (sum_x4 * (sum_xy * n - sum_y * sum_x) - 
                     sum_x2y * (sum_x3 * n - sum_x * sum_x2) + 
                     sum_x2 * (sum_x3 * sum_y - sum_x2 * sum_xy))
            
            det_c = (sum_x4 * (sum_x2 * sum_y - sum_x * sum_xy) - 
                     sum_x3 * (sum_x3 * sum_y - sum_x2 * sum_xy) + 
                     sum_x2y * (sum_x3 * sum_x - sum_x2**2))
            
            a = det_a / det_main
            b = det_b / det_main
            c = det_c / det_main
            
            # Limit parameters to reasonable ranges
            a = max(-1.0, min(1.0, a))
            b = max(-1.0, min(1.0, b))
            c = max(0.05, min(0.5, c))
            
            return {'a': a, 'b': b, 'c': c}
        except Exception:
            # Fallback if fitting fails
            return {'a': 0, 'b': 0, 'c': 0.2}  # Default params
    
    def calculate_volatility(self, moneyness):
        """Calculate volatility based on fitted volatility surface
        v_t = a * m_t^2 + b * m_t + c
        """
        a = self.volatility_params['a']
        b = self.volatility_params['b']
        c = self.volatility_params['c']
        
        # Calculate volatility using the parabolic formula
        vol = a * moneyness**2 + b * moneyness + c
        
        # Ensure a minimum positive volatility
        return max(0.05, min(2.0, vol))  # Cap volatility to reasonable range
    
    def calculate_time_to_expiry(self):
        """Calculate time to expiry in years based on current round"""
        # Round 1 corresponds to 7 days to expiry
        # Round 5 corresponds to 2 days to expiry
        days_remaining = max(2, 7 - (self.current_round - 1))
        
        # Convert days to years (assuming 252 trading days in a year)
        return days_remaining / 252
    
    def calculate_theoretical_voucher_prices(self, volcanic_rock_price):
        """Calculate theoretical prices for all vouchers"""
        # Use a fixed risk-free rate
        r = 0.01  # 1% annual risk-free rate
        
        # Calculate time to expiry
        T = self.calculate_time_to_expiry()
        
        theoretical_prices = {}
        vols_data = []  # For volatility surface fitting
        
        for voucher, details in self.voucher_details.items():
            strike = details['strike']
            
            # Calculate moneyness
            moneyness = self.calculate_moneyness(volcanic_rock_price, strike, T)
            
            # Get volatility for this moneyness (from volatility surface)
            sigma = self.calculate_volatility(moneyness)
            
            # Calculate theoretical price using Black-Scholes
            price = self.black_scholes_call(volcanic_rock_price, strike, T, r, sigma)
            theoretical_prices[voucher] = price
        
        return theoretical_prices
    
    def update_implied_vols(self, volcanic_rock_price, voucher_prices, timestamp):
        """Update implied volatility data based on market prices"""
        # Use a fixed risk-free rate
        r = 0.01  # 1% annual risk-free rate
        
        # Calculate time to expiry
        T = self.calculate_time_to_expiry()
        
        if T <= 0:
            return  # Skip if at or past expiry
        
        # Reset IV data for this timestamp
        self.iv_data[timestamp] = []
        
        for voucher, details in self.voucher_details.items():
            # Skip if no market price available
            if voucher not in voucher_prices:
                continue
            
            strike = details['strike']
            market_price = voucher_prices[voucher]
            
            # Calculate implied volatility
            try:
                iv = self.implied_volatility(volcanic_rock_price, strike, T, r, market_price)
                
                # Calculate moneyness
                moneyness = self.calculate_moneyness(volcanic_rock_price, strike, T)
                
                # Store data for volatility surface fitting
                self.iv_data[timestamp].append((moneyness, iv))
                
                # Store base IV (ATM volatility, m=0)
                if abs(moneyness) < 0.1:  # Close to at-the-money
                    self.base_iv_history.append((timestamp, iv))
            except Exception:
                # Skip if calculation fails
                continue
        
        # Fit volatility surface with latest data
        if timestamp in self.iv_data and len(self.iv_data[timestamp]) >= 3:
            self.volatility_params = self.fit_volatility_surface(self.iv_data[timestamp])
    
    def trade_volcanic_rock_vouchers(self, product, order_depth, state, mid_price, theoretical_prices):
        """Trade volcanic rock vouchers based on Black-Scholes pricing"""
        orders = []
        current_position = self.get_current_position(state, product)
        max_position = self.position_limits[product]
        
        # Skip if no theoretical price is available
        if product not in theoretical_prices:
            return []
        
        theoretical_value = theoretical_prices[product]
        
        # Calculate price difference (percentage)
        if mid_price <= 0 or theoretical_value <= 0:
            return []
            
        price_diff_pct = (mid_price - theoretical_value) / theoretical_value
        
        # Determine if voucher is overpriced or underpriced
        # Use threshold to account for model uncertainty and transaction costs
        threshold = 0.03  # 3% threshold
        
        # Adjust trading aggression based on current position and moneyness
        position_ratio = current_position / max_position if max_position > 0 else 0
        
        # Extract strike price from product name
        strike = float(product.split('_')[-1])
        
        # Get underlying price (VOLCANIC_ROCK)
        volcanic_rock_price = self.price_history.get('VOLCANIC_ROCK', {}).get('price', [])
        underlying_price = volcanic_rock_price[-1] if volcanic_rock_price else 0
        
        # Calculate moneyness for position sizing
        T = self.calculate_time_to_expiry()
        moneyness = self.calculate_moneyness(underlying_price, strike, T) if underlying_price > 0 else 0
        
        # If the voucher is significantly overpriced, sell it
        if price_diff_pct > threshold and current_position > -max_position:
            # Calculate order size based on price difference and position
            aggression_factor = 1.0 - abs(position_ratio)  # Reduce aggression as position grows
            order_size = max(1, int(max_position * aggression_factor * min(1, price_diff_pct * 10)))
            
            # Limit order size to available position room
            order_size = min(order_size, max_position + current_position)
            
            if order_size > 0:
                for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                    sell_volume = min(abs(volume), order_size)
                    if sell_volume > 0:
                        orders.append(Order(product, price, -sell_volume))
                        current_position -= sell_volume
                        order_size -= sell_volume
                        if order_size <= 0 or current_position <= -max_position:
                            break
        
        # If the voucher is significantly underpriced, buy it
        elif price_diff_pct < -threshold and current_position < max_position:
            # Calculate order size based on price difference and position
            aggression_factor = 1.0 - abs(position_ratio)  # Reduce aggression as position grows
            order_size = max(1, int(max_position * aggression_factor * min(1, abs(price_diff_pct) * 10)))
            
            # Limit order size to available position room
            order_size = min(order_size, max_position - current_position)
            
            if order_size > 0:
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
        """Trade volcanic rock based on voucher positions"""
        orders = []
        current_position = self.get_current_position(state, product)
        max_position = self.position_limits[product]
        
        # Calculate aggregate voucher position
        voucher_position = 0
        for voucher in self.voucher_details.keys():
            voucher_position += self.get_current_position(state, voucher)
        
        # Calculate delta hedge ratio (simplified approach)
        T = self.calculate_time_to_expiry()
        if T <= 0:
            hedge_ratio = 1.0  # At expiry, delta is 1 for ITM options
        else:
            hedge_ratio = 0.5  # Simplified delta approximation for ATM options
        
        # Target position for delta hedging
        target_position = -int(voucher_position * hedge_ratio)
        
        # Limit to position limits
        target_position = max(-max_position, min(max_position, target_position))
        
        # Calculate quantity needed
        quantity_needed = target_position - current_position
        
        # If no action needed
        if quantity_needed == 0:
            return []
        
        # Place orders to reach the target position
        if quantity_needed > 0:  # Need to buy
            for price, volume in sorted(order_depth.sell_orders.items()):
                # Don't exceed quantity needed
                executable_volume = min(abs(quantity_needed), abs(volume))
                if executable_volume > 0:
                    orders.append(Order(product, price, executable_volume))
                    quantity_needed -= executable_volume
                    if quantity_needed <= 0:
                        break
        else:  # Need to sell
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                # Don't exceed quantity needed
                executable_volume = min(abs(quantity_needed), abs(volume))
                if executable_volume > 0:
                    orders.append(Order(product, price, -executable_volume))
                    quantity_needed += executable_volume
                    if quantity_needed >= 0:
                        break
        
        return orders
    
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
                # For other products, use standard trend following
                signal += deviation
        
        # Normalize signal
        signal = signal / len(ma_values)
        
        # Determine position target based on signal
        max_position = self.position_limits[product]
        position_target = int(signal * max_position)
        
        # Limit position target to position limits
        position_target = max(-max_position, min(max_position, position_target))
        
        # Calculate quantity needed to reach target
        quantity_needed = position_target - current_position
        
        # If no action needed
        if quantity_needed == 0:
            return []
        
        # Place orders to reach the target position
        if quantity_needed > 0:  # Need to buy
            for price, volume in sorted(order_depth.sell_orders.items()):
                # Don't exceed quantity needed
                executable_volume = min(abs(quantity_needed), abs(volume))
                if executable_volume > 0:
                    orders.append(Order(product, price, executable_volume))
                    quantity_needed -= executable_volume
                    if quantity_needed <= 0:
                        break
        else:  # Need to sell
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
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
    
    def estimate_round(self, timestamp):
        """Estimate current round based on timestamp"""
        # This is a heuristic - adjust based on how timestamps actually change between rounds
        # Assuming timestamps increment in a pattern that allows us to infer the round
        if not hasattr(self, 'first_timestamp'):
            self.first_timestamp = timestamp
            return 1
        
        # Rough estimate of round based on timestamp difference
        # This will need to be calibrated based on actual timestamp pattern
        time_diff = timestamp - self.first_timestamp
        estimated_round = 1 + time_diff // 1000  # Adjust divisor based on actual timestamps
        
        return max(1, min(7, int(estimated_round)))  # Cap between 1 and 7
    
    def run(self, state: TradingState):
        """Main trading method"""
        # Initialize trader data if needed
        if not state.traderData:
            state.traderData = jsonpickle.encode({"round": 1})
        else:
            try:
                # Parse trader data
                trader_data = jsonpickle.decode(state.traderData)
                if isinstance(trader_data, dict) and "round" in trader_data:
                    self.current_round = trader_data["round"]
            except:
                self.current_round = 1
        
        # Update round counter if new day detected
        if hasattr(self, 'last_timestamp'):
            if state.timestamp - self.last_timestamp > 5000:  # Significant jump in timestamp
                self.current_round += 1
        
        self.last_timestamp = state.timestamp
        
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
            elif product.startswith('VOLCANIC_ROCK_VOUCHER'):
                # Use options pricing strategy for volcanic rock vouchers
                orders = self.trade_volcanic_rock_vouchers(product, order_depth, state, mid_price, theoretical_voucher_prices)
            elif product == 'VOLCANIC_ROCK':
                # Use delta hedging strategy for volcanic rock
                orders = self.trade_volcanic_rock(product, order_depth, state, mid_price)
            elif product == 'SQUID_INK':
                # Use mean reversion strategy for SQUID_INK
                orders = self.trade_based_on_moving_average(product, order_depth, state, mid_price)
            else:
                # Use standard moving average strategy for other products
                orders = self.trade_based_on_moving_average(product, order_depth, state, mid_price)
            
            result[product] = orders
        
        # Always return 0 conversions since we're not converting products
        conversions = 0
        
        # Save state data with updated round
        traderData = jsonpickle.encode({"round": self.current_round})
        
        return result, conversions, traderData