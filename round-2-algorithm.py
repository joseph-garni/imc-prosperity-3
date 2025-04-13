import numpy as np
import math
import pandas as pd
import statistics
import jsonpickle
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple


class Trader:
    def sigmoid(self, x):
        """Sigmoid activation function with overflow protection"""
        if x < -100:
            return 0
        elif x > 100:
            return 1
        return 1 / (1 + math.exp(-x))
    
    def tanh(self, x):
        """Tanh activation function with overflow protection"""
        if x < -100:
            return -1
        elif x > 100:
            return 1
        return (math.exp(x) - math.exp(-x)) / (math.exp(x) + math.exp(-x))
    
    def lstm_step(self, product, inputs, trader_data):
        """
        Perform one LSTM step for prediction
        
        Args:
            product: The product to predict for
            inputs: Input features [price, momentum, ma_ratio, volatility]
            trader_data: Trading state data
            
        Returns:
            LSTM prediction
        """
        # Get previous states
        h_prev = trader_data["lstm_hidden"][product]
        c_prev = trader_data["lstm_cells"][product]
        
        # Full input vector (concatenated)
        x_combined = np.concatenate(([1], inputs))  # Adding bias term
        
        # Get weights
        w = trader_data["lstm_weights"][product]
        
        # Input gate
        i_t = np.zeros(4)
        for i in range(4):
            i_t[i] = self.sigmoid(np.dot(w['input_gate'][i], x_combined))
        
        # Forget gate
        f_t = np.zeros(4)
        for i in range(4):
            f_t[i] = self.sigmoid(np.dot(w['forget_gate'][i], x_combined))
        
        # Output gate
        o_t = np.zeros(4)
        for i in range(4):
            o_t[i] = self.sigmoid(np.dot(w['output_gate'][i], x_combined))
        
        # Cell candidate
        g_t = np.zeros(4)
        for i in range(4):
            g_t[i] = self.tanh(np.dot(w['candidate'][i], x_combined))
        
        # Cell state update
        c_t = f_t * c_prev + i_t * g_t
        
        # Hidden state update
        h_t = np.zeros(4)
        for i in range(4):
            h_t[i] = o_t[i] * self.tanh(c_t[i])
        
        # Update states in trader_data
        trader_data["lstm_hidden"][product] = h_t
        trader_data["lstm_cells"][product] = c_t
        
        return h_t
    
    def calculate_fair_value(self, product, trader_data, positions):
        """Calculate fair value of a product using LSTM and historical data"""
        price_history = trader_data["price_history"].get(product, [])
        
        if len(price_history) < 5:
            # Not enough history, use last price or default
            return price_history[-1] if price_history else 100.0
        
        # Extract features for LSTM
        recent_prices = price_history[-5:]
        
        # Price momentum (percent change)
        price_momentum = (recent_prices[-1] / recent_prices[0] - 1) if recent_prices[0] > 0 else 0
        
        # Short and long term moving averages
        ma_short = statistics.mean(recent_prices[-3:]) if len(recent_prices) >= 3 else recent_prices[-1]
        ma_long = statistics.mean(recent_prices) if recent_prices else recent_prices[-1]
        ma_ratio = ma_short / ma_long if ma_long > 0 else 1.0
        
        # Volatility
        if len(recent_prices) >= 3:
            volatility = statistics.stdev(recent_prices) / statistics.mean(recent_prices) if statistics.mean(recent_prices) > 0 else 0
        else:
            volatility = 0.0
        
        # LSTM input features
        lstm_input = np.array([recent_prices[-1], price_momentum, ma_ratio, volatility])
        
        # Get LSTM prediction
        lstm_output = self.lstm_step(product, lstm_input, trader_data)
        
        # Combine LSTM output with recent price for fair value estimate
        # Using a weighted average of LSTM prediction and recent price
        lstm_weight = 0.3  # 30% weight to LSTM prediction
        fair_value = (1 - lstm_weight) * recent_prices[-1] + lstm_weight * (recent_prices[-1] * (1 + lstm_output[0] * 0.01))
        
        return fair_value
    
    def calculate_squid_ink_signal(self, trader_data):
        """
        Calculate trading signal for squid ink based on mean reversion
        
        Returns:
            Tuple of (signal direction, confidence)
            where signal is -1 (sell), 0 (neutral), or 1 (buy)
        """
        # Check if we have enough data
        if len(trader_data["squid_ink_short_ma"]) < 10 or len(trader_data["squid_ink_long_ma"]) < 30:
            return 0, 0  # Not enough data
        
        # Get current price
        current_price = trader_data["price_history"]["SQUID_INK"][-1]
        
        # Calculate moving averages
        short_ma = statistics.mean(trader_data["squid_ink_short_ma"])
        long_ma = statistics.mean(trader_data["squid_ink_long_ma"])
        
        # Reversion threshold - how many standard deviations to trigger signal
        reversion_threshold = 2.0
        
        # Calculate z-score of current price vs long MA
        if trader_data["squid_ink_volatility"]:
            recent_volatility = trader_data["squid_ink_volatility"][-1]
            if recent_volatility > 0:
                z_score = (current_price - long_ma) / (long_ma * recent_volatility)
                
                # Generate signal based on z-score
                if z_score < -reversion_threshold:
                    # Price is significantly below average - buy signal
                    confidence = min(abs(z_score) / 4, 1.0)  # Cap at 1.0
                    return 1, confidence
                elif z_score > reversion_threshold:
                    # Price is significantly above average - sell signal
                    confidence = min(abs(z_score) / 4, 1.0)  # Cap at 1.0
                    return -1, confidence
        
        # Check short vs long moving average crossover
        if short_ma > long_ma * 1.05:
            # Short MA significantly above long MA - bullish momentum
            return 1, 0.3
        elif short_ma < long_ma * 0.95:
            # Short MA significantly below long MA - bearish momentum
            return -1, 0.3
        
        return 0, 0  # Neutral
    
    def handle_basket_arbitrage(self, basket_name, trader_data, order_depths, current_positions):
        """
        Handle basket arbitrage opportunities
        
        Returns:
            List of orders for arbitrage
        """
        # Check if the basket is in our tracked baskets
        if basket_name not in trader_data["basket_compositions"]:
            return []
        
        # Get components of this basket
        basket_composition = trader_data["basket_compositions"][basket_name]
        
        # Check if we have all required components in market data
        for component in basket_composition:
            if component not in order_depths:
                return []  # Missing component data
        
        # Get basket order depth
        basket_depth = order_depths[basket_name]
        if not basket_depth.buy_orders or not basket_depth.sell_orders:
            return []  # No liquidity in basket
        
        # Calculate theoretical basket value from components
        theoretical_value = 0
        component_prices = {}
        component_liquidity = {}
        
        for component, quantity in basket_composition.items():
            component_depth = order_depths[component]
            
            # Get best bid and ask for component
            if component_depth.buy_orders and component_depth.sell_orders:
                best_bid = max(component_depth.buy_orders.keys())
                best_ask = min(component_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                component_prices[component] = mid_price
                
                # Track available liquidity
                component_liquidity[component] = {
                    "bid": (best_bid, sum(component_depth.buy_orders.values())),  # Price, volume
                    "ask": (best_ask, -sum(component_depth.sell_orders.values()))  # Price, volume (positive)
                }
                
                theoretical_value += mid_price * quantity
            else:
                return []  # Missing liquidity in component
        
        # Get basket prices
        best_basket_bid = max(basket_depth.buy_orders.keys())
        best_basket_ask = min(basket_depth.sell_orders.keys())
        basket_bid_volume = basket_depth.buy_orders[best_basket_bid]
        basket_ask_volume = -basket_depth.sell_orders[best_basket_ask]  # Convert to positive
        
        # Check for arbitrage opportunities
        arb_threshold = 0.01  # 1% price difference
        
        # List to store arbitrage orders
        arb_orders = []
        
        # CASE 1: Basket price > sum of components
        # Strategy: Sell basket, buy components
        if best_basket_bid > theoretical_value * (1 + arb_threshold):
            # Calculate max quantity we can trade
            max_basket_sell = min(
                basket_bid_volume,
                trader_data["position_limits"][basket_name] + current_positions.get(basket_name, 0)
            )
            
            # Check if we can buy all required components
            max_component_buys = []
            for component, qty_per_basket in basket_composition.items():
                required_qty = qty_per_basket * max_basket_sell
                max_buyable = min(
                    component_liquidity[component]["ask"][1],  # Available ask volume
                    trader_data["position_limits"][component] - current_positions.get(component, 0)  # Position limit
                )
                max_component_buys.append(math.floor(max_buyable / qty_per_basket))
            
            # Take the minimum of all constraints
            max_trade_size = min(max_basket_sell, *max_component_buys)
            
            if max_trade_size > 0:
                # Sell the basket
                arb_orders.append(Order(basket_name, best_basket_bid, -max_trade_size))
                
                # Buy the components
                for component, qty_per_basket in basket_composition.items():
                    buy_qty = qty_per_basket * max_trade_size
                    buy_price = component_liquidity[component]["ask"][0]  # Best ask price
                    arb_orders.append(Order(component, buy_price, buy_qty))
        
        # CASE 2: Basket price < sum of components
        # Strategy: Buy basket, sell components
        elif best_basket_ask < theoretical_value * (1 - arb_threshold):
            # Calculate max quantity we can trade
            max_basket_buy = min(
                basket_ask_volume,
                trader_data["position_limits"][basket_name] - current_positions.get(basket_name, 0)
            )
            
            # Check if we can sell all required components
            max_component_sells = []
            for component, qty_per_basket in basket_composition.items():
                required_qty = qty_per_basket * max_basket_buy
                max_sellable = min(
                    component_liquidity[component]["bid"][1],  # Available bid volume
                    trader_data["position_limits"][component] + current_positions.get(component, 0)  # Position limit
                )
                max_component_sells.append(math.floor(max_sellable / qty_per_basket))
            
            # Take the minimum of all constraints
            max_trade_size = min(max_basket_buy, *max_component_sells)
            
            if max_trade_size > 0:
                # Buy the basket
                arb_orders.append(Order(basket_name, best_basket_ask, max_trade_size))
                
                # Sell the components
                for component, qty_per_basket in basket_composition.items():
                    sell_qty = qty_per_basket * max_trade_size
                    sell_price = component_liquidity[component]["bid"][0]  # Best bid price
                    arb_orders.append(Order(component, sell_price, -sell_qty))
        
        return arb_orders
    
    def run(self, state: TradingState):
        """
        Main trading logic to be called on each market update
        
        Args:
            state: The current market state containing order depths, positions, etc.
                
        Returns:
            Dict of orders to place, conversions, and trader data
        """
        # Initialize trader_data - either load existing or create new
        if state.traderData == "":
            trader_data = {
                # Position limits for all products
                "position_limits": {
                    "CROISSANTS": 250,
                    "JAMS": 350,
                    "DJEMBES": 60,
                    "PICNIC_BASKET1": 60,
                    "PICNIC_BASKET2": 100,
                    "RAINFOREST_RESIN": 50,
                    "KELP": 50,
                    "SQUID_INK": 50
                },
                # Basket compositions
                "basket_compositions": {
                    "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
                    "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
                },
                # Price history - store last 100 mid prices for each product
                "price_history": {},
                # Specific for SQUID_INK mean reversion strategy
                "squid_ink_short_ma": [],
                "squid_ink_long_ma": [],
                "squid_ink_volatility": [],
                # LSTM components (simplified)
                "lstm_cells": {},
                "lstm_hidden": {},
                # Weights for the LSTM model
                "lstm_weights": {},
                # Track previous observations
                "time_steps": 0
            }
            
            # Initialize price history for all products
            for product in ["CROISSANTS", "JAMS", "DJEMBES", "PICNIC_BASKET1", 
                          "PICNIC_BASKET2", "RAINFOREST_RESIN", "KELP", "SQUID_INK"]:
                trader_data["price_history"][product] = []
                trader_data["lstm_cells"][product] = np.zeros(4)
                trader_data["lstm_hidden"][product] = np.zeros(4)
            
            # Initialize LSTM weights (simplified)
            np.random.seed(42)  # For reproducibility
            for product in trader_data["position_limits"].keys():
                trader_data["lstm_weights"][product] = {
                    'input_gate': np.random.normal(0, 0.1, (4, 5)),
                    'forget_gate': np.random.normal(0, 0.1, (4, 5)),
                    'output_gate': np.random.normal(0, 0.1, (4, 5)),
                    'candidate': np.random.normal(0, 0.1, (4, 5)),
                    'hidden': np.random.normal(0, 0.1, 4)
                }
        else:
            # Load existing trader data
            trader_data = jsonpickle.decode(state.traderData)
        
        # Increment time step
        trader_data["time_steps"] += 1
        
        # Update price history with current market data
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            
            # Calculate mid price if possible
            if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
            elif len(order_depth.buy_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                mid_price = best_bid
            elif len(order_depth.sell_orders) > 0:
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = best_ask
            else:
                # No orders, use last price if available
                if product in trader_data["price_history"] and len(trader_data["price_history"][product]) > 0:
                    mid_price = trader_data["price_history"][product][-1]
                else:
                    mid_price = 100  # Default starting price
            
            # Update price history (keep last 100 prices)
            if product in trader_data["price_history"]:
                trader_data["price_history"][product].append(mid_price)
                if len(trader_data["price_history"][product]) > 100:
                    trader_data["price_history"][product] = trader_data["price_history"][product][-100:]
            
            # Update SQUID_INK specific data
            if product == "SQUID_INK":
                # Update short moving average (10 periods)
                trader_data["squid_ink_short_ma"].append(mid_price)
                if len(trader_data["squid_ink_short_ma"]) > 10:
                    trader_data["squid_ink_short_ma"] = trader_data["squid_ink_short_ma"][-10:]
                
                # Update long moving average (30 periods)
                trader_data["squid_ink_long_ma"].append(mid_price)
                if len(trader_data["squid_ink_long_ma"]) > 30:
                    trader_data["squid_ink_long_ma"] = trader_data["squid_ink_long_ma"][-30:]
                
                # Calculate volatility if enough data
                if len(trader_data["price_history"][product]) >= 10:
                    recent_prices = trader_data["price_history"][product][-10:]
                    volatility = statistics.stdev(recent_prices) / statistics.mean(recent_prices) if statistics.mean(recent_prices) > 0 else 0
                    trader_data["squid_ink_volatility"].append(volatility)
                    if len(trader_data["squid_ink_volatility"]) > 30:
                        trader_data["squid_ink_volatility"] = trader_data["squid_ink_volatility"][-30:]
        
        # Initialize result dict for all products
        result = {}
        
        # Track current positions for position limit checks
        current_positions = {product: 0 for product in trader_data["position_limits"]}
        for product, position in state.position.items():
            if product in current_positions:
                current_positions[product] = position
        
        # Process each product
        for product in state.order_depths:
            order_depth = state.order_depths[product]
            orders = []
            
            # Get product fair value using LSTM
            fair_value = self.calculate_fair_value(product, trader_data, current_positions)
            
            # Special handling for SQUID_INK - mean reversion strategy
            if product == "SQUID_INK":
                signal, confidence = self.calculate_squid_ink_signal(trader_data)
                
                # Buy signal
                if signal > 0 and len(order_depth.sell_orders) > 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_volume = order_depth.sell_orders[best_ask]
                    
                    # Only buy if price is below our fair value
                    if best_ask < fair_value * 0.99:
                        # Calculate quantity based on confidence and position limits
                        max_position = trader_data["position_limits"][product]
                        available_position = max_position - current_positions[product]
                        target_quantity = int(available_position * confidence)
                        if target_quantity > 0:
                            quantity = min(target_quantity, -best_ask_volume)  # Best ask volume is negative
                            if quantity > 0:
                                orders.append(Order(product, best_ask, quantity))
                                current_positions[product] += quantity
                
                # Sell signal
                elif signal < 0 and len(order_depth.buy_orders) > 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_volume = order_depth.buy_orders[best_bid]
                    
                    # Only sell if price is above our fair value
                    if best_bid > fair_value * 1.01:
                        # Calculate quantity based on confidence and position limits
                        max_position = trader_data["position_limits"][product]
                        available_position = max_position + current_positions[product]  # For selling
                        target_quantity = int(available_position * confidence)
                        if target_quantity > 0:
                            quantity = min(target_quantity, best_bid_volume)
                            if quantity > 0:
                                orders.append(Order(product, best_bid, -quantity))
                                current_positions[product] -= quantity
            
            # Handle basket arbitrage for PICNIC_BASKET1 and PICNIC_BASKET2
            elif "PICNIC_BASKET" in product:
                # Check for arbitrage opportunities
                arb_orders = self.handle_basket_arbitrage(
                    product, 
                    trader_data, 
                    state.order_depths, 
                    current_positions
                )
                orders.extend(arb_orders)
                
                # Update positions based on arbitrage orders
                for order in arb_orders:
                    current_positions[order.symbol] += order.quantity
            
            # Regular trading logic for other products
            else:
                # Determine acceptable price ranges based on fair value
                buy_threshold = fair_value * 0.98
                sell_threshold = fair_value * 1.02
                
                # Buy orders - if ask price is below our buy threshold
                if len(order_depth.sell_orders) > 0:
                    for ask_price, ask_volume in sorted(order_depth.sell_orders.items()):
                        if ask_price <= buy_threshold:
                            max_position = trader_data["position_limits"][product]
                            available_position = max_position - current_positions[product]
                            
                            if available_position > 0:
                                # Calculate signal strength based on price difference
                                signal_strength = (buy_threshold - ask_price) / buy_threshold
                                target_quantity = min(int(available_position * signal_strength * 5), -ask_volume)
                                
                                if target_quantity > 0:
                                    orders.append(Order(product, ask_price, target_quantity))
                                    current_positions[product] += target_quantity
                
                # Sell orders - if bid price is above our sell threshold
                if len(order_depth.buy_orders) > 0:
                    for bid_price, bid_volume in sorted(order_depth.buy_orders.items(), reverse=True):
                        if bid_price >= sell_threshold:
                            max_position = trader_data["position_limits"][product]
                            available_position = max_position + current_positions[product]  # For selling
                            
                            if available_position > 0:
                                # Calculate signal strength based on price difference
                                signal_strength = (bid_price - sell_threshold) / sell_threshold
                                target_quantity = min(int(available_position * signal_strength * 5), bid_volume)
                                
                                if target_quantity > 0:
                                    orders.append(Order(product, bid_price, -target_quantity))
                                    current_positions[product] -= target_quantity
            
            result[product] = orders
        
        # Serialize trader_data for next iteration
        traderData = jsonpickle.encode(trader_data)
        
        # No conversions in this implementation
        conversions = 0
        
        return result, conversions, traderData