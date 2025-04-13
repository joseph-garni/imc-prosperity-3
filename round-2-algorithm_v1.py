import numpy as np
import math
import pandas as pd
import statistics
import jsonpickle
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple

class Trader:
    def __init__(self):
        # Position limits
        self.position_limits = {
            "CROISSANTS": 250,
            "JAMS": 350,
            "DJEMBES": 60,
            "PICNIC_BASKET1": 60,
            "PICNIC_BASKET2": 100,
            "RAINFOREST_RESIN": 50,
            "KELP": 50,
            "SQUID_INK": 50
        }
        
        # Basket compositions
        self.basket_compositions = {
            "PICNIC_BASKET1": {"CROISSANTS": 6, "JAMS": 3, "DJEMBES": 1},
            "PICNIC_BASKET2": {"CROISSANTS": 4, "JAMS": 2}
        }
        
        # Price history for products (store last 50 prices)
        self.price_history = {}
        
        # Trade history (to track our performance)
        self.trade_history = {}
        
        # For SQUID_INK trend following
        self.squid_short_ma = []  # 5-period MA
        self.squid_long_ma = []   # 20-period MA
    
    def run(self, state: TradingState):
        # Initialize result dictionary
        result = {}
        
        # Load state if available
        if state.traderData != "":
            try:
                saved_state = jsonpickle.decode(state.traderData)
                self.price_history = saved_state.price_history
                self.trade_history = saved_state.trade_history
                self.squid_short_ma = saved_state.squid_short_ma
                self.squid_long_ma = saved_state.squid_long_ma
            except:
                # Reset if deserialization fails
                self.price_history = {}
                self.trade_history = {}
                self.squid_short_ma = []
                self.squid_long_ma = []
                print("Failed to load trader state, resetting")
        
        # Get current positions
        positions = {product: 0 for product in self.position_limits}
        for product, position in state.position.items():
            if product in positions:
                positions[product] = position
        
        # Update price history for all products
        for product, order_depth in state.order_depths.items():
            if product not in self.price_history:
                self.price_history[product] = []
            
            # Calculate mid price if possible
            if len(order_depth.buy_orders) > 0 and len(order_depth.sell_orders) > 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                mid_price = (best_bid + best_ask) / 2
                self.price_history[product].append(mid_price)
                
                # Keep only recent prices
                if len(self.price_history[product]) > 50:
                    self.price_history[product] = self.price_history[product][-50:]
        
        # Update SQUID_INK moving averages
        if "SQUID_INK" in self.price_history and len(self.price_history["SQUID_INK"]) >= 5:
            # Short MA (5 periods)
            self.squid_short_ma.append(sum(self.price_history["SQUID_INK"][-5:]) / 5)
            if len(self.squid_short_ma) > 30:
                self.squid_short_ma = self.squid_short_ma[-30:]
            
            # Long MA (20 periods)
            if len(self.price_history["SQUID_INK"]) >= 20:
                self.squid_long_ma.append(sum(self.price_history["SQUID_INK"][-20:]) / 20)
                if len(self.squid_long_ma) > 30:
                    self.squid_long_ma = self.squid_long_ma[-30:]
        
        # Initialize orders with emergency position unwinding logic
        # This ensures we don't get trapped in bad positions
        for product, pos in positions.items():
            if product not in result:
                result[product] = []
            
            # If we have a large position, try to reduce it
            if abs(pos) > self.position_limits[product] * 0.8:  # If >80% of limit
                if product in state.order_depths:
                    order_depth = state.order_depths[product]
                    
                    # If we have a large positive position, try to sell
                    if pos > 0 and order_depth.buy_orders:
                        best_bid = max(order_depth.buy_orders.keys())
                        unwind_size = min(pos // 2, order_depth.buy_orders[best_bid])  # Unwind half
                        if unwind_size > 0:
                            result[product].append(Order(product, best_bid, -unwind_size))
                            positions[product] -= unwind_size
                    
                    # If we have a large negative position, try to buy
                    elif pos < 0 and order_depth.sell_orders:
                        best_ask = min(order_depth.sell_orders.keys())
                        unwind_size = min(abs(pos) // 2, -order_depth.sell_orders[best_ask])  # Unwind half
                        if unwind_size > 0:
                            result[product].append(Order(product, best_ask, unwind_size))
                            positions[product] += unwind_size
        
        # Process each product with specific strategies
        for product, order_depth in state.order_depths.items():
            if product not in result:
                result[product] = []
            
            # Skip if product not in our list
            if product not in self.position_limits:
                continue
            
            # Only trade if we have both bids and asks
            if order_depth.buy_orders and order_depth.sell_orders:
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                spread = best_ask - best_bid
                spread_pct = spread / ((best_bid + best_ask) / 2)
                
                # STRATEGY 1: SQUID_INK Trend Following + Mean Reversion
                if product == "SQUID_INK" and len(self.squid_short_ma) >= 3 and len(self.squid_long_ma) >= 3:
                    # Calculate trend direction using moving averages
                    short_ma = self.squid_short_ma[-1]
                    long_ma = self.squid_long_ma[-1]
                    
                    # Get previous MAs to detect crossovers
                    prev_short_ma = self.squid_short_ma[-2]
                    prev_long_ma = self.squid_long_ma[-2]
                    
                    # Recent volatility to adjust position size
                    if len(self.price_history["SQUID_INK"]) >= 10:
                        volatility = statistics.stdev(self.price_history["SQUID_INK"][-10:]) / statistics.mean(self.price_history["SQUID_INK"][-10:])
                    else:
                        volatility = 0.01  # Default low volatility
                    
                    # Base position size - smaller in high volatility
                    base_size = max(2, min(10, int(5 / (volatility * 100 + 1))))
                    
                    # Trend-following signals
                    # Buy when short MA crosses above long MA (bullish crossover)
                    if prev_short_ma <= prev_long_ma and short_ma > long_ma:
                        # Only buy if we have capacity
                        if positions[product] < self.position_limits[product] - base_size:
                            # Check if we can execute
                            ask_volume = -order_depth.sell_orders[best_ask]
                            trade_size = min(base_size, ask_volume)
                            if trade_size > 0:
                                result[product].append(Order(product, best_ask, trade_size))
                                positions[product] += trade_size
                    
                    # Sell when short MA crosses below long MA (bearish crossover)
                    elif prev_short_ma >= prev_long_ma and short_ma < long_ma:
                        # Only sell if we have capacity
                        if positions[product] > -self.position_limits[product] + base_size:
                            # Check if we can execute
                            bid_volume = order_depth.buy_orders[best_bid]
                            trade_size = min(base_size, bid_volume)
                            if trade_size > 0:
                                result[product].append(Order(product, best_bid, -trade_size))
                                positions[product] -= trade_size
                    
                    # Mean reversion when price deviates significantly from longer MA
                    current_price = (best_bid + best_ask) / 2
                    deviation = (current_price - long_ma) / long_ma if long_ma > 0 else 0
                    
                    # If price is significantly above long MA, sell (mean reversion)
                    if deviation > 0.05 and best_bid > long_ma * 1.05:
                        reversion_size = max(1, min(5, int(abs(deviation) * 50)))
                        
                        # Only sell if we have capacity
                        if positions[product] > -self.position_limits[product] + reversion_size:
                            # Check if we can execute
                            bid_volume = order_depth.buy_orders[best_bid]
                            trade_size = min(reversion_size, bid_volume)
                            if trade_size > 0:
                                result[product].append(Order(product, best_bid, -trade_size))
                                positions[product] -= trade_size
                    
                    # If price is significantly below long MA, buy (mean reversion)
                    elif deviation < -0.05 and best_ask < long_ma * 0.95:
                        reversion_size = max(1, min(5, int(abs(deviation) * 50)))
                        
                        # Only buy if we have capacity
                        if positions[product] < self.position_limits[product] - reversion_size:
                            # Check if we can execute
                            ask_volume = -order_depth.sell_orders[best_ask]
                            trade_size = min(reversion_size, ask_volume)
                            if trade_size > 0:
                                result[product].append(Order(product, best_ask, trade_size))
                                positions[product] += trade_size
                
                # STRATEGY 2: Basket Arbitrage (with improved safety)
                elif product.startswith("PICNIC_BASKET"):
                    # Calculate theoretical basket value more carefully
                    basket_value = self.calculate_basket_value(product, state.order_depths)
                    
                    if basket_value > 0:  # Only proceed if we could calculate a value
                        # Use 5% threshold for safer arbitrage
                        arb_threshold = 0.05
                        
                        # Small fixed size for arbitrage
                        arb_size = 1
                        
                        # Case 1: Sell basket, buy components
                        if best_bid > basket_value * (1 + arb_threshold):
                            # Check position limits
                            if positions[product] > -self.position_limits[product] + arb_size:
                                # Verify component trades are possible
                                can_execute = True
                                component_orders = []
                                
                                for component, quantity in self.basket_compositions[product].items():
                                    if component not in state.order_depths:
                                        can_execute = False
                                        break
                                    
                                    component_depth = state.order_depths[component]
                                    if not component_depth.sell_orders:
                                        can_execute = False
                                        break
                                    
                                    # Get best ask
                                    component_ask = min(component_depth.sell_orders.keys())
                                    ask_volume = -component_depth.sell_orders[component_ask]
                                    
                                    # Check if enough volume
                                    needed_qty = quantity * arb_size
                                    if ask_volume < needed_qty:
                                        can_execute = False
                                        break
                                    
                                    # Check position limit
                                    new_position = positions.get(component, 0) + needed_qty
                                    if new_position > self.position_limits[component]:
                                        can_execute = False
                                        break
                                    
                                    component_orders.append((component, component_ask, needed_qty))
                                
                                if can_execute:
                                    # Sell basket first
                                    result[product].append(Order(product, best_bid, -arb_size))
                                    positions[product] -= arb_size
                                    
                                    # Buy components
                                    for comp, price, qty in component_orders:
                                        if comp not in result:
                                            result[comp] = []
                                        
                                        result[comp].append(Order(comp, price, qty))
                                        positions[comp] = positions.get(comp, 0) + qty
                        
                        # Case 2: Buy basket, sell components
                        elif best_ask < basket_value * (1 - arb_threshold):
                            # Check position limits
                            if positions[product] < self.position_limits[product] - arb_size:
                                # Verify component trades are possible
                                can_execute = True
                                component_orders = []
                                
                                for component, quantity in self.basket_compositions[product].items():
                                    if component not in state.order_depths:
                                        can_execute = False
                                        break
                                    
                                    component_depth = state.order_depths[component]
                                    if not component_depth.buy_orders:
                                        can_execute = False
                                        break
                                    
                                    # Get best bid
                                    component_bid = max(component_depth.buy_orders.keys())
                                    bid_volume = component_depth.buy_orders[component_bid]
                                    
                                    # Check if enough volume
                                    needed_qty = quantity * arb_size
                                    if bid_volume < needed_qty:
                                        can_execute = False
                                        break
                                    
                                    # Check position limit
                                    new_position = positions.get(component, 0) - needed_qty
                                    if new_position < -self.position_limits[component]:
                                        can_execute = False
                                        break
                                    
                                    component_orders.append((component, component_bid, needed_qty))
                                
                                if can_execute:
                                    # Buy basket first
                                    result[product].append(Order(product, best_ask, arb_size))
                                    positions[product] += arb_size
                                    
                                    # Sell components
                                    for comp, price, qty in component_orders:
                                        if comp not in result:
                                            result[comp] = []
                                        
                                        result[comp].append(Order(comp, price, -qty))
                                        positions[comp] = positions.get(comp, 0) - qty
                
                # STRATEGY 3: Selective Market Making (only when spread is large)
                else:
                    # Only do market making if spread is sufficiently large
                    if spread_pct > 0.01:  # >1% spread
                        # Market making size based on spread
                        mm_size = max(1, min(5, int(spread_pct * 100)))
                        
                        # Buy at best ask if spread is attractive and position allows
                        if positions[product] < self.position_limits[product] - mm_size:
                            ask_volume = -order_depth.sell_orders[best_ask]
                            trade_size = min(mm_size, ask_volume)
                            if trade_size > 0:
                                result[product].append(Order(product, best_ask, trade_size))
                                positions[product] += trade_size
                        
                        # Sell at best bid if spread is attractive and position allows
                        if positions[product] > -self.position_limits[product] + mm_size:
                            bid_volume = order_depth.buy_orders[best_bid]
                            trade_size = min(mm_size, bid_volume)
                            if trade_size > 0:
                                result[product].append(Order(product, best_bid, -trade_size))
                                positions[product] -= trade_size
        
        # No conversions
        conversions = 0
        
        # Save state
        traderData = jsonpickle.encode(self)
        
        return result, conversions, traderData
    
    def calculate_basket_value(self, basket_name, order_depths):
        """Calculate theoretical value of a basket from component prices with safety"""
        if basket_name not in self.basket_compositions:
            return 0
        
        total_value = 0
        components_found = 0
        
        for component, quantity in self.basket_compositions[basket_name].items():
            if component not in order_depths:
                continue
            
            component_depth = order_depths[component]
            if not component_depth.buy_orders or not component_depth.sell_orders:
                continue
            
            # Use mid price
            component_bid = max(component_depth.buy_orders.keys())
            component_ask = min(component_depth.sell_orders.keys())
            component_mid = (component_bid + component_ask) / 2
            
            total_value += component_mid * quantity
            components_found += 1
        
        # Only return a value if we found all components
        if components_found == len(self.basket_compositions[basket_name]):
            return total_value
        
        return 0