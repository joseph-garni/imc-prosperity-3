
    """
Replace these methods in your existing Trader class to implement 
the improved volcanic rock trading strategy with better risk management.
"""

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
    
    # Volcanic rock specific variables
    self.volcanic_rock_pnl_history = []
    self.volcanic_rock_recent_pnl = []  # Keep track of recent P&L for stop-loss
    self.max_consecutive_losses = 0     # Track consecutive losses for risk adjustment
    self.volcanic_position_change = 0   # Track position change magnitude
    self.last_volcanic_position = 0     # Track last position
    
    self.volcanic_voucher_pnl_history = {
        'VOLCANIC_ROCK_VOUCHER_9500': [],
        'VOLCANIC_ROCK_VOUCHER_9750': [],
        'VOLCANIC_ROCK_VOUCHER_10000': [],
        'VOLCANIC_ROCK_VOUCHER_10250': [],
        'VOLCANIC_ROCK_VOUCHER_10500': []
    }
    self.price_diff_history = {
        'VOLCANIC_ROCK_VOUCHER_9500': [],
        'VOLCANIC_ROCK_VOUCHER_9750': [],
        'VOLCANIC_ROCK_VOUCHER_10000': [],
        'VOLCANIC_ROCK_VOUCHER_10250': [],
        'VOLCANIC_ROCK_VOUCHER_10500': []
    }
    self.theoretical_fair_values = {
        'VOLCANIC_ROCK_VOUCHER_9500': [],
        'VOLCANIC_ROCK_VOUCHER_9750': [],
        'VOLCANIC_ROCK_VOUCHER_10000': [],
        'VOLCANIC_ROCK_VOUCHER_10250': [],
        'VOLCANIC_ROCK_VOUCHER_10500': []
    }
    self.max_history_size = 500
    self.position_history = {}
    
    # Risk management parameters
    self.risk_factors = {
        'VOLCANIC_ROCK': {
            'max_position_change_pct': 0.15,  # Maximum position change in one step (% of max position)
            'volatility_scaling': True,       # Adjust position sizes based on volatility
            'post_loss_scaling': 0.7,         # Scale down positions after losses
            'consecutive_loss_threshold': 3,   # How many consecutive losses before scaling down
            'stop_loss_level': -3000,         # P&L level to trigger stop loss
            'cool_down_period': 5,            # How many timestamps to wait after large losses
        }
    }
    
    # Transaction cost tracking
    self.transaction_costs_history = {}
    
    self.initialized = False

def get_volcanic_risk_factor(self):
    """Calculate a risk factor for volcanic rock trading based on historical volatility and recent PnL"""
    if 'VOLCANIC_ROCK' not in self.price_history or len(self.price_history['VOLCANIC_ROCK']['price']) < 20:
        return 0.7  # Default conservative factor
    
    prices = self.price_history['VOLCANIC_ROCK']['price'][-100:]
    if len(prices) < 2:
        return 0.7
    
    # Calculate returns
    returns = [prices[i]/prices[i-1] - 1 for i in range(1, len(prices))]
    
    # Calculate volatility
    vol = math.sqrt(sum(r*r for r in returns) / len(returns))
    
    # Base factor adjusted for volatility
    base_factor = 0.8
    vol_adjustment = max(-0.4, min(0.2, 0.1 - vol*10))
    risk_factor = base_factor + vol_adjustment
    
    # Adjust for recent PnL performance
    if hasattr(self, 'volcanic_rock_recent_pnl') and len(self.volcanic_rock_recent_pnl) > 0:
        recent_pnl_sum = sum(self.volcanic_rock_recent_pnl[-3:]) if len(self.volcanic_rock_recent_pnl) >= 3 else 0
        
        # If recent large losses, reduce risk factor
        if recent_pnl_sum < -5000:
            risk_factor *= 0.6  # Significantly reduce risk
        elif recent_pnl_sum < -2000:
            risk_factor *= 0.8  # Moderately reduce risk
    
    # If consecutive losses, further reduce risk
    if hasattr(self, 'max_consecutive_losses') and self.max_consecutive_losses >= 2:
        risk_factor *= (1.0 - min(0.5, self.max_consecutive_losses * 0.1))
    
    return max(0.3, min(1.0, risk_factor))  # Limit factor between 0.3 and 1.0

def simple_volcanic_rock_hedging(self, product, order_depth, state, mid_price):
    """Simple hedging strategy when we don't have enough historical data"""
    orders = []
    current_position = self.get_current_position(state, product)
    max_position = self.position_limits[product]
    
    # Calculate aggregate voucher position
    voucher_position = 0
    for voucher in self.voucher_details.keys():
        voucher_position += self.get_current_position(state, voucher)
    
    # Simple delta hedge ratio
    T = self.calculate_time_to_expiry()
    hedge_ratio = 1.0 if T <= 0 else 0.5
    
    # Target position for delta hedging
    target_position = -int(voucher_position * hedge_ratio)
    
    # Limit to position limits
    target_position = max(-max_position, min(max_position, target_position))
    
    # Apply more conservative limits in early trading
    max_position_change = int(max_position * 0.1)  # 10% of max position per step in early trading
    
    if target_position > current_position:
        target_position = min(target_position, current_position + max_position_change)
    else:
        target_position = max(target_position, current_position - max_position_change)
    
    # Calculate quantity needed
    quantity_needed = target_position - current_position
    
    # If no action needed
    if abs(quantity_needed) < 5:  # Minimum trade size
        return []
    
    # Execute orders with limited size
    if quantity_needed > 0:  # Buy
        quantity_needed = min(quantity_needed, 20)  # Limit order size when we lack data
        for price, volume in sorted(order_depth.sell_orders.items()):
            executable_volume = min(abs(volume), abs(quantity_needed))
            if executable_volume > 0:
                orders.append(Order(product, price, executable_volume))
                quantity_needed -= executable_volume
                if quantity_needed <= 0:
                    break
    else:  # Sell
        quantity_needed = max(quantity_needed, -20)  # Limit order size when we lack data
        for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
            executable_volume = min(abs(volume), abs(quantity_needed))
            if executable_volume > 0:
                orders.append(Order(product, price, -executable_volume))
                quantity_needed += executable_volume
                if quantity_needed >= 0:
                    break
    
    return orders

def trade_volcanic_rock(self, product, order_depth, state, mid_price):
    orders = []
    current_position = self.get_current_position(state, product)
    max_position = self.position_limits[product]
    
    # Update price history for statistical analysis
    if 'VOLCANIC_ROCK' in self.price_history:
        history_length = len(self.price_history['VOLCANIC_ROCK']['price'])
    else:
        history_length = 0
    
    # Track recent profit/loss for risk management
    if not hasattr(self, 'volcanic_rock_recent_pnl'):
        self.volcanic_rock_recent_pnl = []
    
    # Record transaction cost metrics
    if not hasattr(self, 'transaction_costs_history'):
        self.transaction_costs_history = {'buy': [], 'sell': []}
    
    # Use simple strategy if not enough history
    if history_length < 20:
        return self.simple_volcanic_rock_hedging(product, order_depth, state, mid_price)
    
    # Calculate aggregate voucher position
    voucher_position = 0
    voucher_value = 0
    total_vouchers = 0
    
    # Calculate voucher positions with weighting based on moneyness
    for voucher in self.voucher_details.keys():
        voucher_pos = self.get_current_position(state, voucher)
        if voucher_pos != 0:
            total_vouchers += abs(voucher_pos)
            # Extract strike price for weighted hedging
            strike = self.voucher_details[voucher]['strike']
            
            # Calculate moneyness-adjusted weight
            # Deep ITM options need more hedging, OTM need less
            if mid_price > 0 and strike > 0:
                moneyness = strike / mid_price
                
                # Weight by moneyness - ITM options (moneyness < 1) get higher weight
                weight = 1.0
                if moneyness < 0.95:  # Deep ITM
                    weight = 1.2
                elif moneyness > 1.05:  # OTM
                    weight = 0.8
                
                voucher_position += voucher_pos * weight
                voucher_value += voucher_pos * strike
    
    # Calculate delta hedge ratio with enhanced risk adjustments
    T = self.calculate_time_to_expiry()
    
    # Risk adjustment based on recent PnL - reduce exposure after losses
    risk_adjustment = 1.0
    if len(self.volcanic_rock_recent_pnl) > 0:
        recent_pnl_sum = sum(self.volcanic_rock_recent_pnl)
        if recent_pnl_sum < -5000:  # Large recent losses
            risk_adjustment = 0.7  # Significantly reduce hedging ratio
        elif recent_pnl_sum < -2000:  # Moderate recent losses
            risk_adjustment = 0.85  # Moderately reduce hedging ratio
        elif recent_pnl_sum > 5000:  # Large recent profits
            risk_adjustment = 1.1  # Slightly increase hedging aggression
    
    # Base hedge ratio calculation
    if T <= 0:
        base_hedge_ratio = 1.0  # At expiry, delta is 1 for ITM options
    else:
        # Calculate recent price volatility for dynamic hedging
        prices = self.price_history['VOLCANIC_ROCK']['price']
        recent_prices = prices[-20:]
        
        # Different hedging based on volatility
        if len(recent_prices) > 1:
            returns = [recent_prices[i]/recent_prices[i-1] - 1 for i in range(1, len(recent_prices))]
            vol = math.sqrt(sum(r*r for r in returns) / len(returns)) * math.sqrt(252)
            
            # More conservative hedging in high volatility
            base_delta = 0.65  # Slightly more conservative base delta
            vol_adjustment = max(-0.3, min(0.15, 0.25 - vol))  # More conservative in high vol
            base_hedge_ratio = base_delta + vol_adjustment
        else:
            base_hedge_ratio = 0.5  # Default
    
    # Apply risk adjustment
    hedge_ratio = base_hedge_ratio * risk_adjustment
    
    # Make hedge ratio relative to the size of position
    # Smaller positions need more complete hedging
    position_size_factor = min(1.0, abs(voucher_position) / 50)
    adjusted_hedge_ratio = hedge_ratio * (0.8 + 0.2 * position_size_factor)
    
    # Add a small dampening factor when current position is large
    # This prevents extreme position swings
    if abs(current_position) > max_position / 2:
        dampening = 0.9  # Reduce target position when already large
        adjusted_hedge_ratio *= dampening
    
    # Target position for delta hedging
    raw_target_position = -int(voucher_position * adjusted_hedge_ratio)
    
    # Check statistical validity of the target position
    recent_positions = []
    if hasattr(self, 'position_history') and 'VOLCANIC_ROCK' in self.position_history:
        recent_positions = self.position_history['VOLCANIC_ROCK'][-50:]
    
    # Implement position smoothing and risk mitigation
    target_position = raw_target_position
    
    # Calculate position statistics if we have history
    if recent_positions:
        avg_position = sum(recent_positions) / len(recent_positions)
        position_std = math.sqrt(sum((p - avg_position)**2 for p in recent_positions) / len(recent_positions))
        
        if position_std > 0:
            position_z_score = (raw_target_position - avg_position) / position_std
            
            # If target position is too extreme, moderate it more aggressively
            if abs(position_z_score) > 2.0:
                # Moderate the position change by moving only partially toward target
                # More extreme positions get more moderation
                moderation_factor = max(0.3, 0.7 - (abs(position_z_score) - 2.0) * 0.1)
                target_position = int(avg_position + (raw_target_position - avg_position) * moderation_factor)
                
                # Additional sanity check - limit maximum position change per step
                max_position_change = int(max_position * 0.15)  # Max 15% of position limit per step
                if abs(target_position - current_position) > max_position_change:
                    # Limit the step size to avoid jumps that cause large losses
                    if target_position > current_position:
                        target_position = current_position + max_position_change
                    else:
                        target_position = current_position - max_position_change
    
    # If no voucher positions, gradually reduce rock position to avoid unnecessary risk
    if total_vouchers == 0 and abs(current_position) > 0:
        reduction_rate = 0.25  # Reduce position by 25% each step
        target_position = int(current_position * (1 - reduction_rate))
        
        # If position is small enough, fully close it
        if abs(target_position) < 10:
            target_position = 0
    
    # Implement local minimum/maximum avoidance
    # Check if mid price is at a local extreme using recent history
    prices = self.price_history['VOLCANIC_ROCK']['price']
    if len(prices) >= 10:
        recent_prices = prices[-10:]
        min_price = min(recent_prices)
        max_price = max(recent_prices)
        
        # If price is near local minimum, be cautious about adding short positions
        if mid_price < min_price * 1.02 and target_position < current_position:
            # Only allow smaller position changes at local mins (avoid selling at bottom)
            max_change = int(abs(current_position) * 0.2)
            target_position = max(target_position, current_position - max_change)
        
        # If price is near local maximum, be cautious about adding long positions
        if mid_price > max_price * 0.98 and target_position > current_position:
            # Only allow smaller position changes at local maxs (avoid buying at top)
            max_change = int(abs(current_position) * 0.2)
            target_position = min(target_position, current_position + max_change)
    
    # Limit to position limits
    target_position = max(-max_position, min(max_position, target_position))
    
    # Calculate quantity needed
    quantity_needed = target_position - current_position
    
    # If change is very small, don't trade to avoid unnecessary transaction costs
    min_trade_size = 5
    if abs(quantity_needed) < min_trade_size:
        return []
    
    # Implement dynamic stop-loss logic
    should_stop_loss = False
    stop_loss_ratio = 0.0
    
    # Check for stop loss based on recent losses
    if len(self.volcanic_rock_recent_pnl) >= 3:
        last_three_pnl = self.volcanic_rock_recent_pnl[-3:]
        # If multiple consecutive losses, consider stop loss
        if all(pnl < -1000 for pnl in last_three_pnl):
            should_stop_loss = True
            stop_loss_ratio = 0.5  # Close half the position on severe drawdown
            
            # Adjust target and quantity
            if current_position > 0:
                target_position = int(current_position * (1 - stop_loss_ratio))
                quantity_needed = target_position - current_position
            elif current_position < 0:
                target_position = int(current_position * (1 - stop_loss_ratio))
                quantity_needed = target_position - current_position
    
    # Execute orders with liquidity awareness
    if quantity_needed > 0:  # Buy
        # Check liquidity and limit aggression if necessary
        if order_depth.sell_orders:
            # Analyze order book depth to avoid market impact
            total_sell_volume = sum(abs(vol) for vol in order_depth.sell_orders.values())
            buy_aggression = min(1.0, total_sell_volume / (abs(quantity_needed) * 2))
            
            # Limit buy volume based on historical volatility and liquidity
            limit_factor = min(buy_aggression, self.get_volcanic_risk_factor())
            max_buy_volume = int(max(min_trade_size, abs(quantity_needed) * limit_factor))
            
            for price, volume in sorted(order_depth.sell_orders.items()):
                executable_volume = min(abs(volume), max_buy_volume, abs(quantity_needed))
                if executable_volume > 0:
                    orders.append(Order(product, price, executable_volume))
                    current_position += executable_volume
                    quantity_needed -= executable_volume
                    max_buy_volume -= executable_volume
                    if quantity_needed <= 0 or max_buy_volume <= 0:
                        break
    else:  # Sell
        if order_depth.buy_orders:
            # Analyze order book depth to avoid market impact
            total_buy_volume = sum(abs(vol) for vol in order_depth.buy_orders.values())
            sell_aggression = min(1.0, total_buy_volume / (abs(quantity_needed) * 2))
            
            # Limit sell volume based on historical volatility and liquidity
            limit_factor = min(sell_aggression, self.get_volcanic_risk_factor())
            max_sell_volume = int(max(min_trade_size, abs(quantity_needed) * limit_factor))
            
            for price, volume in sorted(order_depth.buy_orders.items(), reverse=True):
                executable_volume = min(abs(volume), max_sell_volume, abs(quantity_needed))
                if executable_volume > 0:
                    orders.append(Order(product, price, -executable_volume))
                    current_position -= executable_volume
                    quantity_needed += executable_volume
                    max_sell_volume -= executable_volume
                    if quantity_needed >= 0 or max_sell_volume <= 0:
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
                
            # Load additional state data if available
            if isinstance(trader_data, dict) and "volcanic_rock_recent_pnl" in trader_data:
                self.volcanic_rock_recent_pnl = trader_data["volcanic_rock_recent_pnl"]
            if isinstance(trader_data, dict) and "max_consecutive_losses" in trader_data:
                self.max_consecutive_losses = trader_data["max_consecutive_losses"]
            if isinstance(trader_data, dict) and "last_volcanic_position" in trader_data:
                self.last_volcanic_position = trader_data["last_volcanic_position"]
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
        
        # Record current position for historical analysis
        current_position = self.get_current_position(state, product)
        if product not in self.position_history:
            self.position_history[product] = []
        self.position_history[product].append(current_position)
        if len(self.position_history[product]) > self.max_history_size:
            self.position_history[product].pop(0)
        
        # Track position changes for volcanic rock
        if product == 'VOLCANIC_ROCK':
            new_position = current_position
            if hasattr(self, 'last_volcanic_position'):
                position_change = new_position - self.last_volcanic_position
                self.volcanic_position_change = position_change
            self.last_volcanic_position = new_position
        
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
    
    # Get previous profit/loss information if available
    previous_pnl = {}
    if hasattr(state, 'previous_state') and state.previous_state is not None:
        if hasattr(state.previous_state, 'profit_and_loss'):
            previous_pnl = state.previous_state.profit_and_loss
    
    # Check for volcanic rock PnL and update records
    if 'VOLCANIC_ROCK' in previous_pnl:
        rock_pnl = previous_pnl['VOLCANIC_ROCK']
        
        # Update PnL history
        self.volcanic_rock_pnl_history.append(rock_pnl)
        if len(self.volcanic_rock_pnl_history) > self.max_history_size:
            self.volcanic_rock_pnl_history.pop(0)
        
        # Update recent PnL for risk management
        self.volcanic_rock_recent_pnl.append(rock_pnl)
        if len(self.volcanic_rock_recent_pnl) > 10:  # Only keep last 10 entries
            self.volcanic_rock_recent_pnl.pop(0)
        
        # Track consecutive losses
        if rock_pnl < -1000:  # Significant loss
            self.max_consecutive_losses += 1
        else:
            self.max_consecutive_losses = 0
    
    # Apply risk limits if consecutive losses exceed threshold
    volcanic_risk_limit = 1.0
    if hasattr(self, 'risk_factors') and 'VOLCANIC_ROCK' in self.risk_factors and self.max_consecutive_losses >= self.risk_factors['VOLCANIC_ROCK']['consecutive_loss_threshold']:
        volcanic_risk_limit = self.risk_factors['VOLCANIC_ROCK']['post_loss_scaling']
    
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
            # Use improved statistical options pricing strategy for volcanic rock vouchers
            mid_price = mid_prices[product]
            orders = self.trade_volcanic_rock_vouchers(product, order_depth, state, mid_price, theoretical_voucher_prices)
        elif product == 'VOLCANIC_ROCK':
            # Use improved statistical delta hedging strategy for volcanic rock
            # Apply risk limits for volcanic rock after consecutive losses
            if volcanic_risk_limit < 1.0:
                # Modify order depth to limit trade sizes during high risk periods
                limited_order_depth = OrderDepth()
                
                # Copy the order book but limit volumes
                if order_depth.buy_orders:
                    limited_order_depth.buy_orders = {
                        price: int(volume * volcanic_risk_limit) 
                        for price, volume in order_depth.buy_orders.items()
                    }
                if order_depth.sell_orders:
                    limited_order_depth.sell_orders = {
                        price: int(volume * volcanic_risk_limit) 
                        for price, volume in order_depth.sell_orders.items()
                    }
                
                # Use the limited order depth during high risk periods
                mid_price = mid_prices[product]
                orders = self.trade_volcanic_rock(product, limited_order_depth, state, mid_price)
            else:
                # Use normal order depth in normal conditions
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
    
    # Save state data with updated round and risk metrics
    
    traderData = jsonpickle.encode({
        "round": self.current_round,
        "volcanic_rock_recent_pnl": self.volcanic_rock_recent_pnl,
        "max_consecutive_losses": self.max_consecutive_losses,
        "last_volcanic_position": self.last_volcanic_position
    })
    
    return result, conversions, traderData