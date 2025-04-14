import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def profit(x):
    """
    Computes profit based on x (selected value) while accounting for the gap between 200 and 250.
    
    For x between 160 and 200:
      - Average = (160 + x) / 2
      - Population percentage = (x - 160) / 120
    For x in [200, 250):
      - Average remains constant at 180 (the average of 160 and 200)
      - Percentage remains constant at (200-160)/120
    For x in [250, 320]:
      - Active population = 40 (from 160 to 200) + (x - 250) from the second segment
      - Population percentage = (40 + (x - 250)) / 120
      - Average is the weighted average:
            avg = [180 * 40 + ((250 + x)/2) * (x - 250)] / (40 + (x - 250))
    The final profit is:
          profit(x) = average * (320 - x) * (population percentage)
    """
    if x < 160:
        return 0
    elif x <= 200:
        avg = (160 + x) / 2
        pct = (x - 160) / 120
        return avg * (320 - x) * pct
    elif x < 250:
        # In the gap: no extra population beyond 200.
        avg = (160 + 200) / 2  # constant average = 180
        pct = (200 - 160) / 120  # constant percentage = 40/120
        return avg * (320 - x) * pct
    else:  # x between 250 and 320
        active_population = 40 + (x - 250)  # total active population in both segments
        pct = active_population / 120
        # Weighted average:
        avg_first = 180               # average for the first segment (160-200) over weight 40
        avg_second = (250 + x) / 2      # average for the second segment (250 to x)
        avg = (avg_first * 40 + avg_second * (x - 250)) / active_population
        return avg * (320 - x) * pct

# Generate data for plotting.
x_vals = np.linspace(160, 320, 300)
y_vals = np.array([profit(x) for x in x_vals])

# Create the plot.
fig, ax = plt.subplots()
plt.subplots_adjust(bottom=0.25)
line, = ax.plot(x_vals, y_vals, label='Profit Curve')

# Set initial slider value.
init_x = 200
slider_ax = plt.axes([0.25, 0.1, 0.65, 0.03])
x_slider = Slider(slider_ax, 'Select x', 160, 320, valinit=init_x)

# Plot a marker for the selected value.
marker, = ax.plot(init_x, profit(init_x), 'ro')

# Text annotation for current x value and profit.
info_text = ax.text(0.05, 0.95, '', transform=ax.transAxes, verticalalignment='top',
                    bbox=dict(facecolor='white', alpha=0.7))

def update(val):
    x_val = x_slider.val
    p_val = profit(x_val)
    marker.set_data([x_val], [p_val])
    info_text.set_text(f"x = {x_val:.1f}\nProfit = {p_val:.2f}")
    fig.canvas.draw_idle()

x_slider.on_changed(update)

ax.set_xlabel('Selected Value')
ax.set_ylabel('Profit')
ax.set_title('Interactive Profit Calculation (Including Gap)')
ax.legend()

plt.show()
