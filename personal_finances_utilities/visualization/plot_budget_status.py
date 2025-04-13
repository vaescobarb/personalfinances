import matplotlib.pyplot as plt
import numpy as np

def plot_budget_status(merged_budget, show_percentage=True, month=None):
    """
    Plots a horizontal bar chart of budget performance by category/subcategory with color-coded labels and alternating row backgrounds.
    """
    color_gradient = [
        '#006400', '#1e7a1e', '#2e8b57', '#32cd32', '#adff2f',
        '#ffff00', '#ffd700', '#ffa500', '#ff4500', '#ff0000'
    ]

    def get_color(value):
        if value < 0:
            return '#ff0000'
        value = min(max(value, 0), 100)
        idx = int((1 - value / 100) * (len(color_gradient) - 1))
        return color_gradient[idx]

    # Determine data source
    if month:
        month_col = f"{month.capitalize()}_over_under"
        if month_col not in merged_budget.columns:
            raise ValueError(f"Column '{month_col}' not found in merged_budget.")
        merged_budget['value_to_plot'] = merged_budget[month_col].astype(float)
        merged_budget['bar_color'] = merged_budget['value_to_plot'].apply(
            lambda x: '#ff0000' if x < 0 else '#32cd32'
        )
        x_label = f'{month.capitalize()} Budget Status (Currency)'
        is_percentage = False
    else:
        merged_budget['percentage_of_budget'] = (
            merged_budget['year_remaining'].astype(float) / merged_budget['annually'].astype(float)
        ) * 100
        if show_percentage:
            merged_budget['value_to_plot'] = merged_budget['percentage_of_budget']
            merged_budget['bar_color'] = merged_budget['value_to_plot'].apply(get_color)
            x_label = 'Percentage of Budget (Year Remaining)'
            is_percentage = True
        else:
            merged_budget['value_to_plot'] = merged_budget['year_remaining'].astype(float)
            merged_budget['bar_color'] = merged_budget['value_to_plot'].apply(
                lambda x: '#ff0000' if x < 0 else '#32cd32'
            )
            x_label = 'Amount Remaining (Currency)'
            is_percentage = False

    # Sort and extract
    merged_budget = merged_budget.sort_values('value_to_plot', ascending=True)
    categories = merged_budget['Category'] + " - " + merged_budget['Subcategory']
    values = merged_budget['value_to_plot']
    colors = merged_budget['bar_color']

    fig, ax = plt.subplots(figsize=(16, 8))
    y_positions = np.arange(len(categories))

    # Alternating row backgrounds
    for i in range(len(categories)):
        if i % 2 == 0:
            ax.axhspan(i - 0.5, i + 0.5, color='#f5f5f5', zorder=0)

    bars = ax.barh(y_positions, values, color=colors, zorder=2)
    max_val = max(values.max(), abs(values.min()))
    ax.set_xlim(-max_val * 1.3, max_val * 1.3)

    # Add bar labels
    for bar, val, color in zip(bars, values, colors):
        y = bar.get_y() + bar.get_height() / 2
        x = bar.get_width()
        label = f"{val:.0f}%" if is_percentage else f"NOK {val:,.2f}"
        align = 'left' if x >= 0 else 'right'
        offset = 8 if x >= 0 else -8
        ax.text(x + offset, y, label, va='center', ha=align, color=color, fontsize=9, zorder=3)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(categories)
    ax.set_xlabel(x_label)
    ax.set_ylabel('Category - Subcategory')
    ax.set_title(f'Budget Status: {month.capitalize() if month else "Annual Overview"}')
    ax.axvline(0, color='black', linewidth=0.8, linestyle='--', zorder=1)

    plt.subplots_adjust(left=0.3)
    plt.show()
