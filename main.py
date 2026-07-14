import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Import rich elements for premium terminal layout
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.align import Align
from rich.status import Status
from rich.columns import Columns

# Import our ML engine
import ml_engine

# Initialize rich Console
console = Console()

def clear_screen():
    """
    Clears the terminal screen for a clean dashboard look.
    """
    os.system('cls' if os.name == 'nt' else 'clear')

def generate_matplotlib_chart(ticker, data):
    """
    Generates and saves a premium, dark-themed forecast chart for the stock.
    Plots the last 60 trading days of history and the 10-day forecasts.
    """
    # Use dark style to match dark terminal theme
    plt.style.use('dark_background')
    fig, ax = plt.subplots(figsize=(10, 5.5))
    
    # Customize colors
    bg_color = '#0a0e1a'
    card_color = '#101626'
    grid_color = '#1f293d'
    
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(card_color)
    
    # We will plot the last 90 trading days of history for context
    lookback_plot = 90
    dates = data['dates'][-lookback_plot:]
    close = data['close'][-lookback_plot:]
    
    # Plot historical data
    ax.plot(dates, close, label='Historical Close', color='#a78bfa', linewidth=2)
    
    # Prepare forecast plot
    # The forecast line starts at the last historical day close to be continuous
    last_date = dates[-1]
    last_close = close[-1]
    
    forecast_dates = [last_date] + data['forecast']['dates']
    
    lr_forecast = [last_close] + data['forecast']['lr']
    rf_forecast = [last_close] + data['forecast']['rf']
    lstm_forecast = [last_close] + data['forecast']['lstm']
    
    # Plot forecasts
    ax.plot(forecast_dates, lr_forecast, label='Linear Regression Forecast', color='#3b82f6', linestyle='--', marker='o', markersize=4)
    ax.plot(forecast_dates, rf_forecast, label='Random Forest Forecast', color='#10b981', linestyle='--', marker='s', markersize=4)
    ax.plot(forecast_dates, lstm_forecast, label='LSTM Deep Learning Forecast', color='#ec4899', linestyle='--', marker='^', markersize=4)
    
    # Formatting
    ax.set_title(f"{ticker.upper()} Stock Price Prediction - 10-Day Forecast", fontsize=14, fontweight='bold', pad=15, color='#f3f4f6')
    ax.set_xlabel("Date", fontsize=11, labelpad=10, color='#9ca3af')
    ax.set_ylabel("Price ($)", fontsize=11, labelpad=10, color='#9ca3af')
    
    # Style grid
    ax.grid(True, which='both', linestyle=':', linewidth=0.5, color=grid_color)
    
    # Style borders
    for spine in ax.spines.values():
        spine.set_color('#2d3748')
        
    # Rotate x labels for readability
    plt.xticks(rotation=45)
    
    # Adjust x axis ticks to not crowd
    # Combine hist + forecast dates for tick selection
    all_dates = dates + data['forecast']['dates']
    tick_step = max(1, len(all_dates) // 10)
    ax.set_xticks(all_dates[::tick_step])
    
    ax.legend(frameon=True, facecolor=card_color, edgecolor='#2d3748')
    plt.tight_layout()
    
    filename = f"{ticker.lower()}_forecast.png"
    plt.savefig(filename, dpi=150, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    return filename

def render_cli_dashboard(ticker, data):
    """
    Renders the terminal-only stock forecast dashboard using rich panels and tables.
    """
    # 1. Title Banner
    title_text = Text.assemble(
        ("Predictify ", "bold white"),
        ("ML", "bold magenta"),
        (" - Stock Forecasting System (CLI Mode)", "white")
    )
    console.print(Panel(Align.center(title_text), style="bold purple", border_style="purple"))

    # 2. General Stats
    last_close = data['close'][-1]
    prev_close = data['close'][-2]
    change = last_close - prev_close
    change_pct = (change / prev_close) * 100
    
    price_text = Text(f"${last_close:.2f}", style="bold white")
    if change >= 0:
        change_text = Text(f"+${change:.2f} (+{change_pct:.2f}%)", style="bold green")
    else:
        change_text = Text(f"-${abs(change):.2f} (-{abs(change_pct):.2f}%)", style="bold red")

    # 3. Technical Crossover Signal Recommendation
    # Calculate indicators
    rsi = float(data['indicators']['rsi'][-1])
    macd = float(data['indicators']['macd'][-1])
    macd_sig = float(data['indicators']['macd_signal'][-1])
    sma20 = float(data['indicators']['sma_20'][-1])
    
    bullish = 0
    bearish = 0
    reasons = []
    
    if rsi <= 35:
        bullish += 1.5
        reasons.append("RSI Oversold (Bullish)")
    elif rsi >= 65:
        bearish += 1.5
        reasons.append("RSI Overbought (Bearish)")
        
    if macd > macd_sig:
        bullish += 1.0
        reasons.append("MACD Bullish Crossover")
    else:
        bearish += 1.0
        reasons.append("MACD Bearish Crossover")
        
    if last_close > sma20:
        bullish += 1.0
        reasons.append("Price > 20-Day SMA")
    else:
        bearish += 1.0
        reasons.append("Price < 20-Day SMA")
        
    net_score = bullish - bearish
    
    if net_score >= 1.0:
        signal_badge = "[bold green]BUY[/bold green]"
        signal_border = "green"
    elif net_score <= -1.0:
        signal_badge = "[bold red]SELL[/bold red]"
        signal_border = "red"
    else:
        signal_badge = "[bold yellow]HOLD[/bold yellow]"
        signal_border = "yellow"
        
    signal_reason = " & ".join(reasons[:2]) if reasons else "Neutral conditions"

    # Print general stats layout
    stats_table = Table.grid(expand=True)
    stats_table.add_column(justify="left", ratio=3)
    stats_table.add_column(justify="right", ratio=3)
    stats_table.add_row(
        Text.assemble(("Symbol: ", "grey70"), (f"{ticker.upper()}", "bold white"), (" | Close: ", "grey70"), price_text, (" (", "grey50"), change_text, (")", "grey50")),
        Text.assemble(("RSI(14): ", "grey70"), (f"{rsi:.1f}", "bold cyan"), (" | MACD: ", "grey70"), (f"{macd:.3f}", "bold cyan"))
    )
    
    console.print(Panel(stats_table, title="[bold]Stock Status[/bold]", border_style="blue"))
    console.print(Panel(Align.center(f"Trading Signal Recommendation: {signal_badge} ({signal_reason})"), border_style=signal_border))

    # 4. Model Performance comparison table
    metrics_table = Table(title="[bold]Model Evaluation Metrics (Test Set)[/bold]", border_style="grey30")
    metrics_table.add_column("Model Name", style="bold")
    metrics_table.add_column("RMSE ($)", justify="right")
    metrics_table.add_column("MAE ($)", justify="right")
    metrics_table.add_column("R2 Score", justify="right")
    metrics_table.add_column("Rank", justify="center")

    models = ['lr', 'rf', 'lstm']
    names = {
        'lr': 'Linear Regression',
        'rf': 'Random Forest Regressor',
        'lstm': 'LSTM Deep Learning Network'
    }
    
    best_model = 'lr'
    best_r2 = -999
    for m in models:
        r2 = data['metrics'][m]['r2']
        if r2 > best_r2:
            best_r2 = r2
            best_model = m

    for m in models:
        metrics = data['metrics'][m]
        if m == best_model:
            metrics_table.add_row(
                names[m],
                f"${metrics['rmse']:.2f}",
                f"${metrics['mae']:.2f}",
                f"{metrics['r2']:.4f}",
                "[bold green]TOP PERFORMER[/bold green]"
            )
        else:
            metrics_table.add_row(
                names[m],
                f"${metrics['rmse']:.2f}",
                f"${metrics['mae']:.2f}",
                f"{metrics['r2']:.4f}",
                "[grey50]Alternative[/grey50]"
            )
    console.print(metrics_table)

    # 5. 10-Day Forecast Table
    forecast_table = Table(title="[bold]10-Day Future Price Predictions[/bold]", border_style="grey30")
    forecast_table.add_column("Day / Date", style="bold")
    forecast_table.add_column("Linear Regression", justify="right", style="cyan")
    forecast_table.add_column("Random Forest", justify="right", style="green")
    forecast_table.add_column("LSTM Deep Learning", justify="right", style="magenta")

    forecast_len = len(data['forecast']['dates'])
    for i in range(forecast_len):
        date_str = data['forecast']['dates'][i]
        
        # Colorize prices depending on trend vs previous predicted day close
        prev_lr = last_close if i == 0 else data['forecast']['lr'][i-1]
        curr_lr = data['forecast']['lr'][i]
        lr_styled = f"[green]${curr_lr:.2f}[/green]" if curr_lr >= prev_lr else f"[red]${curr_lr:.2f}[/red]"
        
        prev_rf = last_close if i == 0 else data['forecast']['rf'][i-1]
        curr_rf = data['forecast']['rf'][i]
        rf_styled = f"[green]${curr_rf:.2f}[/green]" if curr_rf >= prev_rf else f"[red]${curr_rf:.2f}[/red]"

        prev_lstm = last_close if i == 0 else data['forecast']['lstm'][i-1]
        curr_lstm = data['forecast']['lstm'][i]
        lstm_styled = f"[green]${curr_lstm:.2f}[/green]" if curr_lstm >= prev_lstm else f"[red]${curr_lstm:.2f}[/red]"

        forecast_table.add_row(
            f"Day {i+1} ({date_str})",
            lr_styled,
            rf_styled,
            lstm_styled
        )
    console.print(forecast_table)
    
    # 6. Matplotlib saving confirmation
    chart_file = generate_matplotlib_chart(ticker, data)
    console.print(Panel(Align.center(f"[bold green]Success:[/bold green] Forecast chart saved in workspace: [bold underline]{chart_file}[/bold underline]"), border_style="green"))

def main():
    """
    Main interactive console loop.
    """
    clear_screen()
    console.print(Panel(Align.center("[bold white]Welcome to Predictify ML Terminal Interface[/bold white]\nType 'EXIT' at the ticker prompt to exit."), style="bold blue", border_style="blue"))
    
    while True:
        ticker = console.input("\n[bold yellow]Enter stock ticker (e.g. AAPL, TSLA, MSFT): [/bold yellow]").strip()
        
        if not ticker:
            continue
            
        if ticker.upper() == "EXIT":
            console.print("\n[bold cyan]Exiting Predictify ML. Goodbye![/bold cyan]")
            break
            
        try:
            # Spinner status during data retrieval and LSTM model fitting
            with Status("[bold magenta]Fetching stock history and fitting ML models (LSTM sequence training)...[/bold magenta]", spinner="dots") as status:
                data = ml_engine.get_predictions(ticker)
                
            clear_screen()
            render_cli_dashboard(ticker, data)
            
        except Exception as e:
            console.print(f"\n[bold red]Error processing '{ticker}':[/bold red] {e}")
            import traceback
            # Uncomment below for verbose debugging if requested
            # console.print(traceback.format_exc())

if __name__ == "__main__":
    main()
