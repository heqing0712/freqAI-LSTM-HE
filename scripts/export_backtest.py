import argparse
import json
import os
import zipfile
from datetime import datetime

import pandas as pd
import plotly.graph_objects as go


def load_backtest(zip_path: str) -> dict:
    z = zipfile.ZipFile(zip_path)
    name = [n for n in z.namelist() if n.endswith('.json') and '_config' not in n][0]
    data = json.loads(z.read(name))
    return data


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def to_dt(ts):
    return pd.to_datetime(ts, unit='ms', utc=True)


def export(zip_path: str, outdir: str, plotdir: str, plotname: str = 'profit.html'):
    data = load_backtest(zip_path)
    key = next(iter(data['strategy'].keys()))
    strat = data['strategy'][key]
    trades = pd.DataFrame(strat['trades'])
    trades['close_dt'] = trades['close_timestamp'].apply(to_dt)
    trades['date'] = trades['close_dt'].dt.tz_convert('UTC')
    trades['profit_abs'] = trades['profit_abs'].astype(float)
    trades['enter_tag'] = trades['enter_tag'].astype(str)
    trades['exit_reason'] = trades['exit_reason'].astype(str)
    trades['pair'] = trades['pair'].astype(str)

    ensure_dir(outdir)
    ensure_dir(plotdir)

    trades.to_csv(os.path.join(outdir, 'trades.csv'), index=False)

    day = trades.groupby(trades['date'].dt.date).agg(
        trades=('profit_abs', 'size'),
        profit_usdt=('profit_abs', 'sum'),
        wins=('profit_abs', lambda s: (s > 0).sum()),
        losses=('profit_abs', lambda s: (s < 0).sum()),
    ).reset_index().rename(columns={'date': 'day'})
    day.to_csv(os.path.join(outdir, 'breakdown_day.csv'), index=False)

    week = trades.set_index('date').groupby(pd.Grouper(freq='W-MON')).agg(
        trades=('profit_abs', 'size'),
        profit_usdt=('profit_abs', 'sum'),
        wins=('profit_abs', lambda s: (s > 0).sum()),
        losses=('profit_abs', lambda s: (s < 0).sum()),
    ).reset_index().rename(columns={'date': 'week'})
    week.to_csv(os.path.join(outdir, 'breakdown_week.csv'), index=False)

    month = trades.set_index('date').groupby(pd.Grouper(freq='M')).agg(
        trades=('profit_abs', 'size'),
        profit_usdt=('profit_abs', 'sum'),
        wins=('profit_abs', lambda s: (s > 0).sum()),
        losses=('profit_abs', lambda s: (s < 0).sum()),
    ).reset_index().rename(columns={'date': 'month'})
    month.to_csv(os.path.join(outdir, 'breakdown_month.csv'), index=False)

    enter_exit = trades.groupby(['enter_tag', 'exit_reason']).agg(
        trades=('profit_abs', 'size'),
        profit_usdt=('profit_abs', 'sum'),
        wins=('profit_abs', lambda s: (s > 0).sum()),
        losses=('profit_abs', lambda s: (s < 0).sum()),
    ).reset_index()
    enter_exit.to_csv(os.path.join(outdir, 'enter_exit.csv'), index=False)

    by_pair = trades.groupby(['pair']).agg(
        trades=('profit_abs', 'size'),
        profit_usdt=('profit_abs', 'sum'),
        wins=('profit_abs', lambda s: (s > 0).sum()),
        losses=('profit_abs', lambda s: (s < 0).sum()),
    ).reset_index()
    by_pair.to_csv(os.path.join(outdir, 'pair_breakdown.csv'), index=False)

    starting_balance = strat.get('starting_balance', 8000)
    equity = trades.sort_values('close_dt')[['close_dt', 'profit_abs']].copy()
    equity['equity'] = starting_balance + equity['profit_abs'].cumsum()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=equity['close_dt'], y=equity['equity'], mode='lines', name='Equity'))
    fig.update_layout(title='Equity Curve', xaxis_title='Time', yaxis_title='Equity (USDT)')
    fig.write_html(os.path.join(plotdir, plotname))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--zip', required=True)
    ap.add_argument('--outdir', required=True)
    ap.add_argument('--plotdir', required=True)
    ap.add_argument('--plotname', default='profit.html')
    args = ap.parse_args()
    export(args.zip, args.outdir, args.plotdir, args.plotname)


if __name__ == '__main__':
    main()
