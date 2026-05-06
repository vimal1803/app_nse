import yfinance as yf
import pandas as pd
import numpy as np
import os
from flask import Flask, render_template_string, jsonify
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ================== CONFIGURATION ==================
ETF_DATA = {
    "BANKBEES.NS": "Nippon India Bank BeES", 
    "PSUBNKBEES.NS": "Nippon India PSU Bank BeES",
    "ITBEES.NS": "Nippon India IT BeES", 
    "PHARMABEES.NS": "Nippon India Pharma BeES", 
    "AUTOIETF.NS": "ICICI Prudential Nifty Auto ETF", 
    "FMCGIETF.NS": "ICICI Prudential Nifty FMCG ETF",
    "METALIETF.NS": "ICICI Prudential Nifty Metal ETF", 
    "CONSUMBEES.NS": "Nippon India ETF Consumption",
    "MOREALTY.NS": "Motilal Oswal Nifty Realty ETF", 
    "INFRABEES.NS": "Nippon India Infra BeES",
    "CPSEETF.NS": "CPSE ETF (Energy & PSUs)", 
    "OIL.NS": "Nippon India ETF Oil & Gas",
    "MODEFENCE.NS": "Motilal Oswal Nifty Defence Index ETF", 
    "ALPHA.NS" : "Kotak Nifty Alpha 50 ETF", 
    "MOMENTUM.NS": "Motilal Oswal Momentum 30 ETF", 
    "MIDCAPETF.NS": "Mirae Asset Midcap 150 ETF", 
    "EVINDIA.NS": "Mirae Asset Nifty EV & New Mobility ETF"
}

TICKERS = list(ETF_DATA.keys())
BENCHMARK = "^CRSLDX" 
VIX_TICKER = "^INDIAVIX"

# ================== ANALYTICS ENGINE ==================

def calculate_curl(prices, benchmark, span=5):
    """Enhanced Curl Engine with adaptive smoothing."""
    rs = prices.div(benchmark + 1e-9, axis=0) * 100
    rs_smooth = rs.ewm(span=span, adjust=False).mean()
    
    # Z-Score normalization for relative strength stability
    ratio = 100 + ((rs_smooth - rs_smooth.rolling(20).mean()) / (rs_smooth.rolling(20).std() + 1e-9))
    
    # Second derivative (Momentum of Momentum)
    diff_ratio = ratio.diff()
    mom = 100 + ((diff_ratio - diff_ratio.rolling(10).mean()) / (diff_ratio.rolling(10).std() + 1e-9))
    
    # Final smoothing to prevent flickering
    return mom.diff().ewm(span=3, adjust=False).mean()

def get_engine_data():
    # Fetch 2 years of data to ensure indicators (like 200-day averages) are fully primed
    raw_all = yf.download(TICKERS + [BENCHMARK, VIX_TICKER], period="2y", interval="1d", progress=False)
    raw_close = raw_all['Close'].ffill().dropna()
    raw_vol = raw_all['Volume'].ffill()
    raw_high = raw_all['High'].ffill()
    raw_low = raw_all['Low'].ffill()
    
    vix = float(raw_close[VIX_TICKER].iloc[-1])
    
    # VIX-Adaptive Threshold Logic
    # High VIX (>18) = Tighten entry, loosen exit (Defensive)
    # Low VIX (<14) = Aggressive entry (Offensive)
    vix_factor = max(1.0, vix / 15.0)
    ENTRY_VELOCITY = 0.35 * vix_factor
    DECAY_THRESHOLD = -0.45 / vix_factor
    
    # Calculate MTF Curls
    q_curls = calculate_curl(raw_close[TICKERS], raw_close[BENCHMARK], span=65)
    m_curls = calculate_curl(raw_close[TICKERS], raw_close[BENCHMARK], span=21)
    w_curls = calculate_curl(raw_close[TICKERS], raw_close[BENCHMARK], span=10)
    d_curls = calculate_curl(raw_close[TICKERS], raw_close[BENCHMARK], span=5)
    
    results = []
    for t in TICKERS:
        qc, mc, wc, dc = float(q_curls[t].iloc[-1]), float(m_curls[t].iloc[-1]), float(w_curls[t].iloc[-1]), float(d_curls[t].iloc[-1])
        qc_p, mc_p, wc_p, dc_p = float(q_curls[t].iloc[-2]), float(m_curls[t].iloc[-2]), float(w_curls[t].iloc[-2]), float(d_curls[t].iloc[-2])
        
        qa, ma, wa, da = ("↑" if qc > qc_p else "↓"), ("↑" if mc > mc_p else "↓"), ("↑" if wc > wc_p else "↓"), ("↑" if dc > dc_p else "↓")
        
        # Volatility Calculation (ATR % based)
        tr = np.maximum(raw_high[t] - raw_low[t], np.maximum(abs(raw_high[t] - raw_close[t].shift(1)), abs(raw_low[t] - raw_close[t].shift(1))))
        atr_pct = (tr.rolling(14).mean() / raw_close[t]).iloc[-1] * 100

        # Volume Confirmation
        curr_vol = raw_vol[t].iloc[-1]
        avg_vol = raw_vol[t].rolling(20).mean().iloc[-1]
        vol_ratio = round(float(curr_vol / (avg_vol + 1e-9)), 2)
        
        price = float(raw_close[t].iloc[-1])
        ema20 = float(raw_close[t].ewm(span=20, adjust=False).mean().iloc[-1])
        dist_from_ema = price / ema20
        
        # Scoring Algorithm (Weighting: Monthly 50%, Weekly 30%, Daily 20%)
        aggro_score = (mc * 0.5) + (wc * 0.3) + (dc * 0.2)
        
        # Decision Matrix
        exit_triggered = False
        if dist_from_ema >= 1.12:
            rec, color, exit_triggered = "PARABOLIC CLIMAX", "danger", True
        elif dc < DECAY_THRESHOLD and dist_from_ema > 1.04:
            rec, color, exit_triggered = "MOMENTUM ROTATION", "warning", True
        elif (dc > ENTRY_VELOCITY and dc > dc_p) and (dist_from_ema <= 1.08):
            rec, color = "STRONG ACCUMULATE", "success"
        elif aggro_score > 0.1:
            rec, color = "BULLISH HOLD", "info"
        else:
            rec, color = "WAIT / OBSERVE", "secondary"

        results.append({
            "ticker": t, "name": ETF_DATA[t], "price": f"₹{price:,.2f}",
            "qc": round(qc, 2), "qa": qa, "mc": round(mc, 2), "ma": ma,
            "wc": round(wc, 2), "wa": wa, "dc": round(dc, 2), "da": da,
            "score": round(aggro_score, 2), "atr": round(atr_pct, 2),
            "vol": vol_ratio, "rec": rec, "color": color, "exit_triggered": exit_triggered
        })
    
    # Sort by the new Weighted Aggro Score
    return sorted(results, key=lambda x: x['score'], reverse=True), round(vix, 2)

# ================== ROUTES ==================

@app.route('/api/signals')
def api_signals():
    data, vix = get_engine_data()
    return jsonify({"vix": vix, "timestamp": pd.Timestamp.now().isoformat(), "signals": data})

@app.route('/')
def index():
    try:
        data, vix = get_engine_data()
        html = """
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
            <style>
                body { background-color: #f4f7f6; color: #2c3e50; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; font-size: 0.85rem; }
                .container-fluid { max-width: 1600px; padding: 2rem; }
                .terminal-card { background: white; border-radius: 15px; border: none; box-shadow: 0 10px 30px rgba(0,0,0,0.08); overflow: hidden; }
                .vix-indicator { font-weight: 800; padding: 10px 20px; border-radius: 8px; background: #e74c3c; color: white; }
                .table thead th { background: #f8f9fa; color: #7f8c8d; font-weight: 600; text-transform: uppercase; border: none; padding: 1.2rem; }
                .score-pill { background: #2c3e50; color: white; padding: 5px 12px; border-radius: 6px; font-weight: bold; font-family: monospace; }
                .exit-row { background-color: rgba(231, 76, 60, 0.05) !important; }
                .status-badge { border-radius: 4px; padding: 8px; width: 100%; display: block; font-weight: bold; text-align: center; }
            </style>
            <title>Alpha Aggressor v11.0 Robust</title>
        </head>
        <body>
            <div class="container-fluid">
                <div class="d-flex justify-content-between align-items-center mb-5">
                    <div>
                        <h1 class="fw-bold mb-0">ALPHA AGGRESSOR <span class="text-primary">v11.0</span></h1>
                        <p class="text-muted">VIX-Adaptive Robust Momentum Engine</p>
                    </div>
                    <div class="vix-indicator">INDIA VIX: {{ vix }}</div>
                </div>

                <div class="terminal-card">
                    <table class="table align-middle mb-0">
                        <thead>
                            <tr>
                                <th>Instrument</th>
                                <th>Aggro Score</th>
                                <th>MTF Curl Chain</th>
                                <th>ATR % (Vol)</th>
                                <th>Vol Ratio</th>
                                <th>Signal Status</th>
                            </tr>
                        </thead>
                        <tbody>
                            {% for row in data %}
                            <tr class="{{ 'exit-row' if row.exit_triggered else '' }}">
                                <td>
                                    <span class="fw-bold d-block" style="font-size: 1.1rem; color: #0d6efd;">{{ row.ticker }}</span>
                                    <span class="text-dark fw-bold">{{ row.price }}</span><br>
                                    <small class="text-muted">{{ row.name }}</small>
                                </td>
                                <td><span class="score-pill">{{ row.score }}</span></td>
                                <td>
                                    <span class="badge {{ 'bg-success' if row.qa == '↑' else 'bg-danger' }}">{{ row.qc }} {{ row.qa }}</span>
                                    <span class="badge {{ 'bg-success' if row.ma == '↑' else 'bg-danger' }}">{{ row.mc }} {{ row.ma }}</span>
                                    <span class="badge {{ 'bg-success' if row.wa == '↑' else 'bg-danger' }}">{{ row.wc }} {{ row.wa }}</span>
                                    <span class="badge {{ 'bg-success' if row.da == '↑' else 'bg-danger' }}">{{ row.dc }} {{ row.da }}</span>
                                </td>
                                <td>{{ row.atr }}%</td>
                                <td><span class="badge bg-dark">{{ row.vol }}x</span></td>
                                <td><span class="status-badge bg-{{ row.color }}">{{ row.rec }}</span></td>
                            </tr>
                            {% endfor %}
                        </tbody>
                    </table>
                </div>
            </div>
        </body>
        </html>
        """
        return render_template_string(html, data=data, vix=vix)
    except Exception as e:
        return f"System Error: {str(e)}"

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
