import yfinance as yf
import pandas as pd
import numpy as np
import os
from flask import Flask, render_template_string, jsonify, request
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

# ================== HYPER-AGGRESSIVE PARAMETERS ==================
BASE_DECAY_THRESHOLD = -0.05      # Tight exit trigger when momentum rolls flat
BASE_ENTRY_VELOCITY = 0.4          
SAFE_UPPER_BOUND = 1.05          # Tightens entry zone near structural value
BASE_PARABOLIC_CAP = 1.06        # Locks profits early when price is +6% above EMA20

# ================== ANALYTICS ENGINE ==================

def calculate_curl(prices, benchmark, span=3):
    rs = prices.div(benchmark + 1e-9, axis=0) * 100
    rs_smooth = rs.ewm(span=span, adjust=False).mean()
    ratio = 100 + ((rs_smooth - rs_smooth.rolling(10).mean()) / (rs_smooth.rolling(10).std() + 1e-9))
    mom = 100 + ((ratio.diff() - ratio.diff().rolling(5).mean()) / (ratio.diff().rolling(5).std() + 1e-9))
    return mom.diff()

def get_engine_data(lookback_days=0):
    # Fetch lookback context cleanly from data servers
    raw_all = yf.download(TICKERS + [BENCHMARK, VIX_TICKER], period="2y", interval="1d", progress=False)
    raw_close = raw_all['Close'].ffill().dropna()
    raw_vol = raw_all['Volume'].ffill()
    
    # Calculate historical time-travel index offsets
    target_offset = -1 - lookback_days
    prev_offset = target_offset - 1
    
    if abs(prev_offset) > len(raw_close):
        target_offset = -1
        prev_offset = -2
        
    target_date = raw_close.index[target_offset]
    
    vix = float(raw_close[VIX_TICKER].iloc[target_offset])
    
    # Adaptive Risk Shield Adjustments
    if vix >= 18.0:
        regime_status = "CRITICAL RISK-OFF (HIGH PANIC)"
        dynamic_entry_velocity = BASE_ENTRY_VELOCITY * 1.8
        dynamic_decay_threshold = 0.00                     # Lock immediately on flatline
        dynamic_parabolic_cap = 1.03                       # Pocket any minor +3% surge instantly
    elif 14.0 <= vix < 18.0:
        regime_status = "MODERATE VOLATILITY CHURN"
        dynamic_entry_velocity = BASE_ENTRY_VELOCITY * 1.2
        dynamic_decay_threshold = BASE_DECAY_THRESHOLD
        dynamic_parabolic_cap = BASE_PARABOLIC_CAP
    else:
        regime_status = "STABLE RISK-ON REGIME"
        dynamic_entry_velocity = BASE_ENTRY_VELOCITY
        dynamic_decay_threshold = BASE_DECAY_THRESHOLD
        dynamic_parabolic_cap = BASE_PARABOLIC_CAP
    
    # Mathematical MTF Modeling Layer
    q_curls = calculate_curl(raw_close[TICKERS], raw_close[BENCHMARK], span=65)
    m_curls = calculate_curl(raw_close[TICKERS], raw_close[BENCHMARK], span=20)
    w_curls = calculate_curl(raw_close[TICKERS], raw_close[BENCHMARK], span=5)  
    d_curls = calculate_curl(raw_close[TICKERS], raw_close[BENCHMARK], span=3)  
    d_curls_smooth = d_curls.ewm(span=3, adjust=False).mean()
    
    results = []
    for t in TICKERS:
        qc, mc, wc, dc = float(q_curls[t].iloc[target_offset]), float(m_curls[t].iloc[target_offset]), float(w_curls[t].iloc[target_offset]), float(d_curls[t].iloc[target_offset])
        qc_p, mc_p, wc_p, dc_p = float(q_curls[t].iloc[prev_offset]), float(m_curls[t].iloc[prev_offset]), float(w_curls[t].iloc[prev_offset]), float(d_curls[t].iloc[prev_offset])
        
        dc_smooth = float(d_curls_smooth[t].iloc[target_offset])
        dc_smooth_p = float(d_curls_smooth[t].iloc[prev_offset])
        
        qa = "↑" if qc > qc_p else "↓"
        ma = "↑" if mc > mc_p else "↓"
        wa = "↑" if wc > wc_p else "↓"
        da = "↑" if dc > dc_p else "↓"
        
        curr_vol = raw_vol[t].iloc[target_offset]
        avg_vol = raw_vol[t].rolling(20).mean().iloc[target_offset]
        vol_ratio = round(float(curr_vol / (avg_vol + 1e-9)), 2) if not np.isnan(avg_vol) else 0.0
        
        price = float(raw_close[t].iloc[target_offset])
        ema20 = float(raw_close[t].ewm(span=20, adjust=False).mean().iloc[target_offset])
        
        # Hyper-Short Trailing Filter Indicator
        ema3 = float(raw_close[t].ewm(span=3, adjust=False).mean().iloc[target_offset])
        
        r_min, r_max = round(ema20, 2), round(ema20 * SAFE_UPPER_BOUND, 2)
        exit_p = round(ema20 * dynamic_parabolic_cap, 2)
        dist_from_ema = price / ema20
        upside = round(((exit_p / price) - 1) * 100, 2)
        
        base_prob = 52
        if dc_smooth > 0.4: base_prob += 10
        if vol_ratio > 1.3: base_prob += 10
        prob_val = min(max(base_prob, 8), 94)

        # ================= HYPER-AGGRESSIVE RULES ENGINE =================
        exit_triggered = False
        
        # Rule 1: Tight Parabolic Cap Exhaustion (Harvest Gains Early)
        if dist_from_ema >= dynamic_parabolic_cap:
            rec, color, exit_triggered = "PARABOLIC EXIT", "danger", True
            
        # Rule 2: Trailing Stop Trap (Instant Liquidation below the 3-day line)
        elif price < ema3 and dist_from_ema > 1.01:
            rec, color, exit_triggered = "TRAILING STOP (EMA3 BREAK)", "danger", True
            
        # Rule 3: Hyper-Sensitive Momentum Decay
        elif dc_smooth < dynamic_decay_threshold and dist_from_ema > 1.02:
            rec, color, exit_triggered = "MOMENTUM DECAY", "warning", True
            
        # Rule 4: Shielded Entry Matrix
        elif (dc_smooth > dynamic_entry_velocity and dc_smooth > dc_smooth_p) and (wc > 0 or mc > 0) and (price <= r_max):
            rec, color = "STRONG BUY", "success"
            
        # Rule 5: System Stability State
        elif (dc_smooth > 0) and (wc > 0 or mc > 0):
            rec, color = "HOLD / ACCUMULATING", "primary"
        else:
            rec, color = "HOLD / NEUTRAL", "secondary"

        results.append({
            "ticker": t, "name": ETF_DATA[t], "price": f"₹{price:,.2f}",
            "qc": round(qc, 2), "qa": qa, "mc": round(mc, 2), "ma": ma,
            "wc": round(wc, 2), "wa": wa, "dc": round(dc, 2), "da": da,
            "vol": vol_ratio if vol_ratio > 0 else "N/A", 
            "entry_range": f"₹{r_min} - ₹{r_max}", "exit_target": f"₹{exit_p}",
            "exit_triggered": exit_triggered, "upside": upside, "prob": prob_val,
            "rec": rec, "color": color
        })
    
    return sorted(results, key=lambda x: x['mc'], reverse=True), round(vix, 2), regime_status, target_date.strftime('%Y-%m-%d')

# ================== ROUTES ==================

@app.route('/api/signals')
def api_signals():
    try:
        days = int(request.args.get('days', 0))
    except ValueError:
        days = 0
    data, vix, regime, t_stamp = get_engine_data(lookback_days=days)
    return jsonify({"historical_offset_days": days, "vix": vix, "market_regime": regime, "data_calculation_date": t_stamp, "signals": data})

@app.route('/')
def index():
    try:
        filter_val = request.args.get('days', '0')
        days = int(filter_val)
    except ValueError:
        filter_val = '0'
        days = 0
        
    data, vix, regime, t_stamp = get_engine_data(lookback_days=days)
    
    filter_options = [
        {"value": "0", "label": "Live Dashboard (Current)"},
        {"value": "1", "label": "Go Back 1 Day"},
        {"value": "2", "label": "Go Back 2 Days"},
        {"value": "3", "label": "Go Back 3 Days"},
        {"value": "4", "label": "Go Back 4 Days"},
        {"value": "5", "label": "Go Back 5 Days"},
        {"value": "10", "label": "Go Back 10 Days"},
        {"value": "15", "label": "Go Back 15 Days"}
    ]
    
    current_label = next((item["label"] for item in filter_options if item["value"] == filter_val), "Custom Lookback")

    html = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
        <style>
            body { background-color: #ffffff; color: #1a1d2e; font-family: 'Inter', sans-serif; font-size: 0.85rem; }
            .container-fluid { max-width: 1500px; padding: 2rem; }
            .terminal-card { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 12px; overflow: hidden; box-shadow: 0 4px 12px rgba(0,0,0,0.05); }
            .vix-badge { background: #212529; color: #ffc107; padding: 8px 15px; border-radius: 50px; font-weight: bold; font-family: monospace; border: 1px solid #ffc107; }
            .regime-banner { background: #e9ecef; border-left: 5px solid #dc3545; padding: 10px 15px; border-radius: 4px; font-weight: 600; font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.5px; }
            .table { color: #1a1d2e; margin-bottom: 0; }
            .table thead th { background: #e9ecef; border-bottom: 2px solid #dee2e6; color: #495057; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 1px; padding: 15px; }
            .table tbody tr { border-bottom: 1px solid #dee2e6; transition: 0.2s; }
            .table tbody tr:hover { background: #f1f3f5; }
            .ticker-cell { font-family: 'JetBrains Mono', monospace; color: #0d6efd; font-weight: 700; font-size: 1.05rem; display: block; margin-bottom: 4px; }
            .price-tag { display: inline-block; background: #212529; color: #20c997; padding: 2px 10px; border-radius: 4px; font-weight: 800; font-size: 0.9rem; margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; }
            .pill { padding: 3px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: bold; margin: 0 2px; }
            .pill-up { background: #d1e7dd; color: #0f5132; }
            .pill-down { background: #f8d7da; color: #842029; }
            .vol-chip { background: #343a40; color: #ffffff; padding: 4px 10px; border-radius: 6px; font-weight: bold; font-size: 0.85rem; }
            .vol-na { background: #ffc107 !important; color: #000 !important; }
            .progress { background: #dee2e6; height: 8px; border-radius: 10px; margin-top: 5px; }
            .progress-bar-prob { background: #0d6efd; }
            .exit-row { background: rgba(220, 53, 69, 0.07) !important; }
            .btn-api { border: 1px solid #dee2e6; color: #6c757d; font-size: 0.75rem; text-decoration: none; padding: 8px 15px; border-radius: 6px; background: #fff; }
            .btn-api:hover { background: #e9ecef; color: #212529; }
            .text-range { color: #055160; font-weight: 700; }
            .text-target { color: #dc3545; font-weight: 700; }
        </style>
        <title>Alpha Aggressor v10.5 SWING ENGINE</title>
    </head>
    <body>
        <div class="container-fluid">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <div>
                    <h2 class="fw-bold mb-0" style="color: #212529;">ALPHA AGGRESSOR <span style="color: #dc3545;">v10.6</span></h2>
                    <p class="text-muted mb-0">Active Matrix State Date: <span class="badge bg-danger fs-6">{{ t_stamp }}</span></p>
                </div>
                
                <div class="d-flex align-items-center gap-2">
                    <div class="dropdown">
                        <button class="btn btn-danger dropdown-toggle fw-bold" type="button" id="timeframeDropdown" data-bs-toggle="dropdown" aria-expanded="false" style="padding: 8px 16px; font-size: 0.85rem;">
                            ⏳ Time-Shift Offset: {{ current_label }}
                        </button>
                        <ul class="dropdown-menu dropdown-menu-end shadow-sm" aria-labelledby="timeframeDropdown">
                            {% for opt in filter_options %}
                            <li><a class="dropdown-item {{ 'active fw-bold bg-danger' if opt.value == filter_val else '' }}" href="/?days={{ opt.value }}">{{ opt.label }}</a></li>
                            {% endfor %}
                        </ul>
                    </div>
                    <a href="/api/signals?days={{ filter_val }}" target="_blank" class="btn-api">View Endpoint JSON</a>
                    <span class="vix-badge">REGIME VIX: {{ vix }}</span>
                </div>
            </div>

            <div class="regime-banner mb-4">
                Hyper-Aggressive System Engine Shield: <span class="text-danger font-weight-bold">{{ regime }}</span>
            </div>

            <div class="terminal-card">
                <table class="table table-borderless align-middle">
                    <thead>
                        <tr>
                            <th>Instrument / Price</th>
                            <th>MTF Curl Chain (Q/M/W/D)</th>
                            <th>Vol Ratio</th>
                            <th>Safe Buy Range</th>
                            <th>Aggressive Target</th>
                            <th>Regime Potential</th>
                            <th width="150">Confidence</th>
                            <th class="text-center">Signal Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in data %}
                        <tr class="{{ 'exit-row' if row.exit_triggered else '' }}">
                            <td>
                                <span class="ticker-cell">{{ row.ticker }}</span>
                                <span class="price-tag">{{ row.price }}</span><br>
                                <small class="text-muted" style="font-size: 0.7rem;">{{ row.name }}</small>
                            </td>
                            <td>
                                <span class="pill {{ 'pill-up' if row.qa == '↑' else 'pill-down' }}">{{ row.qc }} {{ row.qa }}</span>
                                <span class="pill {{ 'pill-up' if row.ma == '↑' else 'pill-down' }}">{{ row.mc }} {{ row.ma }}</span>
                                <span class="pill {{ 'pill-up' if row.wa == '↑' else 'pill-down' }}">{{ row.wc }} {{ row.wa }}</span>
                                <span class="pill {{ 'pill-up' if row.da == '↑' else 'pill-down' }}">{{ row.dc }} {{ row.da }}</span>
                            </td>
                            <td><div class="vol-chip {{ 'vol-na' if row.vol == 'N/A' else '' }}">{{ row.vol }}{{ 'x' if row.vol != 'N/A' else '' }}</div></td>
                            <td class="text-range">{{ row.entry_range }}</td>
                            <td class="text-target">{{ row.exit_target }}</td>
                            <td><span class="fw-bold {{ 'text-success' if row.upside > 0 else 'text-danger' }}">{{ row.upside }}%</span></td>
                            <td>
                                <div class="d-flex justify-content-between mb-1" style="font-size: 0.65rem;"><strong>{{ row.prob }}%</strong></div>
                                <div class="progress"><div class="progress-bar progress-bar-prob" style="width: {{ row.prob }}%"></div></div>
                            </td>
                            <td class="text-center"><span class="badge bg-{{ row.color }} py-2 px-3 w-100">{{ row.rec }}</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html, data=data, vix=vix, regime=regime, t_stamp=t_stamp, filter_options=filter_options, filter_val=filter_val, current_label=current_label)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
