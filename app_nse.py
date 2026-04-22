import os
from flask import Flask, jsonify, render_template_string, request
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ================== CONFIGURATION ==================
ETF_DATA = {
    "BANKBEES.NS": "Banking (Private)", "PSUBNKBEES.NS": "Banking (PSU)",
    "ITBEES.NS": "IT / Tech", "HEALTHIETF.NS": "Healthcare",
    "PHARMABEES.NS": "Pharmaceuticals", "AUTOIETF.NS": "Automobile",
    "FMCGIETF.NS": "FMCG", "MOREALTY.NS": "Real Estate",
    "CPSEETF.NS": "Energy", "INFRABEES.NS": "Infrastructure",
    "METALIETF.NS": "Metals", "CONSUMBEES.NS": "Consumption",
    "NIFTY_FIN_SERVICE.NS": "Financial Services", "MODEFENCE.NS": "Defense",
    "MAKEINDIA.NS": "Manufacturing", "CAPITALSFB.NS": "Capital Markets",
    "TNIDETF.NS": "Digital India", "EVINDIA.NS": "EV & New Age Auto",
    "ICICIB22.NS": "Diversified PSU", "ALPHA.NS" : "Alpha"
}

BENCHMARKS = {
    "^CRSLDX": "Nifty 500 (Broad)",
    "^NSEI": "Nifty 50 (Large Cap)",
    "^NSEMDCP50": "Nifty Midcap 50",
    "MON100.NS": "Nasdaq 100 (Global Tech)"
}

TICKERS = list(ETF_DATA.keys())

# ================== RRG ENGINE LOGIC ==================

def calculate_rrg_metrics(data, tickers, bench):
    rs = data[tickers].div(data[bench], axis=0) * 100
    rs_smooth = rs.ewm(span=3, adjust=False).mean()
    ratio = 100 + ((rs_smooth - rs_smooth.rolling(10).mean()) / rs_smooth.rolling(10).std())
    mom_raw = ratio.diff()
    mom = 100 + ((mom_raw - mom_raw.rolling(5).mean()) / mom_raw.rolling(5).std())
    curl = mom.diff().iloc[-1]
    return ratio.iloc[-1], mom.iloc[-1], curl

def get_market_intelligence(benchmark_ticker):
    m_data = yf.download(TICKERS + [benchmark_ticker], period="5y", interval="1mo", progress=False)['Close'].ffill().dropna()
    w_data = yf.download(TICKERS + [benchmark_ticker], period="2y", interval="1wk", progress=False)['Close'].ffill().dropna()
    d_data = yf.download(TICKERS + [benchmark_ticker], period="6mo", interval="1d", progress=False)['Close'].ffill().dropna()

    m_ratio, m_mom, m_curl = calculate_rrg_metrics(m_data, TICKERS, benchmark_ticker)
    _, _, w_curl = calculate_rrg_metrics(w_data, TICKERS, benchmark_ticker)
    _, _, d_curl = calculate_rrg_metrics(d_data, TICKERS, benchmark_ticker)

    results = []
    for t in TICKERS:
        q = "Leading" if m_ratio[t] >= 100 and m_mom[t] >= 100 else \
            "Weakening" if m_ratio[t] >= 100 and m_mom[t] < 100 else \
            "Lagging" if m_ratio[t] < 100 and m_mom[t] < 100 else "Improving"
        
        score = (m_curl[t] * 2.5) 
        if w_curl[t] > 0.5: score += 2.0
        if d_curl[t] > 0.2: score += 1.0
        if q == "Improving": score += 2.5 
        if q in ["Leading", "Weakening"]: score -= 3.0
        
        final_score = score if m_curl[t] > 0 else min(score, 2.0)
        status = "🔥 BUY/HOLD" if final_score > 6 else "❄️ EXIT" if final_score < 0 else "⏳ NEUTRAL"

        results.append({
            "ticker": t, "sector": ETF_DATA[t], "quad": q,
            "m_curl": round(m_curl[t], 2), "w_curl": round(w_curl[t], 2), "d_curl": round(d_curl[t], 2),
            "score": round(final_score, 2), "status": status
        })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)

# ================== FLASK ROUTES ==================

@app.route('/')
def index():
    selected_bench = request.args.get('bench', '^CRSLDX')
    data = get_market_intelligence(selected_bench)
    
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NSE RRG Engine</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background-color: #f1f4f9; font-family: 'Segoe UI', sans-serif; }
            .header-card { background: #1a237e; color: white; border-radius: 15px; margin-bottom: 20px; padding: 25px; }
            .badge-leading { background: #2e7d32; } 
            .badge-improving { background: #1565c0; }
            .badge-lagging { background: #c62828; }
            .badge-weakening { background: #6c757d; color: white; } /* Updated to Grey */
            .score-cell { font-size: 1.2rem; font-weight: bold; color: #1a237e; }
        </style>
    </head>
    <body>
        <div class="container py-4">
            <div class="header-card d-flex justify-content-between align-items-center">
                <div>
                    <h2 class="mb-0">🚀 NSE Sector Engine</h2>
                    <p class="mb-0 text-white-50">Triple-Timeframe Rotation Analysis</p>
                </div>
                <div class="d-flex align-items-center">
                    <label class="me-2 fw-bold">Benchmark:</label>
                    <select class="form-select w-auto" onchange="window.location.href='/?bench='+this.value">
                        {% for ticker, name in benchmarks.items() %}
                        <option value="{{ ticker }}" {{ 'selected' if ticker == current_bench else '' }}>{{ name }}</option>
                        {% endfor %}
                    </select>
                </div>
            </div>

            <div class="card shadow-sm border-0 rounded-4 overflow-hidden">
                <table class="table table-hover align-middle mb-0 text-center">
                    <thead class="table-light">
                        <tr>
                            <th class="text-start">Ticker / Sector</th>
                            <th>Trend</th>
                            <th>M_Curl</th>
                            <th>W_Curl</th>
                            <th>D_Curl</th>
                            <th>Prob_Score</th>
                            <th>Action</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for row in results %}
                        <tr>
                            <td class="text-start ps-3">
                                <strong>{{ row.ticker }}</strong><br>
                                <small class="text-muted">{{ row.sector }}</small>
                            </td>
                            <td><span class="badge badge-{{ row.quad|lower }}">{{ row.quad }}</span></td>
                            <td>{{ row.m_curl }}</td>
                            <td>{{ row.w_curl }}</td>
                            <td>{{ row.d_curl }}</td>
                            <td class="score-cell">{{ row.score }}</td>
                            <td>
                                <span class="badge {{ 'bg-success' if 'BUY' in row.status else 'bg-danger' if 'EXIT' in row.status else 'bg-secondary' }}">
                                    {{ row.status }}
                                </span>
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, results=data, benchmarks=BENCHMARKS, current_bench=selected_bench)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
