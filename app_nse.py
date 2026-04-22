from flask import Flask, jsonify, render_template_string
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

# ================== NSE CONFIGURATION ==================
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
TICKERS = list(ETF_DATA.keys())
BENCHMARK = "^CRSLDX"
TOTAL_CAPITAL = 100000

# ================== RRG ENGINE CORE ==================
def calculate_rrg_metrics(data, tickers, bench):
    rs = data[tickers].div(data[bench], axis=0) * 100
    rs_smooth = rs.ewm(span=3, adjust=False).mean()
    
    # Standard RRG Normalization (10/5 lookback)
    ratio = 100 + ((rs_smooth - rs_smooth.rolling(10).mean()) / rs_smooth.rolling(10).std())
    mom_raw = ratio.diff()
    mom = 100 + ((mom_raw - mom_raw.rolling(5).mean()) / mom_raw.rolling(5).std())
    
    curl = mom.diff().iloc[-1]
    velocity = np.sqrt(ratio.diff()**2 + mom.diff()**2).iloc[-1]
    
    return ratio.iloc[-1], mom.iloc[-1], curl, velocity

def get_nse_rrg_data():
    # Fetch Monthly (Guardrail) and Weekly (Timing) Data
    m_data = yf.download(TICKERS + [BENCHMARK], period="5y", interval="1mo", progress=False)['Close'].ffill().dropna()
    w_data = yf.download(TICKERS + [BENCHMARK], period="2y", interval="1wk", progress=False)['Close'].ffill().dropna()

    m_ratio, m_mom, m_curl, m_vel = calculate_rrg_metrics(m_data, TICKERS, BENCHMARK)
    w_ratio, w_mom, w_curl, w_vel = calculate_rrg_metrics(w_data, TICKERS, BENCHMARK)

    results = []
    for t in TICKERS:
        q = "Leading" if m_ratio[t] >= 100 and m_mom[t] >= 100 else \
            "Weakening" if m_ratio[t] >= 100 and m_mom[t] < 100 else \
            "Lagging" if m_ratio[t] < 100 and m_mom[t] < 100 else "Improving"
        
        # Scoring Logic
        score = (m_curl[t] * 2.5) 
        if w_curl[t] > 0.5: score += 2.0
        if q == "Improving": score += 2.5
        if q in ["Leading", "Weakening"]: score -= 3.0
        
        # Monthly Directional Guardrail
        is_guarded = m_curl[t] <= 0
        final_score = score if not is_guarded else min(score, 2.0)
        
        status = "🔥 BUY/HOLD" if final_score > 6 else "❄️ EXIT" if final_score < 0 else "⏳ NEUTRAL"

        results.append({
            "ticker": t, "sector": ETF_DATA[t], "quadrant": q,
            "m_curl": round(m_curl[t], 2), "w_curl": round(w_curl[t], 2),
            "vel": round(m_vel[t], 2), "score": round(final_score, 2), "status": status
        })
    
    return sorted(results, key=lambda x: x['score'], reverse=True)

# ================== ROUTES ==================

@app.route('/')
def dashboard():
    data = get_nse_rrg_data()
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>NSE RRG Dashboard</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { background: #f8f9fa; padding: 25px; font-family: 'Segoe UI', sans-serif; }
            .card { border: none; box-shadow: 0 4px 15px rgba(0,0,0,0.05); border-radius: 12px; }
            .status-buy { color: #198754; font-weight: bold; background: #e8f5e9; padding: 4px 10px; border-radius: 20px; }
            .status-exit { color: #dc3545; font-weight: bold; background: #ffebee; padding: 4px 10px; border-radius: 20px; }
            .badge-leading { background-color: #198754; } .badge-improving { background-color: #0d6efd; }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="d-flex justify-content-between align-items-center mb-4">
                <h2>🚀 NSE RRG Sector Engine</h2>
                <span class="text-muted">Updated: {{ now }}</span>
            </div>
            <div class="card p-4">
                <table class="table table-hover">
                    <thead class="table-dark">
                        <tr>
                            <th>Ticker</th><th>Sector</th><th>Quadrant</th><th>M_Curl</th><th>Score</th><th>Status</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for item in results %}
                        <tr>
                            <td><strong>{{ item.ticker }}</strong></td>
                            <td>{{ item.sector }}</td>
                            <td><span class="badge {{ 'badge-leading' if item.quadrant=='Leading' else 'badge-improving' if item.quadrant=='Improving' else 'bg-secondary' }}">
                                {{ item.quadrant }}</span></td>
                            <td>{{ item.m_curl }}</td>
                            <td><strong>{{ item.score }}</strong></td>
                            <td><span class="{{ 'status-buy' if 'BUY' in item.status else 'status-exit' if 'EXIT' in item.status else '' }}">
                                {{ item.status }}</span></td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, results=data, now=datetime.now().strftime('%H:%M IST'))

@app.route('/api/v1/signals')
def api_signals():
    try:
        data = get_nse_rrg_data()
        top_picks = [i for i in data if i['score'] > 6.0][:2]
        return jsonify({
            "timestamp": datetime.now().isoformat(),
            "all_results": data,
            "top_conviction_picks": top_picks
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=8080)