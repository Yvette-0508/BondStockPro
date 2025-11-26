import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests
import yaml
from alpaca_trade_api.rest import REST
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


# ============== Core Helpers ==============

def load_accounts_from_config(config_path: str = "portfolio_config.yaml") -> List[Dict[str, Any]]:
    config_file = Path(config_path)
    if not config_file.exists():
        return []
    raw = yaml.safe_load(config_file.read_text())
    return [
        {"name": e["name"], "key_id": e["key_id"], "secret_key": e["secret_key"],
         "base_url": e.get("base_url", "https://paper-api.alpaca.markets")}
        for e in raw.get("accounts", [])
    ]


def get_rest_client(account: Optional[Dict] = None) -> Optional[REST]:
    """Get REST client for specified account or first available"""
    if account:
        return REST(key_id=account["key_id"], secret_key=account["secret_key"], base_url=account["base_url"])
    accounts = load_accounts_from_config()
    if not accounts:
        return None
    return REST(key_id=accounts[0]["key_id"], secret_key=accounts[0]["secret_key"], base_url=accounts[0]["base_url"])


# ============== Dashboard API Routes ==============

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/accounts")
def get_accounts():
    return jsonify([{"name": acc["name"]} for acc in load_accounts_from_config()])


@app.route("/api/portfolio-history/<period>")
def portfolio_history(period: str = "1M"):
    import numpy as np
    
    def calc_metrics(equity_values):
        if len(equity_values) < 2:
            return {"volatility": 0.0, "sharpe_ratio": 0.0, "max_drawdown": 0.0}
        arr = np.array(equity_values, dtype=float)
        returns = np.diff(arr) / arr[:-1]
        vol = np.std(returns) * np.sqrt(252) * 100
        sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0.0
        cumul = np.cumprod(1 + returns)
        dd = (cumul - np.maximum.accumulate(cumul)) / np.maximum.accumulate(cumul)
        return {"volatility": float(vol), "sharpe_ratio": float(sharpe), "max_drawdown": float(np.min(dd) * 100)}

    timeframe_map = {"1D": "5Min", "1W": "1H", "1M": "1D", "3M": "1D", "1Y": "1D", "all": "1W"}
    timeframe = timeframe_map.get(period, "1D")
    accounts = load_accounts_from_config()
    if not accounts:
        return jsonify({"error": "No accounts configured"}), 404

    histories = []
    for account in accounts:
        rest = get_rest_client(account)
        try:
            history = rest.get_portfolio_history(period=period, timeframe=timeframe)
            acc_info = rest.get_account()
            metrics = calc_metrics(history.equity)
            eq = list(history.equity)
            init = eq[0] if eq else 0
            histories.append({
                "name": account["name"],
                "timestamps": [datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M") for ts in history.timestamp],
                "equity": history.equity, "pnl": [(e - init) for e in eq],
                "pnl_pct": [((e - init) / init * 100) if init > 0 else 0 for e in eq],
                "current_equity": float(acc_info.equity), "current_cash": float(acc_info.cash),
                "buying_power": float(acc_info.buying_power), **metrics
            })
        except Exception as e:
            histories.append({"name": account["name"], "error": str(e), "timestamps": [], "equity": [], "pnl": [], "pnl_pct": []})

    # Benchmark
    benchmark_data = {}
    benchmark_symbol = request.args.get('benchmark', 'SPY')
    if benchmark_symbol != 'None':
        rest = get_rest_client()
        if rest:
            try:
                period_days = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 365}.get(period, 365*5)
                start = datetime.now() - timedelta(days=period_days)
                bars = rest.get_bars(benchmark_symbol, timeframe, start=start.isoformat(), limit=10000).df
                if not bars.empty:
                    benchmark_data = {"symbol": benchmark_symbol, "timestamps": [ts.strftime("%Y-%m-%d %H:%M") for ts in bars.index], "close": bars["close"].tolist()}
            except Exception:
                pass

    return jsonify({"accounts": histories, "benchmark": benchmark_data})


@app.route("/api/account-summary")
def account_summary():
    accounts = load_accounts_from_config()
    summaries = []
    rest = None
    for account in accounts:
        rest = get_rest_client(account)
        try:
            acc = rest.get_account()
            positions = rest.list_positions()
            day_pnl = float(acc.equity) - float(acc.last_equity)
            summaries.append({
                "name": account["name"], "equity": float(acc.equity), "cash": float(acc.cash),
                "buying_power": float(acc.buying_power), "portfolio_value": float(acc.portfolio_value),
                "positions_count": len(positions), "day_profit_loss": day_pnl,
                "day_profit_loss_pct": (day_pnl / float(acc.last_equity) * 100) if float(acc.last_equity) > 0 else 0
            })
        except Exception as e:
            summaries.append({"name": account["name"], "error": str(e)})
    
    # Add SPY benchmark
    if rest:
        try:
            spy_trade = rest.get_latest_trade("SPY")
            spy_bars = rest.get_bars("SPY", "1D", limit=2).df
            if len(spy_bars) >= 2:
                prev = spy_bars.iloc[-2]["close"]
                curr = float(spy_trade.price)
                summaries.append({"name": "S&P 500 (SPY)", "equity": curr, "day_profit_loss": curr - prev,
                                  "day_profit_loss_pct": (curr - prev) / prev * 100, "is_market": True})
        except Exception:
            pass
    return jsonify(summaries)


@app.route("/api/risk-metrics")
def risk_metrics():
    ASSET_MAP = {"VOO": "Equity", "QQQ": "Equity", "VEA": "Equity", "VWO": "Equity",
                 "VTEB": "Fixed Income", "TIP": "Fixed Income", "IEF": "Fixed Income", "SHYG": "Fixed Income", "BND": "Fixed Income",
                 "VNQ": "Real Estate", "GLD": "Commodities"}
    allocation = {"Equity": 0.0, "Fixed Income": 0.0, "Real Estate": 0.0, "Commodities": 0.0, "Other": 0.0, "Cash": 0.0}
    all_positions = []

    for account in load_accounts_from_config():
        rest = get_rest_client(account)
        try:
            positions = rest.list_positions()
            allocation["Cash"] += float(rest.get_account().cash)
            for pos in positions:
                mv = float(pos.market_value)
                cls = ASSET_MAP.get(pos.symbol, "Other")
                allocation[cls] += mv
                all_positions.append({"symbol": pos.symbol, "market_value": mv,
                                      "pl_pct": float(pos.unrealized_plpc) * 100,
                                      "pl_day_pct": float(pos.change_today) * 100, "account": account["name"]})
        except Exception:
            pass

    return jsonify({
        "allocation": allocation,
        "top_gainers": sorted(all_positions, key=lambda x: x["pl_day_pct"], reverse=True)[:5],
        "top_losers": sorted(all_positions, key=lambda x: x["pl_day_pct"])[:5]
    })


# ============== Chatbot with Function Calling ==============

ALPACA_TOOLS = [
    {"type": "function", "function": {"name": "get_account_info", "description": "Get account balance, equity, cash, buying power, and daily P/L",
        "parameters": {"type": "object", "properties": {"account_name": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_positions", "description": "Get all current stock positions/holdings with quantities, prices, and P/L",
        "parameters": {"type": "object", "properties": {"account_name": {"type": "string"}}, "required": []}}},
    {"type": "function", "function": {"name": "get_stock_quote", "description": "Get the latest price quote for a stock symbol",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}}, "required": ["symbol"]}}},
    {"type": "function", "function": {"name": "place_order", "description": "Place a market order to buy or sell stocks",
        "parameters": {"type": "object", "properties": {"symbol": {"type": "string"}, "qty": {"type": "integer"}, "side": {"type": "string", "enum": ["buy", "sell"]}}, "required": ["symbol", "qty", "side"]}}},
    {"type": "function", "function": {"name": "get_orders", "description": "Get recent orders",
        "parameters": {"type": "object", "properties": {"status": {"type": "string", "enum": ["open", "closed", "all"]}}, "required": []}}},
    {"type": "function", "function": {"name": "get_market_status", "description": "Check if the stock market is open or closed",
        "parameters": {"type": "object", "properties": {}, "required": []}}}
]


def execute_tool(name: str, args: dict) -> str:
    accounts = load_accounts_from_config()
    if not accounts:
        return "No trading accounts configured."
    
    if name == "get_account_info":
        results = []
        for acc in accounts:
            if args.get("account_name") and acc["name"] != args["account_name"]:
                continue
            try:
                rest = get_rest_client(acc)
                info = rest.get_account()
                pnl = float(info.equity) - float(info.last_equity)
                results.append(f"Account: {acc['name']}\n• Equity: ${float(info.equity):,.2f}\n• Cash: ${float(info.cash):,.2f}\n• Buying Power: ${float(info.buying_power):,.2f}\n• Day P/L: ${pnl:+,.2f}")
            except Exception as e:
                results.append(f"{acc['name']}: Error - {e}")
        return "\n\n".join(results) or "Account not found."

    elif name == "get_positions":
        results = []
        for acc in accounts:
            if args.get("account_name") and acc["name"] != args["account_name"]:
                continue
            try:
                positions = get_rest_client(acc).list_positions()
                if not positions:
                    results.append(f"{acc['name']}: No positions")
                else:
                    lines = [f"{acc['name']} - {len(positions)} positions:"]
                    for p in positions:
                        lines.append(f"• {p.symbol}: {p.qty} @ ${float(p.current_price):,.2f} | P/L: {float(p.unrealized_plpc)*100:+.2f}%")
                    results.append("\n".join(lines))
            except Exception as e:
                results.append(f"{acc['name']}: Error - {e}")
        return "\n\n".join(results) or "Account not found."

    elif name == "get_stock_quote":
        try:
            rest = get_rest_client()
            symbol = args["symbol"].upper()
            price = float(rest.get_latest_trade(symbol).price)
            bars = rest.get_bars(symbol, "1D", limit=2).df
            if len(bars) >= 2:
                chg = price - bars.iloc[-2]["close"]
                return f"{symbol}: ${price:,.2f} ({chg:+.2f}, {chg/bars.iloc[-2]['close']*100:+.2f}%)"
            return f"{symbol}: ${price:,.2f}"
        except Exception as e:
            return f"Error: {e}"

    elif name == "place_order":
        try:
            order = get_rest_client().submit_order(symbol=args["symbol"].upper(), qty=args["qty"], side=args["side"], type="market", time_in_force="day")
            return f"✅ Order placed: {args['side'].upper()} {args['qty']} {args['symbol'].upper()} | ID: {order.id}"
        except Exception as e:
            return f"❌ Order failed: {e}"

    elif name == "get_orders":
        results = []
        status = args.get("status", "open")
        for acc in accounts:
            try:
                orders = get_rest_client(acc).list_orders(status=status, limit=10)
                if not orders:
                    results.append(f"{acc['name']}: No {status} orders")
                else:
                    lines = [f"{acc['name']} - {status} orders:"]
                    for o in orders:
                        lines.append(f"• {o.side.upper()} {o.qty} {o.symbol} | {o.status}")
                    results.append("\n".join(lines))
            except Exception as e:
                results.append(f"{acc['name']}: Error - {e}")
        return "\n\n".join(results)

    elif name == "get_market_status":
        try:
            clock = get_rest_client().get_clock()
            return f"Market: {'🟢 OPEN' if clock.is_open else '🔴 CLOSED'}\nNext Open: {clock.next_open}\nNext Close: {clock.next_close}"
        except Exception as e:
            return f"Error: {e}"

    return f"Unknown tool: {name}"


def call_llm_with_tools(message: str, history: list, api_key: str, api_url: str, model: str) -> str:
    """Unified LLM caller with function calling for DeepSeek/Qwen"""
    system = "You are a portfolio assistant with trading tools. Use them when needed. Be concise."
    messages = [{"role": "system", "content": system}]
    messages.extend([{"role": m["role"], "content": m["content"]} for m in history[-10:]])
    messages.append({"role": "user", "content": message})

    resp = requests.post(api_url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                         json={"model": model, "messages": messages, "tools": ALPACA_TOOLS, "tool_choice": "auto", "max_tokens": 1024}, timeout=90)
    resp.raise_for_status()
    result = resp.json()
    asst_msg = result["choices"][0]["message"]

    if asst_msg.get("tool_calls"):
        messages.append(asst_msg)
        for tc in asst_msg["tool_calls"]:
            tool_result = execute_tool(tc["function"]["name"], json.loads(tc["function"]["arguments"]))
            messages.append({"role": "tool", "tool_call_id": tc["id"], "content": tool_result})
        resp = requests.post(api_url, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
                             json={"model": model, "messages": messages, "max_tokens": 1024}, timeout=90)
        resp.raise_for_status()
        result = resp.json()

    return result["choices"][0]["message"]["content"]


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json()
    message, model, history = data.get("message", ""), data.get("model", "deepseek"), data.get("history", [])

    MODEL_CONFIG = {
        "deepseek": ("DEEPSEEK_API_KEY", "https://api.deepseek.com/chat/completions", "deepseek-chat"),
        "qwen3-max": ("DASHSCOPE_API_KEY", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions", "qwen-max"),
    }

    if model in MODEL_CONFIG:
        env_key, api_url, model_name = MODEL_CONFIG[model]
        api_key = os.environ.get(env_key)
        if not api_key:
            return jsonify({"response": f"⚠️ {env_key} not configured."})
        try:
            return jsonify({"response": call_llm_with_tools(message, history, api_key, api_url, model_name)})
        except Exception as e:
            return jsonify({"response": f"Error: {e}"}), 500

    # GPT fallback (no function calling)
    if model == "gpt-4":
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return jsonify({"response": "⚠️ OPENAI_API_KEY not configured."})
        try:
            context = "You are a portfolio assistant.\n" + "\n".join([f"{'User' if m['role']=='user' else 'Assistant'}: {m['content']}" for m in history[-10:]]) + f"\nUser: {message}"
            resp = requests.post("https://api.openai.com/v1/responses", headers={"Authorization": f"Bearer {api_key}"},
                                 json={"model": "gpt-5.1", "input": context}, timeout=90)
            resp.raise_for_status()
            return jsonify({"response": resp.json()["output_text"]})
        except Exception as e:
            return jsonify({"response": f"Error: {e}"}), 500

    return jsonify({"response": "Unknown model."})


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
