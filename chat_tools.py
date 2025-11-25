"""
MCP-style Tools for AI Chatbot
Provides portfolio data access for Claude and other AI models
"""

import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
from pathlib import Path

import yaml


# ============================================================================
# TOOL DEFINITIONS (MCP-style schema for Claude)
# ============================================================================

PORTFOLIO_TOOLS = [
    {
        "name": "get_account_summary",
        "description": "Get a summary of all trading accounts including equity, cash, buying power, and daily P&L. Use this when the user asks about their account balances, total portfolio value, or daily performance.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_name": {
                    "type": "string",
                    "description": "Optional: specific account name to query. If not provided, returns all accounts."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_positions",
        "description": "Get all current positions (holdings) across accounts including symbol, quantity, market value, and unrealized P&L. Use this when the user asks what stocks/ETFs they own or their holdings.",
        "input_schema": {
            "type": "object",
            "properties": {
                "account_name": {
                    "type": "string",
                    "description": "Optional: specific account name to query positions for."
                },
                "symbol": {
                    "type": "string",
                    "description": "Optional: filter positions by specific symbol (e.g., 'VOO', 'QQQ')."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_portfolio_performance",
        "description": "Get portfolio performance metrics including returns, volatility, Sharpe ratio, and max drawdown over a specified period. Use this for performance analysis questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "period": {
                    "type": "string",
                    "enum": ["1D", "1W", "1M", "3M", "1Y"],
                    "description": "Time period for performance calculation. Default is '1M' (1 month)."
                },
                "account_name": {
                    "type": "string",
                    "description": "Optional: specific account name to analyze."
                }
            },
            "required": []
        }
    },
    {
        "name": "get_asset_allocation",
        "description": "Get the current asset allocation breakdown by asset class (Equity, Fixed Income, Real Estate, Commodities, Cash). Use this when the user asks about their portfolio diversification or allocation.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    },
    {
        "name": "get_stock_quote",
        "description": "Get the latest price and daily change for a specific stock or ETF symbol. Use this when the user asks about current prices.",
        "input_schema": {
            "type": "object",
            "properties": {
                "symbol": {
                    "type": "string",
                    "description": "The stock/ETF symbol to look up (e.g., 'AAPL', 'SPY', 'VOO')."
                }
            },
            "required": ["symbol"]
        }
    },
    {
        "name": "get_recent_orders",
        "description": "Get recent trading orders across accounts. Use this when the user asks about their recent trades or order history.",
        "input_schema": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of orders to return. Default is 10."
                },
                "status": {
                    "type": "string",
                    "enum": ["all", "open", "closed", "filled", "canceled"],
                    "description": "Filter by order status. Default is 'all'."
                }
            },
            "required": []
        }
    },
    {
        "name": "compare_to_benchmark",
        "description": "Compare portfolio performance to a benchmark index (SPY, QQQ, etc.) over a specified period. Use this for relative performance questions.",
        "input_schema": {
            "type": "object",
            "properties": {
                "benchmark": {
                    "type": "string",
                    "enum": ["SPY", "QQQ", "IWM", "DIA", "VTI"],
                    "description": "Benchmark symbol to compare against. Default is 'SPY'."
                },
                "period": {
                    "type": "string",
                    "enum": ["1D", "1W", "1M", "3M", "1Y"],
                    "description": "Time period for comparison. Default is '1M'."
                }
            },
            "required": []
        }
    }
]


# ============================================================================
# TOOL IMPLEMENTATIONS
# ============================================================================

def load_accounts_from_config(config_path: str = "portfolio_config.yaml") -> List[Dict[str, Any]]:
    """Load account configurations from YAML file."""
    config_file = Path(config_path)
    if not config_file.exists():
        return []
    
    raw = yaml.safe_load(config_file.read_text())
    accounts = []
    for entry in raw.get("accounts", []):
        accounts.append({
            "name": entry["name"],
            "key_id": entry["key_id"],
            "secret_key": entry["secret_key"],
            "base_url": entry.get("base_url", "https://paper-api.alpaca.markets"),
        })
    return accounts


def get_rest_client(account: Dict[str, Any]):
    """Create Alpaca REST client for an account."""
    from alpaca_trade_api.rest import REST
    return REST(
        key_id=account["key_id"],
        secret_key=account["secret_key"],
        base_url=account["base_url"]
    )


def tool_get_account_summary(account_name: Optional[str] = None) -> Dict[str, Any]:
    """Get account summary for all or specific account."""
    accounts = load_accounts_from_config()
    if not accounts:
        return {"error": "No accounts configured"}
    
    summaries = []
    total_equity = 0
    total_cash = 0
    total_day_pnl = 0
    
    for account in accounts:
        if account_name and account["name"] != account_name:
            continue
            
        try:
            rest = get_rest_client(account)
            acc_info = rest.get_account()
            positions = rest.list_positions()
            
            equity = float(acc_info.equity)
            cash = float(acc_info.cash)
            last_equity = float(acc_info.last_equity)
            day_pnl = equity - last_equity
            day_pnl_pct = (day_pnl / last_equity * 100) if last_equity > 0 else 0
            
            total_equity += equity
            total_cash += cash
            total_day_pnl += day_pnl
            
            summaries.append({
                "account": account["name"],
                "equity": f"${equity:,.2f}",
                "cash": f"${cash:,.2f}",
                "buying_power": f"${float(acc_info.buying_power):,.2f}",
                "positions_count": len(positions),
                "day_pnl": f"${day_pnl:+,.2f}",
                "day_pnl_percent": f"{day_pnl_pct:+.2f}%"
            })
        except Exception as e:
            summaries.append({"account": account["name"], "error": str(e)})
    
    return {
        "accounts": summaries,
        "total": {
            "total_equity": f"${total_equity:,.2f}",
            "total_cash": f"${total_cash:,.2f}",
            "total_day_pnl": f"${total_day_pnl:+,.2f}"
        }
    }


def tool_get_positions(account_name: Optional[str] = None, symbol: Optional[str] = None) -> Dict[str, Any]:
    """Get all positions across accounts."""
    accounts = load_accounts_from_config()
    if not accounts:
        return {"error": "No accounts configured"}
    
    all_positions = []
    
    for account in accounts:
        if account_name and account["name"] != account_name:
            continue
            
        try:
            rest = get_rest_client(account)
            positions = rest.list_positions()
            
            for pos in positions:
                if symbol and pos.symbol != symbol.upper():
                    continue
                    
                all_positions.append({
                    "account": account["name"],
                    "symbol": pos.symbol,
                    "qty": float(pos.qty),
                    "market_value": f"${float(pos.market_value):,.2f}",
                    "avg_entry_price": f"${float(pos.avg_entry_price):,.2f}",
                    "current_price": f"${float(pos.current_price):,.2f}",
                    "unrealized_pnl": f"${float(pos.unrealized_pl):+,.2f}",
                    "unrealized_pnl_percent": f"{float(pos.unrealized_plpc) * 100:+.2f}%",
                    "today_change": f"{float(pos.change_today) * 100:+.2f}%"
                })
        except Exception as e:
            pass
    
    # Group by symbol for summary
    symbol_totals = {}
    for pos in all_positions:
        sym = pos["symbol"]
        if sym not in symbol_totals:
            symbol_totals[sym] = {"total_value": 0, "count": 0}
        # Parse the market value string back to float
        value = float(pos["market_value"].replace("$", "").replace(",", ""))
        symbol_totals[sym]["total_value"] += value
        symbol_totals[sym]["count"] += 1
    
    return {
        "positions": all_positions,
        "summary": {
            "total_positions": len(all_positions),
            "unique_symbols": len(symbol_totals)
        }
    }


def tool_get_portfolio_performance(period: str = "1M", account_name: Optional[str] = None) -> Dict[str, Any]:
    """Get portfolio performance metrics."""
    import numpy as np
    
    accounts = load_accounts_from_config()
    if not accounts:
        return {"error": "No accounts configured"}
    
    timeframe_map = {"1D": "5Min", "1W": "1H", "1M": "1D", "3M": "1D", "1Y": "1D"}
    timeframe = timeframe_map.get(period, "1D")
    
    results = []
    
    for account in accounts:
        if account_name and account["name"] != account_name:
            continue
            
        try:
            rest = get_rest_client(account)
            history = rest.get_portfolio_history(period=period, timeframe=timeframe)
            
            equity = list(history.equity)
            if len(equity) >= 2:
                returns = np.diff(equity) / np.array(equity[:-1])
                
                total_return = (equity[-1] - equity[0]) / equity[0] * 100
                volatility = np.std(returns) * np.sqrt(252) * 100
                sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
                
                cumulative = np.cumprod(1 + returns)
                running_max = np.maximum.accumulate(cumulative)
                drawdown = (cumulative - running_max) / running_max
                max_drawdown = np.min(drawdown) * 100
                
                results.append({
                    "account": account["name"],
                    "period": period,
                    "total_return": f"{total_return:+.2f}%",
                    "annualized_volatility": f"{volatility:.2f}%",
                    "sharpe_ratio": f"{sharpe:.2f}",
                    "max_drawdown": f"{max_drawdown:.2f}%",
                    "start_value": f"${equity[0]:,.2f}",
                    "end_value": f"${equity[-1]:,.2f}"
                })
        except Exception as e:
            results.append({"account": account["name"], "error": str(e)})
    
    return {"performance": results}


def tool_get_asset_allocation() -> Dict[str, Any]:
    """Get asset allocation breakdown."""
    accounts = load_accounts_from_config()
    if not accounts:
        return {"error": "No accounts configured"}
    
    ASSET_CLASSES = {
        "Equity": ["VOO", "QQQ", "VEA", "VWO", "VTI", "SPY"],
        "Fixed Income": ["VTEB", "TIP", "IEF", "SHYG", "BND", "AGG"],
        "Real Estate": ["VNQ", "VNQI"],
        "Commodities": ["GLD", "SLV", "IAU"],
    }
    
    allocation = {
        "Equity": 0.0,
        "Fixed Income": 0.0,
        "Real Estate": 0.0,
        "Commodities": 0.0,
        "Other": 0.0,
        "Cash": 0.0
    }
    
    for account in accounts:
        try:
            rest = get_rest_client(account)
            positions = rest.list_positions()
            acc_info = rest.get_account()
            
            allocation["Cash"] += float(acc_info.cash)
            
            for pos in positions:
                value = float(pos.market_value)
                symbol = pos.symbol
                
                classified = False
                for asset_class, symbols in ASSET_CLASSES.items():
                    if symbol in symbols:
                        allocation[asset_class] += value
                        classified = True
                        break
                
                if not classified:
                    allocation["Other"] += value
        except:
            pass
    
    total = sum(allocation.values())
    
    return {
        "allocation": {k: f"${v:,.2f}" for k, v in allocation.items()},
        "percentages": {k: f"{(v/total*100):.1f}%" for k, v in allocation.items()} if total > 0 else {},
        "total_value": f"${total:,.2f}"
    }


def tool_get_stock_quote(symbol: str) -> Dict[str, Any]:
    """Get latest quote for a symbol."""
    accounts = load_accounts_from_config()
    if not accounts:
        return {"error": "No accounts configured"}
    
    try:
        rest = get_rest_client(accounts[0])
        trade = rest.get_latest_trade(symbol.upper())
        bars = rest.get_bars(symbol.upper(), "1D", limit=2).df
        
        current_price = float(trade.price)
        
        if not bars.empty and len(bars) >= 2:
            prev_close = bars.iloc[-2]["close"]
            change = current_price - prev_close
            change_pct = (change / prev_close) * 100
        else:
            change = 0
            change_pct = 0
        
        return {
            "symbol": symbol.upper(),
            "price": f"${current_price:,.2f}",
            "change": f"${change:+,.2f}",
            "change_percent": f"{change_pct:+.2f}%",
            "timestamp": trade.timestamp.strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        return {"error": f"Could not get quote for {symbol}: {str(e)}"}


def tool_get_recent_orders(limit: int = 10, status: str = "all") -> Dict[str, Any]:
    """Get recent orders across accounts."""
    accounts = load_accounts_from_config()
    if not accounts:
        return {"error": "No accounts configured"}
    
    all_orders = []
    
    for account in accounts:
        try:
            rest = get_rest_client(account)
            orders = rest.list_orders(status=status if status != "all" else None, limit=limit)
            
            for order in orders[:limit]:
                all_orders.append({
                    "account": account["name"],
                    "symbol": order.symbol,
                    "side": order.side,
                    "qty": order.qty,
                    "type": order.type,
                    "status": order.status,
                    "filled_avg_price": f"${float(order.filled_avg_price):,.2f}" if order.filled_avg_price else "N/A",
                    "submitted_at": order.submitted_at.strftime("%Y-%m-%d %H:%M") if order.submitted_at else "N/A"
                })
        except:
            pass
    
    return {"orders": all_orders[:limit]}


def tool_compare_to_benchmark(benchmark: str = "SPY", period: str = "1M") -> Dict[str, Any]:
    """Compare portfolio to benchmark."""
    import numpy as np
    
    accounts = load_accounts_from_config()
    if not accounts:
        return {"error": "No accounts configured"}
    
    timeframe_map = {"1D": "5Min", "1W": "1H", "1M": "1D", "3M": "1D", "1Y": "1D"}
    timeframe = timeframe_map.get(period, "1D")
    
    try:
        rest = get_rest_client(accounts[0])
        
        # Get benchmark data
        now = datetime.now()
        period_days = {"1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 365}
        start = now - timedelta(days=period_days.get(period, 30))
        
        bars = rest.get_bars(benchmark, timeframe, start=start.isoformat(), end=now.isoformat()).df
        
        if not bars.empty:
            benchmark_return = (bars["close"].iloc[-1] - bars["close"].iloc[0]) / bars["close"].iloc[0] * 100
        else:
            benchmark_return = 0
        
        # Get portfolio returns
        portfolio_returns = []
        for account in accounts:
            try:
                rest_acc = get_rest_client(account)
                history = rest_acc.get_portfolio_history(period=period, timeframe=timeframe)
                equity = list(history.equity)
                if len(equity) >= 2:
                    ret = (equity[-1] - equity[0]) / equity[0] * 100
                    portfolio_returns.append({"account": account["name"], "return": ret})
            except:
                pass
        
        avg_portfolio_return = np.mean([p["return"] for p in portfolio_returns]) if portfolio_returns else 0
        
        return {
            "period": period,
            "benchmark": {
                "symbol": benchmark,
                "return": f"{benchmark_return:+.2f}%"
            },
            "portfolio": {
                "accounts": [{**p, "return": f"{p['return']:+.2f}%"} for p in portfolio_returns],
                "average_return": f"{avg_portfolio_return:+.2f}%"
            },
            "outperformance": f"{(avg_portfolio_return - benchmark_return):+.2f}%"
        }
    except Exception as e:
        return {"error": str(e)}


# ============================================================================
# TOOL EXECUTOR
# ============================================================================

def execute_tool(tool_name: str, tool_input: Dict[str, Any]) -> str:
    """Execute a tool and return the result as a string."""
    tool_functions = {
        "get_account_summary": tool_get_account_summary,
        "get_positions": tool_get_positions,
        "get_portfolio_performance": tool_get_portfolio_performance,
        "get_asset_allocation": tool_get_asset_allocation,
        "get_stock_quote": tool_get_stock_quote,
        "get_recent_orders": tool_get_recent_orders,
        "compare_to_benchmark": tool_compare_to_benchmark,
    }
    
    if tool_name not in tool_functions:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})
    
    try:
        result = tool_functions[tool_name](**tool_input)
        return json.dumps(result, indent=2)
    except Exception as e:
        return json.dumps({"error": str(e)})

