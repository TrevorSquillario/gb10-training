
# Testing
```
docker exec -it hermes /bin/bash
cd /mcp_servers

fastmcp list finance/yfinance_mcp.py
fastmcp call finance/yfinance_mcp.py get_news symbol="DELL"
fastmcp call finance/yfinance_mcp.py get_stock_history symbols="DELL"
fastmcp call finance/yfinance_mcp.py get_technical_indicators symbol="DELL" period="1y"

```