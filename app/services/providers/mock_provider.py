from datetime import datetime
import random

class MockProvider:
    name = "mock"

    def _deterministic_price(self, key: str) -> float:
        rnd = random.Random(hash(key) & 0xffffffff)
        return round(100 + rnd.random()*200, 2)

    def fetch_adj_close(self, tickers: list[str], date_str: str) -> list[dict]:
        asof = datetime.utcnow().isoformat()
        rows = []
        for t in tickers:
            price = self._deterministic_price(t+date_str)
            rows.append({
                "ticker": t.upper(),
                "date": date_str,
                "close": price,
                "adj_close": price,  # adjusted == close in mock
                "volume": None,
                "dividend": None,
                "split": None,
                "source": self.name,
                "asof_ts": asof,
            })
        return rows

    def fetch_benchmark(self, symbol: str, date_str: str) -> dict:
        asof = datetime.utcnow().isoformat()
        price = self._deterministic_price(symbol+date_str)
        return {"symbol": symbol, "date": date_str, "adj_close": price, "source": self.name, "asof_ts": asof}
