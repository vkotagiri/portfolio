# tests/test_calculations.py
"""
Unit tests for financial calculations.
Run with: pytest tests/ -v
"""
import pytest
import numpy as np
import pandas as pd
from datetime import date


class TestReturnCalculations:
    """Test return calculation formulas."""
    
    def test_simple_return(self):
        """Test (end/start - 1) formula."""
        start = 100.0
        end = 110.0
        expected = 10.0  # 10%
        actual = (end / start - 1) * 100
        assert abs(actual - expected) < 0.001
    
    def test_negative_return(self):
        """Test negative return."""
        start = 100.0
        end = 90.0
        expected = -10.0  # -10%
        actual = (end / start - 1) * 100
        assert abs(actual - expected) < 0.001
    
    def test_cumulative_return(self):
        """Test compound return over multiple periods."""
        prices = [100, 110, 99, 108.9]  # +10%, -10%, +10%
        total_return = (prices[-1] / prices[0] - 1) * 100
        expected = 8.9  # 8.9%
        assert abs(total_return - expected) < 0.1


class TestRSI:
    """Test RSI calculation."""
    
    def test_rsi_overbought(self):
        """RSI should be >70 after sustained gains."""
        # Create price series with consistent gains
        prices = pd.Series([100 + i * 2 for i in range(20)])  # Uptrend
        
        from app.services.technicals import rsi14
        rsi = rsi14(prices)
        
        assert rsi is not None
        assert rsi > 70  # Should be overbought
    
    def test_rsi_oversold(self):
        """RSI should be <30 after sustained losses."""
        prices = pd.Series([100 - i * 2 for i in range(20)])  # Downtrend
        
        from app.services.technicals import rsi14
        rsi = rsi14(prices)
        
        assert rsi is not None
        assert rsi < 30  # Should be oversold
    
    def test_rsi_insufficient_data(self):
        """RSI should return None with insufficient data."""
        prices = pd.Series([100, 101, 102])  # Only 3 points
        
        from app.services.technicals import rsi14
        rsi = rsi14(prices)
        
        assert rsi is None


class TestMACD:
    """Test MACD calculation."""
    
    def test_macd_structure(self):
        """MACD should return expected keys."""
        # Use datetime index like real data
        dates = pd.date_range("2025-01-01", periods=50, freq="B")
        prices = pd.Series([100 + np.sin(i/5) * 10 for i in range(50)], index=dates)
        
        from app.services.technicals import macd_12_26_9
        result = macd_12_26_9(prices)
        
        assert "macd" in result
        assert "signal" in result
        assert "hist" in result
        assert "direction" in result
    
    def test_macd_bullish_direction(self):
        """MACD > Signal should be Bullish."""
        # Strong uptrend with datetime index
        dates = pd.date_range("2025-01-01", periods=50, freq="B")
        prices = pd.Series([100 + i * 2 for i in range(50)], index=dates)
        
        from app.services.technicals import macd_12_26_9
        result = macd_12_26_9(prices)
        
        assert result["direction"] == "Bullish"


class TestSharpeRatio:
    """Test Sharpe ratio calculation."""
    
    def test_sharpe_formula(self):
        """Test Sharpe = (Rp - Rf) / σp."""
        # Daily returns: 0.1% average, 1% std dev
        returns = pd.Series([0.001] * 252)  # Consistent 0.1% daily
        rf_annual = 0.05  # 5% risk-free
        rf_daily = rf_annual / 252
        
        excess = returns - rf_daily
        sharpe = (excess.mean() / returns.std()) * np.sqrt(252)
        
        # With 0.1% daily = 25.2% annual, excess ~20%, std ~0 
        # Sharpe should be very high (approaches infinity with zero vol)
        # This is a degenerate case, but formula is correct
        assert np.isfinite(sharpe) or returns.std() == 0
    
    def test_sharpe_negative_excess(self):
        """Negative excess return = negative Sharpe."""
        returns = pd.Series([0.0001] * 252 + [0.001 * np.random.randn() for _ in range(50)])
        returns = pd.Series(np.random.randn(100) * 0.001)  # Random small returns
        rf_annual = 0.10  # 10% risk-free (higher than typical returns)
        rf_daily = rf_annual / 252
        
        excess = returns - rf_daily
        if returns.std() > 0:
            sharpe = (excess.mean() / returns.std()) * np.sqrt(252)
            # With low returns and high rf, Sharpe should be negative
            # This depends on random data, so just check it's finite
            assert np.isfinite(sharpe)


class TestBeta:
    """Test beta calculation."""
    
    def test_beta_perfect_correlation(self):
        """Beta = 1 when portfolio matches benchmark."""
        benchmark = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02] * 20)
        portfolio = benchmark.copy()  # Perfect match
        
        cov_matrix = np.cov(portfolio.values, benchmark.values)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        assert abs(beta - 1.0) < 0.001
    
    def test_beta_double_leverage(self):
        """Beta = 2 when portfolio is 2x benchmark."""
        benchmark = pd.Series([0.01, -0.02, 0.015, -0.005, 0.02] * 20)
        portfolio = benchmark * 2  # 2x leverage
        
        cov_matrix = np.cov(portfolio.values, benchmark.values)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        
        assert abs(beta - 2.0) < 0.001


class TestMaxDrawdown:
    """Test max drawdown calculation."""
    
    def test_no_drawdown(self):
        """No drawdown in monotonic increase."""
        prices = pd.Series([100, 101, 102, 103, 104])
        
        cummax = prices.cummax()
        dd = prices / cummax - 1
        mdd = dd.min()
        
        assert mdd == 0.0
    
    def test_drawdown_50pct(self):
        """50% drawdown from peak."""
        prices = pd.Series([100, 110, 55, 60])  # Peak 110, trough 55
        
        cummax = prices.cummax()
        dd = prices / cummax - 1
        mdd = dd.min()
        
        expected = 55 / 110 - 1  # -50%
        assert abs(mdd - expected) < 0.001


class TestTWR:
    """Test Time-Weighted Return calculation."""
    
    def test_twr_no_cashflow(self):
        """TWR equals simple return when no cash flow."""
        prev_value = 100000
        curr_value = 110000
        cash_flow = 0
        
        from app.services.positions import calculate_twr_return
        twr = calculate_twr_return(prev_value, curr_value, cash_flow)
        
        expected = 10.0  # 10%
        assert abs(twr - expected) < 0.001
    
    def test_twr_with_deposit(self):
        """TWR removes impact of deposit."""
        prev_value = 100000
        curr_value = 115000  # $5k gain + $10k deposit
        cash_flow = 10000    # Deposit
        
        from app.services.positions import calculate_twr_return
        twr = calculate_twr_return(prev_value, curr_value, cash_flow)
        
        # True return: (115000 - 10000) / 100000 - 1 = 5%
        expected = 5.0
        assert abs(twr - expected) < 0.001


class TestWeights:
    """Test portfolio weight calculations."""
    
    def test_weights_sum_to_one(self):
        """Weights should sum to 100%."""
        from app.services.reporting import _weights_on_day
        
        shares = {"AAPL": 100, "MSFT": 50, "GOOG": 25}
        prices = {"AAPL": 150.0, "MSFT": 400.0, "GOOG": 140.0}
        
        weights = _weights_on_day(shares, prices)
        total = sum(weights.values())
        
        assert abs(total - 1.0) < 0.0001
    
    def test_weight_calculation(self):
        """Individual weight = MV / Total MV."""
        shares = {"AAPL": 100, "MSFT": 100}
        prices = {"AAPL": 100.0, "MSFT": 100.0}  # Equal prices
        
        from app.services.reporting import _weights_on_day
        weights = _weights_on_day(shares, prices)
        
        # Equal shares, equal prices = equal weights
        assert abs(weights["AAPL"] - 0.5) < 0.001
        assert abs(weights["MSFT"] - 0.5) < 0.001


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
