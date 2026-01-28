# tests/conftest.py
"""
Pytest configuration and fixtures.
"""
import pytest
import os
import tempfile
from pathlib import Path

# Set test environment before importing app modules
os.environ["ENV"] = "test"
os.environ["DATABASE_URL"] = "sqlite:///:memory:"


@pytest.fixture(scope="session")
def test_db():
    """Create a temporary test database."""
    from app.db import engine, Base
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def session(test_db):
    """Provide a database session for each test."""
    from app.db import SessionLocal
    session = SessionLocal()
    yield session
    session.rollback()
    session.close()


@pytest.fixture
def sample_prices():
    """Sample price data for testing."""
    import pandas as pd
    import numpy as np
    
    dates = pd.date_range("2025-01-01", periods=100, freq="B")
    prices = pd.Series(
        100 * np.cumprod(1 + np.random.randn(100) * 0.02),
        index=dates
    )
    return prices


@pytest.fixture
def sample_holdings():
    """Sample holdings for testing."""
    return [
        {"ticker": "AAPL", "shares": 100, "avg_cost": 150.0},
        {"ticker": "MSFT", "shares": 50, "avg_cost": 400.0},
        {"ticker": "GOOG", "shares": 25, "avg_cost": 140.0},
    ]
