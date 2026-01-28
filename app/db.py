from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase, Session
from .config import settings

engine = create_engine(settings.database_url, future=True, pool_pre_ping=True)
if settings.database_url.startswith("sqlite"):
    with engine.connect() as conn:
        conn.execute(text("PRAGMA foreign_keys=ON;"))
        conn.execute(text("PRAGMA journal_mode=WAL;"))
        conn.execute(text("PRAGMA synchronous=NORMAL;"))
        conn.commit()

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)

class Base(DeclarativeBase):
    pass


@contextmanager
def get_session() -> Generator[Session, None, None]:
    """
    Provide a transactional scope around a series of operations.
    
    Usage:
        with get_session() as sess:
            sess.query(...)
            sess.commit()  # if needed
    
    Session is automatically closed when exiting the context,
    even if an exception occurs.
    """
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()
