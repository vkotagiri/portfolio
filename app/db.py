from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, DeclarativeBase
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

def get_session():
    return SessionLocal()
