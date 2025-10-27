from sqlalchemy.orm import Session
from sqlalchemy import select
from ..models import Holding, Security

def upsert_holdings(sess: Session, rows: list[dict]):
    for r in rows:
        t = r["ticker"].upper()
        if not sess.get(Security, t):
            sess.add(Security(ticker=t))
        h = sess.get(Holding, t)
        avg_cost = r.get("avg_cost")
        if avg_cost in (None, "", "Data not available"):
            avg_cost = None
        else:
            avg_cost = float(avg_cost)
        if h:
            h.shares = float(r["shares"])
            h.avg_cost = avg_cost
        else:
            sess.add(Holding(ticker=t, shares=float(r["shares"]), avg_cost=avg_cost))

def all_holdings(sess: Session) -> list[Holding]:
    return list(sess.scalars(select(Holding)).all())
