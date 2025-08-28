# oanda_adapter.py
import os
import time
import oandapyV20
from oandapyV20.endpoints.pricing import PricingInfo
from oandapyV20.endpoints.orders import OrderCreate
from oandapyV20.endpoints.positions import PositionsClose
from oandapyV20.endpoints.accounts import AccountDetails
import logging

ACCOUNT_ID = os.getenv("OANDA_ACCOUNT_ID", "101-XXX-XXXXXXX-001")
ACCESS_TOKEN = os.getenv("OANDA_ACCESS_TOKEN", "YOUR_OANDA_TOKEN")
INSTRUMENT = os.getenv("OANDA_INSTRUMENT", "EUR_USD")
ENV = os.getenv("OANDA_ENV", "practice")  # 'practice' o 'live'

client = oandapyV20.API(access_token=ACCESS_TOKEN, environment=ENV)
logger = logging.getLogger("oanda_adapter")
logger.setLevel(logging.INFO)

def get_price():
    params = {"instruments": INSTRUMENT}
    r = PricingInfo(accountID=ACCOUNT_ID, params=params)
    client.request(r)
    p = r.response["prices"][0]
    bid = float(p["bids"][0]["price"])
    ask = float(p["asks"][0]["price"])
    return bid, ask

def get_account_equity():
    r = AccountDetails(accountID=ACCOUNT_ID)
    client.request(r)
    acct = r.response.get("account", {})
    return float(acct.get("balance", 0.0))

def place_bracket(side, units, sl=None, tp=None):
    """
    Envía orden de mercado con SL/TP (takeProfitOnFill / stopLossOnFill)
    units: integer (positive)
    side: 'buy' o 'sell'
    """
    u = int(units) if side == "buy" else -int(units)
    order = {
        "order": {
            "instrument": INSTRUMENT,
            "units": str(u),
            "type": "MARKET",
            "timeInForce": "FOK",
            "positionFill": "DEFAULT",
        }
    }
    if tp:
        order["order"]["takeProfitOnFill"] = {"price": f"{tp:.5f}"}
    if sl:
        order["order"]["stopLossOnFill"] = {"price": f"{sl:.5f}"}
    r = OrderCreate(accountID=ACCOUNT_ID, data=order)
    resp = client.request(r)
    return resp

def close_all():
    r = PositionsClose(accountID=ACCOUNT_ID, instrument=INSTRUMENT, data={"longUnits":"ALL","shortUnits":"ALL"})
    client.request(r)

# wrapper safe para retries
def place_bracket_safe(side, units, sl=None, tp=None, retries=2):
    for i in range(retries):
        try:
            return place_bracket(side, units, sl=sl, tp=tp)
        except Exception as e:
            logger.warning("place_bracket failed, retrying: %s", e)
            time.sleep(1.0)
    raise RuntimeError("place_bracket failed after retries")
