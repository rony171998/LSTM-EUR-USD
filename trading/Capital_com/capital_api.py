# capital_api.py
"""
Módulo de API para interactuar con Capital.com
"""
import os
import requests
from typing import Dict, List, Optional
import json

# Add default fallback for BASE_URL
BASE_URL = os.getenv("CAPITAL_url", "https://demo-api-capital.backend-capital.com/api/v1")
API_KEY = os.getenv("CAPITAL_X-CAP-API-KEY", "1BBMbF03iQD9aNHJ")
API_PASSWORD = os.getenv("CAPITAL_password", "RONY17abril1998&")
IDENTIFIER = os.getenv("CAPITAL_identifier", "rony171998@gmail.com")

# Validate required environment variables
if not API_KEY or not API_PASSWORD:
    raise EnvironmentError("❌ Missing required environment variables: CAPITAL_X-CAP-API-KEY or CAPITAL_password")

if not BASE_URL:
    raise EnvironmentError("❌ Missing required environment variable: CAPITAL_url")

_session = requests.Session()
CST = None
SECURITY_TOKEN = None

def login():
    """Inicia sesión en Capital.com y obtiene los tokens de autenticación"""
    global CST, SECURITY_TOKEN
    headers = {
        "X-CAP-API-KEY": API_KEY,
        "Content-Type": "application/json"
    }
    payload = json.dumps({
        "identifier": IDENTIFIER,
        "password": API_PASSWORD,
        "encryptedPassword": False
    })

    print("🔄 Iniciando sesión en Capital.com...")
    print(f"📤 Payload: {payload}")
    print(f"📤 Headers: {headers}")

    resp = _session.post(f"{BASE_URL}/session", headers=headers, data=payload)
    print(f"📥 Response Status Code: {resp.status_code}")
    print(f"📥 Response Body: {resp.text}")

    resp.raise_for_status()
    CST = resp.headers.get("CST")
    SECURITY_TOKEN = resp.headers.get("X-SECURITY-TOKEN")
    if not (CST and SECURITY_TOKEN):
        raise RuntimeError("No se recibieron tokens CST o security.")
    print("✅ Sesión iniciada en Capital.com")

def auth_headers():
    """Retorna los headers de autenticación necesarios"""
    return {
        "CST": CST,
        "X-SECURITY-TOKEN": SECURITY_TOKEN,
        "Content-Type": "application/json"
    }

def get_account_info() -> Dict:
    """Obtiene información de la cuenta"""
    url = f"{BASE_URL}/accounts"
    resp = _session.get(url, headers=auth_headers())
    resp.raise_for_status()
    accounts = resp.json()['accounts']
    if accounts:
        return accounts[0]  # Retorna la primera cuenta
    raise RuntimeError("No se encontraron cuentas")

def get_market_data(epic: str) -> Dict:
    """Obtiene datos del mercado para un instrumento específico"""
    url = f"{BASE_URL}/markets/{epic}"
    resp = _session.get(url, headers=auth_headers())
    resp.raise_for_status()
    return resp.json()

def get_prices(epic: str, resolution: str = "HOUR", max_values: int = 100) -> List[Dict]:
    """
    Obtiene precios históricos
    
    Args:
        epic: Identificador del instrumento
        resolution: Resolución temporal (MINUTE, MINUTE_5, MINUTE_15, MINUTE_30, HOUR, HOUR_4, DAY, WEEK)
        max_values: Número de valores a obtener
    """
    url = f"{BASE_URL}/prices/{epic}"
    params = {
        "resolution": resolution,
        "max": max_values
    }
    resp = _session.get(url, headers=auth_headers(), params=params)
    resp.raise_for_status()
    return resp.json()['prices']

def get_open_positions() -> List[Dict]:
    """Obtiene las posiciones abiertas"""
    url = f"{BASE_URL}/positions"
    resp = _session.get(url, headers=auth_headers())
    resp.raise_for_status()
    return resp.json()['positions']

def place_order(epic: str, direction: str, size: float, stop_price: float, profit_price: float) -> Dict:
    """
    Coloca una orden en el mercado
    
    Args:
        epic: Identificador del instrumento
        direction: Dirección de la operación (BUY/SELL)
        size: Tamaño de la posición
        stop_price: Precio de stop loss
        profit_price: Precio de take profit
    """
    url = f"{BASE_URL}/positions"
    payload = {
        "epic": epic,
        "direction": direction.upper(),
        "size": size,
        "guaranteedStop": False,  # Cambiar a True si quieres stop garantizado (tiene costo extra)
        "stopLevel": round(stop_price, 5),
        "profitLevel": round(profit_price, 5)
    }
    resp = _session.post(url, headers=auth_headers(), json=payload)
    resp.raise_for_status()
    return resp.json()

def close_position(deal_id: str) -> Dict:
    """
    Cierra una posición específica
    
    Args:
        deal_id: ID de la posición a cerrar
    """
    url = f"{BASE_URL}/positions/{deal_id}"
    resp = _session.delete(url, headers=auth_headers())
    resp.raise_for_status()
    return resp.json()

def modify_position(deal_id: str, stop_price: Optional[float] = None, 
                   profit_price: Optional[float] = None) -> Dict:
    """
    Modifica una posición existente
    
    Args:
        deal_id: ID de la posición
        stop_price: Nuevo precio de stop loss
        profit_price: Nuevo precio de take profit
    """
    url = f"{BASE_URL}/positions/{deal_id}"
    payload = {}
    if stop_price is not None:
        payload['stopLevel'] = round(stop_price, 5)
    if profit_price is not None:
        payload['profitLevel'] = round(profit_price, 5)
    
    resp = _session.put(url, headers=auth_headers(), json=payload)
    resp.raise_for_status()
    return resp.json()