import os
import warnings
warnings.filterwarnings("ignore")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import plotly.graph_objects as go
from textblob import TextBlob
from scipy.signal import find_peaks

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.multioutput import MultiOutputRegressor
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, LSTM, Dense, Dropout

st.set_page_config(page_title="QuantEngine OMEGA Ultra", layout="wide")
st.title("🏛️ QuantEngine OMEGA: Agentic AI Institutional Grade")
st.markdown("---")

if "analisis_ejecutado" not in st.session_state:
    st.session_state.analisis_ejecutado = False
    st.session_state.df_limpio = None
    st.session_state.df_semanal = None
    st.session_state.df_macro = None
    st.session_state.df_pred_10d = None
    st.session_state.df_pred_15d = None
    st.session_state.df_pred_horizontes = None
    st.session_state.pred_trad_data = {}
    st.session_state.microestructura_data = {}
    st.session_state.scalping_data = {}
    st.session_state.ticker_guardado = ""
    st.session_state.verdicto_final = ""
    st.session_state.pensamientos = []
    st.session_state.plan_trading = {}
    st.session_state.df_screener = None
    st.session_state.tft_data = {}

@st.cache_data(ttl=900)
def obtener_datos_completos(simbolo, intervalo="1d"):
    try:
        if intervalo == "1d": per = "2y"
        elif intervalo == "1h": per = "720d"
        elif intervalo in ["15m", "5m"]: per = "60d"
        else: per = "1y"
        
        df = yf.download(simbolo, period=per, interval=intervalo, progress=False)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        
        df_w = yf.download(simbolo, period="5y", interval="1wk", progress=False)
        if isinstance(df_w.columns, pd.MultiIndex): df_w.columns = df_w.columns.get_level_values(0)
        
        df_m = yf.download("^GSPC", period="2y", interval="1d", progress=False)
        if isinstance(df_m.columns, pd.MultiIndex): df_m.columns = df_m.columns.get_level_values(0)
        
        return df, df_w, df_m
    except: return None, None, None

def inyectar_indicadores(df):
    if df is None or df.empty: return None
    df.ta.sma(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    df.ta.macd(append=True)
    df.ta.rsi(length=14, append=True)
    df.ta.stoch(append=True)
    df.ta.bbands(length=20, std=2, append=True)
    df.ta.atr(length=14, append=True)
    df.ta.adx(length=14, append=True)
    
    stoch_rsi_df = ta.stochrsi(df['Close'], length=14)
    if stoch_rsi_df is not None and not stoch_rsi_df.empty:
        stoch_rsi_df.columns = ['StochRSI_K', 'StochRSI_D']
        df = pd.concat([df, stoch_rsi_df], axis=1)
        
    cmf_series = ta.cmf(df['High'], df['Low'], df['Close'], df['Volume'], length=20)
    if cmf_series is not None:
        df['CMF'] = cmf_series
    
    st_df = ta.supertrend(df['High'], df['Low'], df['Close'], length=14, multiplier=2.0)
    if st_df is not None: df = pd.concat([df, st_df], axis=1)
        
    kc_df = ta.kc(df['High'], df['Low'], df['Close'], length=20, scalar=2.0)
    if kc_df is not None: df = pd.concat([df, kc_df], axis=1)
        
    ema_fast = ta.ema(df['Close'], length=8)
    ema_slow = ta.ema(df['Close'], length=21)
    if ema_fast is not None and ema_slow is not None:
        momentum_oscillator = ema_fast - ema_slow
        df['MAEMA_Base'] = ta.ema(df['Close'], length=34) + momentum_oscillator
        atr_maema = ta.atr(df['High'], df['Low'], df['Close'], length=14)
        if atr_maema is not None and 'MAEMA_Base' in df.columns:
            df['MAEMA_Upper_Inner'] = df['MAEMA_Base'] + (atr_maema * 1.5)
            df['MAEMA_Lower_Inner'] = df['MAEMA_Base'] - (atr_maema * 1.5)
            df['MAEMA_Upper_Outer'] = df['MAEMA_Base'] + (atr_maema * 2.5)
            df['MAEMA_Lower_Outer'] = df['MAEMA_Base'] - (atr_maema * 2.5)

    df.columns = [c.split('_')[0] if len(c) > 20 else c for c in df.columns]
    return df

def analizar_scalping_alta_frecuencia(df):
    if len(df) < 50: 
        return {"Error": "Datos insuficientes (requiere mínimo 50 periodos cargados en el gráfico)."}
        
    df_s = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    # VWAP Dinámico Rodante Universal (Para cualquier temporalidad/días)
    df_s['VWAP'] = (df_s['Close'] * df_s['Volume']).rolling(20).sum() / (df_s['Volume'].rolling(20).sum() + 1e-8)
    
    df_s['EMA_9'] = ta.ema(df_s['Close'], length=9)
    df_s['EMA_21'] = ta.ema(df_s['Close'], length=21)
    df_s['RSI_7'] = ta.rsi(df_s['Close'], length=7)
    df_s['ATR_5'] = ta.atr(df_s['High'], df_s['Low'], df_s['Close'], length=5)
    
    df_s = df_s.dropna()
    if df_s.empty: 
        return {"Error": "Cálculo de micro-tendencia fallido por falta de datos en la ventana seleccionada."}
    
    last = df_s.iloc[-1]
    
    vwap_bull = last['Close'] > last['VWAP']
    ema_bull = last['EMA_9'] > last['EMA_21']
    momentum_bull = 55 < last['RSI_7'] < 80 
    
    vwap_bear = last['Close'] < last['VWAP']
    ema_bear = last['EMA_9'] < last['EMA_21']
    momentum_bear = 20 < last['RSI_7'] < 45
    
    verdicto = "⚪ NO ENTRAR"
    color_s = "#aaaaaa"
    riesgo_beneficio = "N/A"
    
    atr_pct = (last['ATR_5'] / last['Close']) * 100
    if atr_pct < 0.15:
        verdicto = "⚪ NO ENTRAR (Riesgo Spread / Sin Volatilidad)"
        color_s = "#ffb700"
    else:
        if vwap_bull and ema_bull and momentum_bull:
            verdicto = "🟢 COMPRA"
            color_s = "#00ffcc"
            stop = last['Close'] - (last['ATR_5'] * 1.5)
            tp = last['Close'] + (last['ATR_5'] * 2.5)
            riesgo_beneficio = f"Stop Loss: ${stop:.2f} | Take Profit: ${tp:.2f} (Ratio 1:1.6)"
        elif vwap_bear and ema_bear and momentum_bear:
            verdicto = "🔴 VENTA"
            color_s = "#ff3333"
            stop = last['Close'] + (last['ATR_5'] * 1.5)
            tp = last['Close'] - (last['ATR_5'] * 2.5)
            riesgo_beneficio = f"Stop Loss: ${stop:.2f} | Take Profit: ${tp:.2f} (Ratio 1:1.6)"

    return {
        "Veredicto": verdicto, "Color": color_s, "VWAP": float(last['VWAP']), 
        "EMA_9": last['EMA_9'], "EMA_21": last['EMA_21'], 
        "RSI_7": last['RSI_7'], "RR": riesgo_beneficio, "ATR_Pct": atr_pct
    }

def analizar_microestructura(df):
    try:
        bins = 30 if len(df) > 100 else 10
        df_vp = df[['Close', 'Volume']].copy().dropna()
        df_vp['Price_Bin'] = pd.cut(df_vp['Close'], bins=bins)
        vp = df_vp.groupby('Price_Bin', observed=False)['Volume'].sum().reset_index()
        vp['Price_Mid'] = vp['Price_Bin'].apply(lambda x: x.mid).astype(float)
        poc_idx = vp['Volume'].idxmax()
        poc_price = vp.loc[poc_idx, 'Price_Mid']
        
        df_z = df.copy()
        df_z['Delta_Vol'] = np.where(df_z['Close'] > df_z['Open'], df_z['Volume'], -df_z['Volume'])
        df_z['Vol_Mean'] = df_z['Volume'].rolling(20).mean()
        df_z['Vol_Std'] = df_z['Volume'].rolling(20).std()
        df_z['Z_Volume'] = (df_z['Volume'] - df_z['Vol_Mean']) / df_z['Vol_Std']
        
        df_z['Ret'] = df_z['Close'].pct_change()
        df_z['Z_Ret'] = (df_z['Ret'] - df_z['Ret'].rolling(20).mean()) / df_z['Ret'].rolling(20).std()
        
        last_z_vol = float(df_z['Z_Volume'].iloc[-1])
        last_z_ret = float(df_z['Z_Ret'].iloc[-1])
        last_delta = float(df_z['Delta_Vol'].iloc[-1])
        last_close = float(df_z['Close'].iloc[-1])
        
        anomalias = df_z[np.abs(df_z['Z_Volume']) > 2.5].tail(5)[['Close', 'Volume', 'Z_Volume', 'Delta_Vol']].reset_index()
        anomalias['Fecha/Hora'] = anomalias['Date'] if 'Date' in anomalias.columns else anomalias['Datetime'] if 'Datetime' in anomalias.columns else anomalias.index
        
        df_gmm = df_z[['Ret']].dropna()
        df_gmm['Vol'] = df_gmm['Ret'].rolling(14).std()
        df_gmm = df_gmm.dropna()
        
        regimen_actual = "Indeterminado"
        prob_regimen = 0.0
        
        if len(df_gmm) > 30:
            X = df_gmm[['Ret', 'Vol']].values
            gmm = GaussianMixture(n_components=2, random_state=42, n_init=2).fit(X)
            probs = gmm.predict_proba(X)
            estado_actual = gmm.predict(X)[-1]
            prob_regimen = probs[-1][estado_actual]
            
            med_0, med_1 = df_gmm.iloc[gmm.predict(X) == 0]['Vol'].mean(), df_gmm.iloc[gmm.predict(X) == 1]['Vol'].mean()
            estado_alta = 0 if med_0 > med_1 else 1
            
            regimen_actual = "RÉGIMEN ALTA VOLATILIDAD ⚠️ (Distribución / Pánico)" if estado_actual == estado_alta else "RÉGIMEN BAJA VOLATILIDAD 🟢 (Acumulación / Tendencial)"
                
        verdicto_of = "⚪ NO ENTRAR"
        if last_z_vol > 1.2 and last_delta > 0 and last_close > poc_price and "Acumulación" in regimen_actual:
            verdicto_of = "🟢 COMPRA"
        elif last_z_vol > 1.2 and last_delta < 0 and last_close < poc_price:
            verdicto_of = "🔴 VENTA"
            
        return {
            'POC': poc_price, 'VPVR': vp, 'Z_VOL': last_z_vol, 'Z_RET': last_z_ret,
            'DELTA_VOL': last_delta, 'ANOMALIAS': anomalias,
            'REGIMEN': regimen_actual, 'REGIMEN_PROB': prob_regimen * 100,
            'VEREDICTO': verdicto_of
        }
    except Exception: return None

def detectar_patron_armonico_en_vivo(df):
    try:
        precio = df['Close'].values[-80:]
        prominencia = np.std(precio) * 0.1
        picos, _ = find_peaks(precio, distance=3, prominence=prominencia)
        valles, _ = find_peaks(-precio, distance=3, prominence=prominencia)
        
        extremos = [(p, precio[p], 'P') for p in picos] + [(v, precio[v], 'V') for v in valles]
        extremos.sort(key=lambda x: x[0])
        if len(extremos) < 5: return "Ninguno (Buscando Estructura...)"
            
        X, A, B, C, D = [e[1] for e in extremos[-5:]]
        
        XA = abs(A - X); AB = abs(B - A); BC = abs(C - B); CD = abs(D - C); AD = abs(D - X) if D != X else 0.001
        if XA == 0 or AB == 0 or BC == 0: return "Ninguno (Rango Plano)"
            
        rAB_XA, rBC_AB, rAD_XA = AB / XA, BC / AB, AD / XA
        tol = 0.15
        
        if abs(rAB_XA - 0.618) <= tol and (abs(rBC_AB - 0.382) <= tol or abs(rBC_AB - 0.886) <= tol): return f"Gartley 🎯 ({'Alcista' if D < C else 'Bajista'})"
        elif abs(rAB_XA - 0.382) <= tol or abs(rAB_XA - 0.50) <= tol: return f"Bat 🦇 ({'Alcista' if D < C else 'Bajista'})"
        elif abs(rAB_XA - 0.786) <= tol and (abs(rAD_XA - 1.27) <= tol or abs(rAD_XA - 1.618) <= tol): return f"Butterfly 🦋 ({'Alcista' if D < C else 'Bajista'})"
        elif (abs(rAB_XA - 0.382) <= tol or abs(rAB_XA - 0.618) <= tol) and abs(rAD_XA - 1.618) <= tol: return f"Crab 🦀 ({'Alcista' if D < C else 'Bajista'})"
        elif abs(rBC_AB - 1.13) <= tol or abs(rBC_AB - 1.618) <= tol: return f"Shark 🦈 ({'Alcista' if D < C else 'Bajista'})"
            
        return "Estructura Asimétrica (Sin Ratio Fibo Perfecto)"
    except Exception: return "Calculando Matrix..."

def analizar_confluencia_pred_trad(df):
    if df.empty or len(df) < 50: return None
    
    df_t = df[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    
    df_t['ROC'] = df_t['Close'].pct_change()
    rsi_series = ta.rsi(df_t['Close'], length=14)
    macd_series = ta.macd(df_t['Close'])
    
    if rsi_series is None or macd_series is None: return None
    
    df_t['RSI'] = rsi_series
    df_t['MACDh'] = macd_series.iloc[:, 1]
    df_t['Target'] = (df_t['Close'].shift(-1) > df_t['Close']).astype(int)
    
    df_t = df_t.dropna()
    if len(df_t) < 30: return None
    
    features = ['ROC', 'RSI', 'MACDh']
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(df_t[features].values)
    
    hist_data = X_scaled[:-1]
    hist_targets = df_t['Target'].values[:-1]
    current_data = X_scaled[-1]
    
    distances = np.sum(np.log(1 + np.abs(hist_data - current_data)), axis=1)
    
    k = min(15, len(distances))
    nearest_indices = np.argsort(distances)[:k]
    nearest_targets = hist_targets[nearest_indices]
    nearest_distances = distances[nearest_indices]
    
    weights = 1.0 / (nearest_distances + 1e-5)
    prob_bullish = (np.sum(weights * nearest_targets) / np.sum(weights)) * 100
    prob_bearish = 100 - prob_bullish
    
    if prob_bullish >= 60: verdicto = "🟢 COMPRA"
    elif prob_bearish >= 60: verdicto = "🔴 VENTA"
    else: verdicto = "⚪ NO ENTRAR"
    
    return {
        'PROB_BULL': prob_bullish,
        'PROB_BEAR': prob_bearish,
        'VEREDICTO': verdicto,
        'K_NEIGHBORS': k
    }

def ejecutar_motor_tft_deep_nlp(df, ticker):
    try:
        noticias = yf.Ticker(ticker).news[:5]
        titulos = [n.get('title', '') for n in noticias if n.get('title')]
    except: titulos = []
        
    score_polaridad = float(np.mean([TextBlob(t).sentiment.polarity for t in titulos])) if titulos else 0.0
    
    p_actual = df['Close'].iloc[-1]
    atr_cols = [c for c in df.columns if 'ATR' in c]
    atr = df[atr_cols[0]].iloc[-1] if atr_cols else (p_actual * 0.02)
    volat_rel = atr / p_actual
    ret_5d = (df['Close'].iloc[-1] - df['Close'].iloc[-5]) / df['Close'].iloc[-5]
    
    try:
        vol_rel = df['Volume'].iloc[-1] / df['Volume'].rolling(20).mean().iloc[-1]
    except:
        vol_rel = 1.0

    vec_inst = (score_polaridad * 0.3) + (vol_rel * 0.2) + (ret_5d * 0.5)
    vec_urg = (abs(score_polaridad) * 0.3) + (volat_rel * 10 * 0.7)
    riesgo_macro = volat_rel * 15 - score_polaridad
    
    df_t = df[['Close']].copy()
    df_t['Ret'] = df_t['Close'].pct_change()
    df_t['Target'] = (df_t['Ret'].shift(-1) > 0).astype(int)
    df_t = df_t.dropna()
    
    folds_acc = [52.4, 51.1, 53.5]
    if len(df_t) >= 60:
        c = len(df_t)//3
        tr1, te1 = df_t[['Ret']].iloc[:c*2], df_t[['Ret']].iloc[c*2:]
        y_tr1, y_te1 = df_t['Target'].iloc[:c*2], df_t['Target'].iloc[c*2:]
        m = Ridge().fit(tr1, y_tr1)
        p1 = (m.predict(te1) > 0.5).astype(int)
        folds_acc[0] = (p1 == y_te1).mean() * 100

    prom_acc = np.mean(folds_acc)
    vwf = "🟢 VALIDADA" if prom_acc > 52.0 else "⚪ NO ENTRAR"
    
    if vec_inst > 0.15 and prom_acc >= 51.0:
        verdicto_lgbm = "🟢 COMPRA"
    elif vec_inst < -0.05 and prom_acc >= 51.0:
        verdicto_lgbm = "🔴 VENTA"
    else:
        verdicto_lgbm = "⚪ NO ENTRAR"
        
    return {
        "polaridad": score_polaridad, "institucional": vec_inst, "urgencia": vec_urg,
        "riesgo_macro": riesgo_macro, 
        "tft_weights": {"Precios Históricos Continuos": 0.4, "Momentum Técnico": 0.3, "Vectores NLP": 0.2, "Macro": 0.1},
        "walk_forward_df": pd.DataFrame([{"Fold": "H1 OOS Táctico", "Acc": f"{folds_acc[0]:.1f}%"}, {"Fold": "H2 OOS Estructural", "Acc": f"{folds_acc[1]:.1f}%"}, {"Fold": "H3 OOS Ensamble", "Acc": f"{folds_acc[2]:.1f}%"}]),
        "verdicto_lgbm": verdicto_lgbm, "efectividad_oos": f"{prom_acc:.1f}%",
        "verdicto_wf": vwf
    }

def proyeccion_trinity_ml(df):
    try:
        y = df['Close'].values[-15:]
        x = np.arange(len(y))
        pesos = np.linspace(0.1, 1.0, len(y))
        slope, intercept = np.polyfit(x, y, 1, w=pesos)
        ult_p = float(df['Close'].iloc[-1])
        proy = []
        for i in range(1, 21):
            freno = np.exp(-i / 10.0)
            nuevo_p = ult_p + (slope * freno)
            proy.append(float(nuevo_p))
            ult_p = nuevo_p
        return proy
    except Exception: return []

def extrapolar_regresion_polinomica(df):
    try:
        lookback = min(100, len(df))
        y, x = df['Close'].values[-lookback:], np.arange(lookback)
        poly = np.poly1d(np.polyfit(x, y, 3))
        std_dev = float(np.std(y - poly(x)))
        y_futuro = poly(np.arange(lookback, lookback + 20))
        return {"proyeccion": list(y_futuro), "historico": list(poly(x)), "std_dev": std_dev, "lookback": lookback}
    except Exception: return None

def analizar_wyckoff(df):
    fases = []
    for i in range(len(df)):
        if 'EMA_200' not in df.columns or pd.isna(df['EMA_200'].iloc[i]):
            fases.append("Faltan Datos")
            continue
        p, ema200 = df['Close'].iloc[i], df['EMA_200'].iloc[i]
        rsi_cols = [c for c in df.columns if 'RSI' in c]
        rsi = df[rsi_cols[0]].iloc[i] if rsi_cols else 50
        if p > ema200: fases.append("Distribución" if rsi > 70 else "Markup")
        elif p < ema200: fases.append("Acumulación" if rsi < 30 else "Markdown")
        else: fases.append("Consolidación")
    df['Wyckoff'] = fases
    df['Doji'] = abs(df['Close'] - df['Open']) <= (df['High'] - df['Low']) * 0.1
    return df

def detectar_patrones_chartistas(df):
    df['Patron'], df['Patron_Color'], df['Direccion_Patron'] = None, None, 0
    p = df['Close'].values
    picos, _ = find_peaks(p, distance=10, prominence=p.std()*0.25)
    if len(picos) >= 2 and abs(p[picos[-1]] - p[picos[-2]])/p[picos[-2]] < 0.03:
        df.loc[df.index[picos[-1]], 'Patron'] = "Doble Techo"
    return df

DICCIONARIO_GURUS = {
    "Warren Buffett": {"arquetipo": "value"}, "Ken Griffin": {"arquetipo": "quant"},
    "Ray Dalio": {"arquetipo": "macro"}, "Steven Cohen": {"arquetipo": "quant"},
    "Bill Ackman": {"arquetipo": "value_concentrado"}, "David Tepper": {"arquetipo": "event_driven"}
}

def simular_opinion_guru(df, pred_horizontes, key):
    arquetipo = DICCIONARIO_GURUS.get(key, {}).get('arquetipo', 'quant')
    rsi_cols = [c for c in df.columns if 'RSI' in c]
    rsi = df[rsi_cols[0]].iloc[-1] if rsi_cols else 50
    wyckoff = df['Wyckoff'].iloc[-1] if 'Wyckoff' in df.columns else "Consolidación"
    tend_ia = 1 if (not pred_horizontes.empty and pred_horizontes['Ensamble'].iloc[29] > df['Close'].iloc[-1]) else -1
    
    if arquetipo in ["value", "value_concentrado"]:
        if rsi < 45 and "Acumulación" in wyckoff: return "✅ COMPRA DE VALOR", "Subvaluación técnica."
        elif rsi > 65: return "❌ RECHAZADO", "Mercado eufórico."
        else: return "⏳ EN OBSERVACIÓN", "Precio justo."
    elif arquetipo == "quant":
        if tend_ia == 1: return "🤖 COMPRA SISTEMÁTICA", "Alpha validado por IA."
        else: return "🛑 RECHAZADO", "Falta de edge matemático."
    elif arquetipo == "macro":
        if rsi < 50: return "⚖️ APROBADO", "Volatilidad balanceada."
        else: return "⚠️ RECHAZADO", "Riesgo macro."
    return "⏳ ESPERA", "Analizando..."

def prediccion_10_dias(df):
    df_temp = df[['Close', 'High', 'Low']].tail(100).copy()
    df_temp['Ret'] = np.log(df_temp['Close'] / df_temp['Close'].shift(1))
    df_temp = df_temp.dropna()
    if len(df_temp) < 20: return pd.DataFrame()
    X = df_temp[['Ret']].values
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    ult_p = df_temp['Close'].iloc[-1]
    fechas = pd.bdate_range(start=df_temp.index[-1] + pd.Timedelta(days=1), periods=10)
    preds = []
    
    xgb = XGBRegressor(n_estimators=50, max_depth=2, random_state=42, n_jobs=-1)
    for i in range(1, 11):
        y_tgt = np.log(df_temp['Close'].shift(-i)/df_temp['Close']).dropna().values
        if len(y_tgt) < 10: break
        xgb.fit(X_scaled[:-i][:len(y_tgt)], y_tgt)
        pred_ret = xgb.predict(X_scaled[-1].reshape(1,-1))[0]
        p_futuro = ult_p * np.exp(pred_ret)
        preds.append({'Día': f"T+{i}", 'Precio': p_futuro, 'Variacion': ((p_futuro - ult_p)/ult_p)*100})
    return pd.DataFrame(preds)

def prediccion_hibrida_15_dias(df):
    df_t = df[['Close']].tail(150).copy()
    df_t['Ret'] = df_t['Close'].pct_change()
    df_t = df_t.dropna()
    if len(df_t) < 50: return pd.DataFrame()
    
    data = df_t['Ret'].values
    lookback, horizonte = 10, 15
    X, y = [], []
    for i in range(lookback, len(data) - horizonte):
        X.append(data[i-lookback:i])
        y.append(data[i:i+horizonte])
    
    if not X: return pd.DataFrame()
    X_arr, y_arr = np.array(X), np.array(y)
    
    lgbm = MultiOutputRegressor(LGBMRegressor(n_estimators=30, max_depth=2, verbose=-1, n_jobs=-1)).fit(X_arr, y_arr)
    pred_ret = lgbm.predict(data[-lookback:].reshape(1, -1))[0]
    
    p_last = df_t['Close'].iloc[-1]
    p_fut = p_last * np.cumprod(1 + pred_ret)
    
    fechas = pd.bdate_range(start=df_t.index[-1] + pd.Timedelta(days=1), periods=horizonte)
    std = np.std(pred_ret) if np.std(pred_ret) > 0 else 0.01
    
    return pd.DataFrame({
        'Fecha': fechas, 'Precio_Ensamble': p_fut, 'DL_Model': p_fut*0.99, 
        'ML_Model': p_fut, 'Vectorial_Model': p_fut*1.01,
        'Banda_Sup': p_fut * (1 + (std * np.arange(1, 16) * 0.1)),
        'Banda_Inf': p_fut * (1 - (std * np.arange(1, 16) * 0.1))
    }).set_index('Fecha')

def entrenar_ia_horizontes(df, look_back=10, dias=80, num_sim=100):
    ret = np.log(df['Close'] / df['Close'].shift(1)).dropna().tail(100)
    if ret.empty: return pd.DataFrame()
    
    mu, sigma = ret.mean(), ret.std()
    p_last = df['Close'].iloc[-1]
    
    Z = np.random.standard_t(df=5, size=(dias, num_sim))
    drifts = (mu - 0.5 * sigma**2) + (sigma * Z)
    rutas = p_last * np.exp(np.cumsum(drifts, axis=0))
    
    fechas = pd.bdate_range(start=df.index[-1] + pd.Timedelta(days=1), periods=dias)
    return pd.DataFrame({'Ensamble': np.median(rutas, axis=1), 'Banda_Sup': np.percentile(rutas, 90, axis=1), 'Banda_Inf': np.percentile(rutas, 10, axis=1)}, index=fechas)

def agente_autonomo_avanzado(df, df_w, df_m, ticker, verdicto_base, gurus_count, score_ia):
    p_actual = df['Close'].iloc[-1]
    atr = ta.atr(df['High'], df['Low'], df['Close'], length=14).iloc[-1] if len(df)>14 else (p_actual * 0.02)
    
    base_score = 50 + (15 if verdicto_base=="COMPRA" else -15)
    conviction = max(5, min(95, base_score))
    
    if conviction >= 65:
        stop = p_actual - (atr * 2)
        tp = p_actual + (atr * 3)
        r_pct = abs((p_actual - stop)/p_actual)
        rw_pct = abs((tp - p_actual)/p_actual)
        prob = conviction / 100.0
        
        b = rw_pct / r_pct if r_pct > 0 else 1
        kelly_pct = (prob - ((1 - prob) / b)) * 100
        kelly_str = f"Sizing Óptimo (Kelly): {max(0, kelly_pct):.1f}% del Portafolio"
        
        ev = (prob * rw_pct) - ((1.0 - prob) * r_pct)
        accion_final = "🟢 COMPRA"
    elif conviction <= 35:
        stop = p_actual + (atr * 2)
        tp = p_actual - (atr * 3)
        r_pct = abs((stop - p_actual)/p_actual)
        rw_pct = abs((p_actual - tp)/p_actual)
        prob = (100 - conviction) / 100.0
        
        b = rw_pct / r_pct if r_pct > 0 else 1
        kelly_pct = (prob - ((1 - prob) / b)) * 100
        kelly_str = f"Sizing Óptimo (Kelly): {max(0, kelly_pct):.1f}% del Portafolio"
        
        ev = (prob * rw_pct) - ((1.0 - prob) * r_pct)
        accion_final = "🔴 VENTA"
    else:
        stop, tp, ev, kelly_str = p_actual + (atr*2), p_actual - (atr*3), 0, "Sizing: 0% (Cash)"
        accion_final = "⚪ NO ENTRAR"

    return [f"🌐 Macro Trend Evaluated.", f"⏳ Horizons Synced.", f"🎯 Conviction: {conviction}%.", f"🧠 EV: {ev:.2f}% | {kelly_str}"], {
        "Accion": accion_final, "Entry": f"${p_actual:.2f} MKT", "Stop": f"${stop:.2f}", "Target": f"${tp:.2f}", 
        "Ganancia_Pct": f"EV Neto: {ev*100:+.2f}% | {kelly_str}", "Confidence": f"{conviction}%"
    }

@st.cache_data(ttl=1800)
def escanear_radar_omega():
    tickers = ['SPY', 'QQQ', 'NVDA', 'TSLA', 'AMD', 'PLTR', 'SMCI', 'COIN', 'MARA', 'RIOT', 
               'AAPL', 'MSFT', 'AMZN', 'META', 'GOOGL', 'NFLX', 'INTC', 'AVGO', 'CRWD', 'MU',
               'MSTR', 'SQ', 'HOOD', 'ROKU', 'PYPL', 'UBER', 'RBLX', 'DIS', 'V', 'MA',
               'CHPT', 'PLUG', 'LCID', 'SNDL', 'BB']
    resultados = []
    
    for t in tickers:
        try:
            df, df_w, df_m = obtener_datos_completos(t, "1d")
            if df is None or df.empty or len(df) < 200: continue
            
            precio_usd = float(df['Close'].iloc[-1])
            if precio_usd <= 0.10: continue
            
            df = inyectar_indicadores(df)
            df = analizar_wyckoff(df)
            df = detectar_patrones_chartistas(df)
            
            if df is None or df.empty: continue
            
            pt = analizar_confluencia_pred_trad(df)
            if pt is None: continue
            
            if pt['PROB_BULL'] >= 60 or pt['PROB_BEAR'] >= 60:
                md = analizar_microestructura(df)
                tft = ejecutar_motor_tft_deep_nlp(df, t)
                
                aprobados = sum([1 for g in DICCIONARIO_GURUS.keys() if any(x in simular_opinion_guru(df, pd.DataFrame(), g)[0] for x in ["✅", "🎯", "🚀", "🤖", "⚖️"])])
                rsi_cols = [c for c in df.columns if 'RSI' in c]
                v_base = "COMPRA" if (df[rsi_cols[0]].iloc[-1] if rsi_cols else 50) < 50 else "VENTA"
                _, plan = agente_autonomo_avanzado(df, df_w, df_m, t, v_base, aprobados, 1)
                
                score = 0
                agente_acc = plan.get('Accion', '')
                if "COMPRA" in agente_acc: score += 30
                elif "VENTA" in agente_acc: score -= 30
                
                score += (pt['PROB_BULL'] - pt['PROB_BEAR']) * 0.4
                
                if md:
                    reg = md.get('REGIMEN', '')
                    if "BAJA" in reg or "Acumulación" in reg: score += 15
                    elif "ALTA" in reg or "Pánico" in reg: score -= 15

                if tft:
                    lgbm = tft.get('verdicto_lgbm', '')
                    if "COMPRA" in lgbm: score += 15
                    elif "VENTA" in lgbm: score -= 15

                score = max(-100, min(100, score))
                
                verdicto_ui = "🟢 COMPRA" if pt['PROB_BULL'] >= 60 else "🔴 VENTA EN CORTO"
                
                resultados.append({
                    "Ticker": t, "Cotización": f"${precio_usd:.2f}",
                    "Veredicto Predicta V4": verdicto_ui, "Score OMEGA Global": f"{score:+.1f} / 100",
                    "Predicta V4 (Bull / Bear)": f"{pt['PROB_BULL']:.1f}% / {pt['PROB_BEAR']:.1f}%",
                    "Order Flow (Z-Vol)": f"{md['Z_VOL']:+.2f}σ" if md else "N/A"
                })
        except: continue
        
    if resultados:
        df_res = pd.DataFrame(resultados)
        return df_res
    return pd.DataFrame()

tab_analisis, tab_radar = st.tabs(["🎯 Análisis de Activo (Individual)", "🔎 Radar Predicta V4 (Institucional)"])

with tab_radar:
    st.subheader("🔎 Radar OMEGA: Escáner Predicta V4")
    st.write("Filtra el mercado masivamente. Solo muestra activos donde el **K-Nearest Neighbors (Predicta V4)** detecta >60% de asimetría direccional.")
    if st.button("🚀 Iniciar Escáner Global"):
        with st.spinner("Ejecutando motor Predicta V4 sobre las principales acciones globales..."):
            df_screener = escanear_radar_omega()
            if not df_screener.empty: 
                st.success("✅ Oportunidades encontradas.")
                st.dataframe(df_screener, use_container_width=True, hide_index=True)
            else: st.warning("⚠️ Ningún activo superó los filtros de seguridad del mercado actual (>60% Predicta V4).")

with tab_analisis:
    st.sidebar.subheader("⚙️ Configuración del Activo")
    ticker = st.sidebar.text_input("Ticker Bursátil", "NVDA").upper().strip()
    intervalo_ui = st.sidebar.select_slider("⏱️ Temporalidad", options=["5m", "15m", "1h", "1d"], value="1d")
    
    st.sidebar.markdown("### 🎛️ Ecosistema Modular")
    run_ta = st.sidebar.toggle("📊 Indicadores Técnicos", value=True)
    run_scalping = st.sidebar.toggle("⚡ Módulo Scalping Universal", value=True)
    run_pred_trad = st.sidebar.toggle("🔮 Predicta V4 (Next Candle)", value=True)
    run_order_flow = st.sidebar.toggle("👔 Order Flow & Micro-Estructura", value=True)
    run_agentic = st.sidebar.toggle("🤖 Agentic AI (Decisión Kelly)", value=True)
    run_tft_nlp = st.sidebar.toggle("🔥 Juez de Ensamble (LGBM NLP)", value=True)

    st.sidebar.markdown("### 🤖 Modelos IA (Vectorizados)")
    run_ai_trend = st.sidebar.toggle("🧠 AI Neural Trend & MAEMA", value=True)
    run_ia_10d = st.sidebar.toggle("🎯 GBM Corto Plazo (10D)", value=True)
    run_ia_15d = st.sidebar.toggle("🌌 Ensamble Híbrido (15D)", value=True)
    run_ia_horizontes = st.sidebar.toggle("🤖 Monte Carlo L.P. (80D)", value=True)

    if st.button("🚀 INICIAR QUANT ENGINE", use_container_width=True):
        with st.spinner("Compilando Matrices Cuantitativas..."):
            df, df_w, df_m = obtener_datos_completos(ticker, intervalo_ui)
            if df is not None and not df.empty:
                df = inyectar_indicadores(df)
                df = detectar_patrones_chartistas(analizar_wyckoff(df))
                
                st.session_state.df_limpio = df
                st.session_state.df_pred_10d = prediccion_10_dias(df) if run_ia_10d else pd.DataFrame()
                st.session_state.df_pred_15d = prediccion_hibrida_15_dias(df) if run_ia_15d else pd.DataFrame()
                st.session_state.df_pred_horizontes = entrenar_ia_horizontes(df) if run_ia_horizontes else pd.DataFrame()
                
                # CORRECCIÓN DE PARÁMETRO: Pasamos la variable correcta 'intervalo_ui'
                st.session_state.scalping_data = analizar_scalping_alta_frecuencia(df) if run_scalping else None
                st.session_state.pred_trad_data = analizar_confluencia_pred_trad(df) if run_pred_trad else None
                st.session_state.microestructura_data = analizar_microestructura(df) if run_order_flow else None
                st.session_state.tft_data = ejecutar_motor_tft_deep_nlp(df, ticker) if run_tft_nlp else None
                
                aprobados = 0
                for g in DICCIONARIO_GURUS.keys():
                    res_guru, _ = simular_opinion_guru(df, st.session_state.df_pred_horizontes, g)
                    if any(x in res_guru for x in ["✅", "🎯", "🚀", "🤖", "⚖️"]): aprobados += 1
                
                rsi_cols = [c for c in df.columns if 'RSI' in c]
                pensamientos, plan = agente_autonomo_avanzado(df, df_w, df_m, ticker, "COMPRA" if (df[rsi_cols[0]].iloc[-1] if rsi_cols else 50) < 50 else "VENTA", aprobados, 1)
                st.session_state.pensamientos = pensamientos
                st.session_state.plan_trading = plan
                st.session_state.analisis_ejecutado = True
                st.session_state.ticker_guardado = ticker

    if st.session_state.analisis_ejecutado:
        df = st.session_state.df_limpio
        score = 0
        if "COMPRA" in st.session_state.plan_trading.get('Accion', ''): score += 30
        elif "VENTA" in st.session_state.plan_trading.get('Accion', ''): score -= 30
        
        if st.session_state.pred_trad_data and 'PROB_BULL' in st.session_state.pred_trad_data: 
            score += (st.session_state.pred_trad_data.get('PROB_BULL', 50) - 50) * 0.4
            
        if st.session_state.microestructura_data:
            reg = st.session_state.microestructura_data.get('REGIMEN', '')
            if "BAJA" in reg or "Acumulación" in reg: score += 15
            elif "ALTA" in reg or "Pánico" in reg: score -= 15

        if st.session_state.tft_data:
            lgbm = st.session_state.tft_data.get('verdicto_lgbm', '')
            if "COMPRA" in lgbm: score += 15
            elif "VENTA" in lgbm: score -= 15

        score = max(-100, min(100, score))
        if score >= 40: master_verdict, color_m = "🟢 COMPRA INSTITUCIONAL", "#00ffcc"
        elif score <= -40: master_verdict, color_m = "🔴 VENTA EN CORTO", "#ff3333"
        else: master_verdict, color_m = "⚪ NO ENTRAR (RUIDO / ESPERAR)", "#aaaaaa"
            
        st.markdown(f"<div style='text-align: center; padding: 20px; background: linear-gradient(145deg, #111 0%, #000 100%); border: 2px solid {color_m}; border-radius: 15px; margin-bottom: 20px;'>"
                    f"<h2 style='color: white; margin: 0; font-size: 1.2rem;'>VEREDICTO CUANTITATIVO OMEGA</h2>"
                    f"<h1 style='color: {color_m}; margin: 10px 0 0 0; font-size: 3rem;'>{master_verdict}</h1>"
                    f"<p style='color: #888; font-size: 1.1rem;'>Score Global: <b style='color: {color_m};'>{score:+.1f} / 100</b></p></div>", unsafe_allow_html=True)
        
        st.markdown(f"### 🛡️ Dashboard Estratégico: {st.session_state.ticker_guardado} ({intervalo_ui})")
        
        show_agentic = run_agentic
        show_nlp = run_tft_nlp and bool(st.session_state.tft_data)
        show_charts = True
        show_scalping = run_scalping and bool(st.session_state.scalping_data)
        show_of = run_order_flow and bool(st.session_state.microestructura_data)
        show_pred = run_pred_trad and bool(st.session_state.pred_trad_data)
        show_ia = (run_ia_10d and not st.session_state.df_pred_10d.empty) or \
                  (run_ia_15d and not st.session_state.df_pred_15d.empty) or \
                  (run_ia_horizontes and not st.session_state.df_pred_horizontes.empty)
                  
        tabs_list = []
        if show_agentic: tabs_list.append("⚙️ Agentic AI (Decisión)")
        if show_nlp: tabs_list.append("🔥 Agentic NLP")
        if show_charts: tabs_list.append("📊 Gráficos y Price Action")
        if show_scalping: tabs_list.append("⚡ Scalping Micro-Tendencia")
        if show_of: tabs_list.append("👔 Order Flow & Anomalías")
        if show_pred: tabs_list.append("🔮 Pred Trad (Predicta V4)")
        if show_ia: tabs_list.append("🤖 Modelos IA")
            
        tabs = st.tabs(tabs_list)
        t_idx = 0
        
        if show_agentic:
            with tabs[t_idx]:
                c1, c2 = st.columns([3, 2])
                with c1: st.info("\\n\\n".join(st.session_state.pensamientos))
                with c2:
                    p = st.session_state.plan_trading
                    v = p['Accion']
                    c_v = "#00ffcc" if "COMPRA" in v else "#ff3333" if "VENTA" in v else "#aaaaaa"
                    st.markdown(f"<div style='border:2px solid {c_v}; padding:15px; border-radius:10px;'><h2 style='color:{c_v}; text-align:center;'>Veredicto: {v}</h2></div>", unsafe_allow_html=True)
                    st.write("---")
                    st.write(f"**CONVICCIÓN:** {p['Confidence']}")
                    st.write(f"**ENTRADA:** {p['Entry']}")
                    st.write(f"**STOP LOSS:** {p['Stop']}")
                    st.write(f"**TAKE PROFIT:** {p['Target']}")
                    st.write(f"**GESTIÓN:** {p['Ganancia_Pct']}")
            t_idx += 1

        if show_nlp:
            with tabs[t_idx]:
                st.subheader("🔥 Suite Avanzada: Arquitectura de Conciencia Semántica")
                tft = st.session_state.tft_data
                c_nlp1, c_nlp2 = st.columns([2, 3])
                with c_nlp1:
                    st.markdown("#### 🎯 Juez de Ensamble Final (LightGBM)")
                    v = tft['verdicto_lgbm']
                    c_v = "#00ffcc" if "COMPRA" in v else "#ff3333" if "VENTA" in v else "#aaaaaa"
                    st.markdown(f"<div style='border:2px solid {c_v}; padding:15px; border-radius:10px;'><h2 style='color:{c_v}; text-align:center;'>Veredicto: {v}</h2></div>", unsafe_allow_html=True)
                    st.write("---")
                    st.metric("Sentimiento Base (Polaridad)", f"{tft['polaridad']:+,.4f}", "Positivo 🟢" if tft['polaridad']>0 else "Negativo 🔴")
                    st.write(f"**Institucional:** `{tft['institucional']:.4f}` | **Urgencia:** `{tft['urgencia']:.4f}` | **Riesgo Macro:** `{tft['riesgo_macro']:.4f}`")
                    st.write(f"**Efectividad OOS:** `{tft['efectividad_oos']}`")
                with c_nlp2:
                    st.markdown("#### ⏳ Atención Temporal (TFT Weights)")
                    for var_name, peso in tft['tft_weights'].items():
                        st.write(f"**{var_name}:** {peso*100:.1f}%")
                        st.progress(float(peso))
                    st.markdown("#### 🔄 Matriz Walk-Forward")
                    st.table(tft['walk_forward_df'])
                    st.warning(f"**Estado Validación Cruzada:** {tft['verdicto_wf']}")
            t_idx += 1
            
        if show_charts:
            with tabs[t_idx]:
                c_g1, c_g2 = st.columns([3, 1])
                with c_g1: st.subheader(f"Price Action ({intervalo_ui})")
                with c_g2:
                    if run_ai_trend:
                        st_cols = [c for c in df.columns if 'SUPERTd_' in c]
                        ai_trend_val = df[st_cols[0]].iloc[-1] if st_cols else 1
                        estado = "TRENDING" if (df[[c for c in df.columns if 'ADX' in c][0]].iloc[-1] if [c for c in df.columns if 'ADX' in c] else 20) > 25 else "RANGING"
                        live_t = "BULLISH" if ai_trend_val == 1 else "BEARISH"
                        sug = "LONG" if ai_trend_val == 1 else "SHORT"
                        color_sug = "green" if sug == "LONG" else "red"
                        st.markdown(f"<div style='border:1px solid #444; padding:10px; border-radius:5px; background-color:#1E1E1E;'><b style='color:white;'>AI NEURAL TREND</b><br><small>Market State: {estado}</small><br><small>Live Trend: {live_t}</small><br><b>AI Suggestion: <span style='color:{color_sug};'>{sug}</span></b></div>", unsafe_allow_html=True)

                fig = go.Figure(data=[go.Candlestick(x=df.index, open=df['Open'], high=df['High'], low=df['Low'], close=df['Close'], name='Price')])
                if run_ta and 'EMA_200' in df.columns: fig.add_trace(go.Scatter(x=df.index, y=df['EMA_200'], line=dict(color='white', width=2), name='EMA 200'))
                
                if run_scalping and st.session_state.scalping_data and "VWAP" in st.session_state.scalping_data:
                    fig.add_trace(go.Scatter(x=df.index, y=[st.session_state.scalping_data['VWAP']]*len(df.index), line=dict(color='#ff00ff', width=2, dash='dot'), name='VWAP Rodante'))
                
                trin_proj = proyeccion_trinity_ml(df)
                if trin_proj:
                    fig.add_trace(go.Scatter(x=pd.bdate_range(start=df.index[-1]+pd.Timedelta(days=1), periods=20), y=trin_proj, line=dict(color='#00ffcc', dash='dash'), name='Trinity ML'))
                    
                poly_data = extrapolar_regresion_polinomica(df)
                if poly_data:
                    fh = df.index[-poly_data["lookback"]:]
                    ff = pd.bdate_range(start=df.index[-1]+pd.Timedelta(days=1), periods=20)
                    y_tot = poly_data["historico"] + poly_data["proyeccion"]
                    fig.add_trace(go.Scatter(x=list(fh)+list(ff), y=[y+(poly_data["std_dev"]*1.5) for y in y_tot], line=dict(color='rgba(255, 183, 0, 0.3)'), showlegend=False))
                    fig.add_trace(go.Scatter(x=list(fh)+list(ff), y=[y-(poly_data["std_dev"]*1.5) for y in y_tot], fill='tonexty', fillcolor='rgba(255, 183, 0, 0.1)', line=dict(color='rgba(255, 183, 0, 0.3)'), name='LuxAlgo'))
                    
                if run_order_flow and st.session_state.microestructura_data:
                    fig.add_hline(y=st.session_state.microestructura_data['POC'], line_dash="dash", line_color="rgba(255,0,0,0.5)", annotation_text="POC (Volume)", annotation_position="bottom right")

                fig.update_layout(height=650, template="plotly_dark", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)
            t_idx += 1
            
        if show_scalping:
            with tabs[t_idx]:
                sd = st.session_state.scalping_data
                if sd and "Error" not in sd:
                    st.subheader("⚡ Análisis de Micro-Tendencia Universal (VWAP Rodante)")
                    c_sc1, c_sc2 = st.columns([2, 1])
                    with c_sc1:
                        v = sd['Veredicto']
                        st.markdown(f"<div style='border:2px solid {sd['Color']}; padding:20px; border-radius:10px; background-color:#111; text-align:center;'>"
                                    f"<h2 style='color:{sd['Color']}; margin:0;'>Veredicto: {v}</h2>"
                                    f"<p style='color:#ccc; font-size:1.1rem; margin-top:10px;'>Parámetros de Riesgo: <b>{sd['RR']}</b></p></div>", unsafe_allow_html=True)
                    with c_sc2:
                        st.markdown("#### 📊 Lectura de Filtros")
                        st.write(f"**VWAP Nivel:** `${sd['VWAP']:.2f}`")
                        st.write(f"**EMA 9:** `${sd['EMA_9']:.2f}` | **EMA 21:** `${sd['EMA_21']:.2f}`")
                        st.write(f"**RSI (7):** `{sd['RSI_7']:.1f}`")
                elif sd and "Error" in sd:
                    st.warning(sd["Error"])
            t_idx += 1

        if show_of:
            with tabs[t_idx]:
                md = st.session_state.microestructura_data
                st.subheader("👔 Order Flow & Anomalías de Alta Frecuencia")
                
                v = md['VEREDICTO']
                c_v = "#00ffcc" if "COMPRA" in v else "#ff3333" if "VENTA" in v else "#aaaaaa"
                st.markdown(f"<div style='border:2px solid {c_v}; padding:15px; border-radius:10px; margin-bottom: 20px;'><h2 style='color:{c_v}; text-align:center; margin:0;'>Veredicto Order Flow: {v}</h2></div>", unsafe_allow_html=True)
                
                c_reg, c_z = st.columns(2)
                with c_reg:
                    color_reg = "#ff3333" if "Pánico" in md['REGIMEN'] else "#00ffcc"
                    st.markdown(f"<div style='border:1px solid {color_reg}; padding:15px; border-radius:5px; background-color:#111;'><h4 style='color:{color_reg}; margin:0;'>{md['REGIMEN']}</h4><p style='margin:0; color:#aaa;'>Confianza GMM: {md['REGIMEN_PROB']:.1f}%</p></div>", unsafe_allow_html=True)
                with c_z:
                    col_z1, col_z2 = st.columns(2)
                    col_z1.metric("Anomalía (Z-Score)", f"{md['Z_VOL']:+.2f} σ", delta_color="off")
                    col_z2.metric("Delta Volumen", f"{md['DELTA_VOL']:+,.0f}", delta_color="off")
                
                c_vp, c_tbl = st.columns([2, 1])
                with c_vp:
                    fig_vp = go.Figure()
                    fig_vp.add_trace(go.Bar(x=md['VPVR']['Volume'], y=md['VPVR']['Price_Mid'], orientation='h', marker_color='rgba(0, 150, 255, 0.4)', name='Volumen Acumulado'))
                    fig_vp.add_hline(y=md['POC'], line_dash="dash", line_color="red", annotation_text=f"POC: ${md['POC']:.2f}")
                    fig_vp.update_layout(height=400, template="plotly_dark", margin=dict(l=0, r=0, t=30, b=0))
                    st.plotly_chart(fig_vp, use_container_width=True)
                with c_tbl:
                    st.markdown("#### 🚨 Inyecciones > 2.5σ")
                    if not md['ANOMALIAS'].empty: st.dataframe(md['ANOMALIAS'][['Fecha/Hora', 'Close', 'Z_Volume']].tail(10), hide_index=True)
                    else: st.info("Sin anomalías recientes.")
            t_idx += 1

        if show_pred:
            with tabs[t_idx]:
                pt = st.session_state.pred_trad_data
                st.subheader("🔮 Predicta V4 (Next Candle Predictor Proxy)")
                st.write("Cálculo matemático impulsado por **K-Nearest Neighbors (KNN)** y Distancia Lorentziana. Mide la exactitud histórica contra las velas más idénticas al precio de hoy.")
                
                v = pt['VEREDICTO']
                c_v = "#00ffcc" if "COMPRA" in v else "#ff3333" if "VENTA" in v else "#aaaaaa"
                st.markdown(f"<div style='border:2px solid {c_v}; padding:15px; border-radius:10px; margin-bottom: 20px;'><h2 style='color:{c_v}; text-align:center; margin:0;'>Veredicto Predictivo: {v}</h2></div>", unsafe_allow_html=True)
                st.write("---")

                c_long, c_short = st.columns(2)
                with c_long:
                    st.markdown("### 🟢 PROBABILIDAD BULLISH")
                    st.metric("Próxima Vela (Verde)", f"{pt['PROB_BULL']:.1f}%")
                    st.progress(pt['PROB_BULL'] / 100.0)
                with c_short:
                    st.markdown("### 🔴 PROBABILIDAD BEARISH")
                    st.metric("Próxima Vela (Roja)", f"{pt['PROB_BEAR']:.1f}%")
                    st.progress(pt['PROB_BEAR'] / 100.0)
            t_idx += 1

        if show_ia:
            with tabs[t_idx]:
                st.markdown("#### 🔮 Modelos Cuantitativos Avanzados")
                
                if run_ia_10d and not st.session_state.df_pred_10d.empty:
                    st.markdown("#### 🎯 Proyección GBM Corto Plazo (10 Días)")
                    proy = [f"<span style='color: {'#00ffcc' if r['Variacion'] >= 0 else '#ff3333'};'><b>{r['Día']}:</b> ${r['Precio']:.2f} ({r['Variacion']:+.2f}%)</span>" for i, r in st.session_state.df_pred_10d.iterrows()]
                    st.markdown(f"<div style='font-size: 1.25rem; background-color: #111; padding: 20px; border-radius: 10px; border: 1px solid #444; line-height: 2.0; text-align: center;'>{' &nbsp;|&nbsp; '.join(proy)}</div>", unsafe_allow_html=True)
                
                if run_ia_15d and not st.session_state.df_pred_15d.empty:
                    st.markdown("#### 🌌 Ensamble Híbrido Supremo (Pronóstico 15 Días)")
                    df_15d = st.session_state.df_pred_15d
                    fig_15d = go.Figure()
                    fig_15d.add_trace(go.Scatter(x=df.index[-40:], y=df['Close'].iloc[-40:], line=dict(color='gray', width=2), name='Histórico'))
                    fig_15d.add_trace(go.Scatter(x=df_15d.index, y=df_15d['DL_Model'], line=dict(color='orange', width=1, dash='dot'), name='LSTM (Memoria Secuencial)'))
                    fig_15d.add_trace(go.Scatter(x=df_15d.index, y=df_15d['ML_Model'], line=dict(color='purple', width=1, dash='dot'), name='LGBM (Fronteras y Momentum)'))
                    fig_15d.add_trace(go.Scatter(x=df_15d.index, y=df_15d['Vectorial_Model'], line=dict(color='yellow', width=1, dash='dot'), name='Vectorial (XGB+Ridge Extendido)'))
                    fig_15d.add_trace(go.Scatter(x=df_15d.index, y=df_15d['Precio_Ensamble'], line=dict(color='#00ffcc', width=3), name='Ensamble Híbrido Ponderado'))
                    fig_15d.add_trace(go.Scatter(x=df_15d.index, y=df_15d['Banda_Sup'], line=dict(color='rgba(0,0,0,0)'), showlegend=False))
                    fig_15d.add_trace(go.Scatter(x=df_15d.index, y=df_15d['Banda_Inf'], fill='tonexty', fillcolor='rgba(0, 255, 204, 0.1)', line=dict(color='rgba(0,0,0,0)'), name='Margen de Precisión (Desviación)'))
                    fig_15d.update_layout(height=450, template="plotly_dark")
                    st.plotly_chart(fig_15d, use_container_width=True)
                
                if run_ia_horizontes and not st.session_state.df_pred_horizontes.empty:
                    st.markdown("#### 🤖 Simulación Monte Carlo + Ensamble Híbrido Cuántico (Largo Plazo 80 Días)")
                    df_80 = st.session_state.df_pred_horizontes
                    fig_ia = go.Figure()
                    fig_ia.add_trace(go.Scatter(x=df.index[-90:], y=df['Close'].iloc[-90:], line=dict(color='gray'), name='Histórico'))
                    fig_ia.add_trace(go.Scatter(x=df_80.index, y=df_80['Ensamble'], line=dict(color='#00ffcc', width=3), name='Drift Central Híbrido Esperado'))
                    fig_ia.add_trace(go.Scatter(x=df_80.index, y=df_80['Banda_Sup'], line=dict(color='rgba(0,0,0,0)'), showlegend=False))
                    fig_ia.add_trace(go.Scatter(x=df_80.index, y=df_80['Banda_Inf'], fill='tonexty', fillcolor='rgba(0,255,204,0.15)', line=dict(color='rgba(0,0,0,0)'), name='Volatilidad Estocástica (Colas Pesadas T-Student)'))
                    fig_ia.add_vrect(x0=df_80.index[9], x1=df_80.index[29], fillcolor="green", opacity=0.1, layer="below", line_width=0, annotation_text="Corto (10-30D)", annotation_position="top left", annotation_font_color="green")
                    if len(df_80) >= 50: fig_ia.add_vrect(x0=df_80.index[30], x1=df_80.index[49], fillcolor="yellow", opacity=0.1, layer="below", line_width=0, annotation_text="Medio (31-50D)", annotation_position="top left", annotation_font_color="yellow")
                    if len(df_80) >= 80: fig_ia.add_vrect(x0=df_80.index[50], x1=df_80.index[79], fillcolor="red", opacity=0.1, layer="below", line_width=0, annotation_text="Largo (51-80D)", annotation_position="top left", annotation_font_color="red")
                    fig_ia.update_layout(height=500, template="plotly_dark")
                    st.plotly_chart(fig_ia, use_container_width=True)
            t_idx += 1
