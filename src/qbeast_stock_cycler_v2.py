#!/usr/bin/env python3
"""
QBEAST AI — Stock Cycler Dashboard v2
Single compact header row · Price Type & Style as dropdowns · Max plot area
"""

import os, glob, json
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import dash
from dash import dcc, html, Input, Output, State, callback_context, no_update, MATCH, ALL
from dash.exceptions import PreventUpdate

# ══════════════════════════════════════════════════════════════════
# CONFIG
# ══════════════════════════════════════════════════════════════════
DATA_DIR      = "data/final/2605091418/eod/EQUITY"
#SYMBOLS_CSV  = "ind_nifty100list.csv"   # Universe filter (NSE-format CSV with `Symbol` column)
SYMBOLS_CSV   = ""
N_DEMO        = 30
APP_PORT      = 80
VOL_FRAC      = 0.22    # volume occupies bottom 22% of price panel

# ══════════════════════════════════════════════════════════════════
# UNIVERSE LOADER — restrict scan to symbols listed in SYMBOLS_CSV
# ══════════════════════════════════════════════════════════════════
def load_universe(csv_path=SYMBOLS_CSV):
    """
    Load the symbol universe from an NSE-format CSV.

    Returns
    -------
    symbols : list[str]
        Symbols in the order they appear in the CSV (uppercased, stripped).
    sectors : dict[str, str]
        Mapping symbol → industry/sector (from the `Industry` column).
    """
    if not os.path.exists(csv_path):
        print(f"  ⚠  Universe file '{csv_path}' not found — falling back to all CSVs in DATA_DIR.")
        return None, {}
    df = pd.read_csv(csv_path)
    # Be tolerant of column-name casing/whitespace
    df.columns = [c.strip() for c in df.columns]
    sym_col = next((c for c in df.columns if c.lower() == "symbol"), None)
    ind_col = next((c for c in df.columns if c.lower() == "industry"), None)
    if sym_col is None:
        raise ValueError(f"'{csv_path}' has no 'Symbol' column. Columns: {list(df.columns)}")
    symbols = (df[sym_col].astype(str).str.strip().str.upper()
                          .replace("", pd.NA).dropna().tolist())
    sectors = ({s: i for s, i in zip(symbols, df[ind_col].astype(str).str.strip().tolist())}
               if ind_col else {})
    return symbols, sectors

UNIVERSE, UNIV_SECTORS = load_universe()

# ══════════════════════════════════════════════════════════════════
# COLOURS
# ══════════════════════════════════════════════════════════════════
C = dict(
    bg="#0d1117",  paper="#0d1117",  grid="#1e2a35",
    hdr="#0a0f16", border="#263238", border2="#37474f",
    font="#e8eaed", sub="#e8eaed",
    bull="#26a69a", bear="#ef5350",
    blue="#42A5F5", purple="#CE93D8", pink="#F48FB1",
    teal="#80CBC4", orange="#FF9800", yellow="#FFF176",
    spike="#546E7A",
    btn="#1a2332",  btn_border="#2d3f50",
    act_blue="#1565C0",  act_green="#1B5E20",  act_purple="#4A148C",
)

# ══════════════════════════════════════════════════════════════════
# SYNTHETIC DATA GENERATOR
# ══════════════════════════════════════════════════════════════════
NSE = ["RELIANCE","TCS","INFY","HDFCBANK","ICICIBANK","HINDUNILVR","ITC",
       "SBIN","BAJFINANCE","KOTAKBANK","LT","ASIANPAINT","AXISBANK","MARUTI",
       "TITAN","SUNPHARMA","ULTRACEMCO","NESTLEIND","WIPRO","TECHM",
       "HCLTECH","POWERGRID","NTPC","ONGC","COALINDIA","GRASIM","JSWSTEEL",
       "TATAMOTORS","TATASTEEL","BAJAJFINSV"]
SECTOR = {"RELIANCE":"Energy","TCS":"IT","INFY":"IT","HDFCBANK":"Banking",
          "ICICIBANK":"Banking","HINDUNILVR":"FMCG","ITC":"FMCG","SBIN":"Banking",
          "BAJFINANCE":"Finance","KOTAKBANK":"Banking","LT":"Infra","ASIANPAINT":"Paints",
          "AXISBANK":"Banking","MARUTI":"Auto","TITAN":"Retail","SUNPHARMA":"Pharma",
          "ULTRACEMCO":"Cement","NESTLEIND":"FMCG","WIPRO":"IT","TECHM":"IT",
          "HCLTECH":"IT","POWERGRID":"Utility","NTPC":"Utility","ONGC":"Energy",
          "COALINDIA":"Energy","GRASIM":"Diversified","JSWSTEEL":"Metal",
          "TATAMOTORS":"Auto","TATASTEEL":"Metal","BAJAJFINSV":"Finance"}
BASE  = {"RELIANCE":2500,"TCS":3800,"INFY":1700,"HDFCBANK":1650,"ICICIBANK":1050,
         "HINDUNILVR":2600,"ITC":460,"SBIN":780,"BAJFINANCE":7200,"KOTAKBANK":1900,
         "LT":3700,"ASIANPAINT":3100,"AXISBANK":1100,"MARUTI":12000,"TITAN":3400,
         "SUNPHARMA":1600,"ULTRACEMCO":11000,"NESTLEIND":24000,"WIPRO":600,
         "TECHM":1600,"HCLTECH":1900,"POWERGRID":340,"NTPC":380,"ONGC":280,
         "COALINDIA":490,"GRASIM":2600,"JSWSTEEL":950,"TATAMOTORS":1050,
         "TATASTEEL":160,"BAJAJFINSV":1700}

def generate_demo_csvs():
    os.makedirs(DATA_DIR, exist_ok=True)
    dates = pd.date_range("2022-01-01","2025-03-31",freq="B"); n = len(dates)
    done = 0
    for sym in NSE[:N_DEMO]:
        p = os.path.join(DATA_DIR,f"{sym}.csv")
        if os.path.exists(p): continue
        rng=np.random.default_rng(abs(hash(sym))%(2**31))
        base=BASE.get(sym,1000); drift=rng.uniform(-0.00005,0.00025)
        sig=rng.uniform(0.008,0.022)
        close=base*np.exp(np.cumsum(rng.normal(drift,sig,n)))
        open_=close*np.exp(rng.normal(0,sig*0.3,n))
        high=np.maximum(close,open_)*(1+np.abs(rng.normal(0,sig*0.4,n)))
        low =np.minimum(close,open_)*(1-np.abs(rng.normal(0,sig*0.4,n)))
        adj =close*(1+np.cumsum(rng.normal(0,0.0002,n)))
        vol =np.abs(rng.normal(2e7,8e6,n)).astype(int)
        pd.DataFrame({"date":dates.strftime("%Y-%m-%d"),
            "open":np.round(open_,2),"high":np.round(high,2),
            "low":np.round(low,2),"close":np.round(close,2),
            "adj_close":np.round(adj,2),"volume":vol}).to_csv(p,index=False)
        done+=1
    if done: print(f"  Generated {done} demo CSVs → '{DATA_DIR}/'")

def scan_stocks():
    """
    Return symbols that exist BOTH on disk (as <SYMBOL>.csv in DATA_DIR)
    AND in UNIVERSE (the SYMBOLS_CSV list). Order = order from SYMBOLS_CSV.
    Falls back to all CSVs (sorted) if UNIVERSE could not be loaded.
    """
    # Map UPPERCASE → original filename stem (handles mixed-case filesystems)
    disk_map = {os.path.splitext(os.path.basename(p))[0].upper():
                os.path.splitext(os.path.basename(p))[0]
                for p in glob.glob(os.path.join(DATA_DIR, "*.csv"))}
    if UNIVERSE is None:
        # No universe file → original behaviour
        return sorted(disk_map.values())
    keep    = [disk_map[s] for s in UNIVERSE if s in disk_map]
    missing = [s for s in UNIVERSE if s not in disk_map]
    extras  = sorted(set(disk_map) - set(UNIVERSE))
    print(f"  Universe : {len(UNIVERSE)} symbols from '{SYMBOLS_CSV}'")
    print(f"  On disk  : {len(disk_map)} CSVs in '{DATA_DIR}'")
    print(f"  Matched  : {len(keep)}  |  Missing: {len(missing)}  |  Extras (filtered out): {len(extras)}")
    if missing:
        preview = ", ".join(missing[:10]) + (" …" if len(missing) > 10 else "")
        print(f"  Missing  : {preview}")
    return keep

import functools

@functools.lru_cache(maxsize=256)
def load_stock(sym):
    df = pd.read_csv(os.path.join(DATA_DIR,f"{sym}.csv"),
                     parse_dates=["date"],index_col="date").sort_index()
    df["ret"]    = df["close"].pct_change()*100
    df["cumret"] = ((1+df["close"].pct_change()).cumprod()-1)*100
    df["up"]     = df["close"]>=df["open"]
    df["vma20"]  = df["volume"].rolling(20).mean()
    return df

# ══════════════════════════════════════════════════════════════════
# CHART BUILDER — no Plotly updatemenus; visibility set directly
# ══════════════════════════════════════════════════════════════════
PRICE_OPTS = [
    {"label":"OHLC Candle",  "value":"ohlc"},
    {"label":"Close",        "value":"close"},
    {"label":"Open",         "value":"open"},
    {"label":"Adj Close",    "value":"adj_close"},
    {"label":"Hi-Lo Band",   "value":"hiloband"},
]
STYLE_OPTS = [
    {"label":"Candlestick",  "value":"candle"},
    {"label":"Line",         "value":"line"},
]
SPEED_OPTS = [
    {"label":"0.5s","value":500},{"label":"1s","value":1000},
    {"label":"2s","value":2000},{"label":"3s","value":3000},
    {"label":"5s","value":5000},
]

def _visibility(price_type, chart_style):
    """Return 10-element visibility list based on selected dropdowns."""
    # Traces 0-5 switchable, 6-9 always on

    mapping = {
        ("ohlc",  "candle"): [0],
        ("ohlc",  "line"):   [1],     # line mode of OHLC → close line
        ("close", "candle"): [1],
        ("close", "line"):   [1],
        ("open",  "candle"): [2],
        ("open",  "line"):   [2],
        ("adj_close","candle"):[3],
        ("adj_close","line"):  [3],
        ("hiloband","candle"): [4,5],
        ("hiloband","line"):   [4,5],
    }

    on = mapping.get((price_type, chart_style), [0])
    return [i in on for i in range(6)] + [True]*4

def build_figure(df, symbol, price_type="ohlc", chart_style="candle",
                 view_start=None, view_end=None):
    vis = _visibility(price_type, chart_style)

    # ── Determine the visible window for axis-range computation ──
    if view_start is not None and view_end is not None:
        win = df.loc[(df.index >= view_start) & (df.index <= view_end)]
        if len(win) < 2:
            win = df  # window too narrow → fall back to full data
    else:
        win = df

    pmin = win["low"].min()*0.999; pmax = win["high"].max()*1.001
    prng = pmax-pmin
    vmax = win["volume"].max() or df["volume"].max() or 1
    vscl = (prng*VOL_FRAC)/vmax
    df   = df.copy()
    df["vs"]  = df["volume"]*vscl+pmin
    df["vms"] = df["vma20"]*vscl+pmin

    # ── Daily / cumulative return ranges from visible window ──
    rmin = float(win["ret"].min()); rmax = float(win["ret"].max())
    rpad = max(abs(rmin), abs(rmax))*0.18 + 0.05      # symmetric headroom
    yaxis2_range = [rmin-rpad, rmax+rpad]

    cmin = float(win["cumret"].min()); cmax = float(win["cumret"].max())
    cpad = (cmax-cmin)*0.06 + 0.5
    yaxis3_range = [cmin-cpad, cmax+cpad]

    vc = [C["bull"] if u else C["bear"] for u in df["up"]]
    rc = [C["bull"] if r>=0 else C["bear"] for r in df["ret"].fillna(0)]

    fig = make_subplots(rows=2,cols=1,shared_xaxes=True,
                        row_heights=[0.65,0.35],vertical_spacing=0.03)

    # 0: OHLC Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,open=df["open"],high=df["high"],low=df["low"],close=df["close"],
        name="OHLC",
        increasing=dict(line=dict(color=C["bull"],width=1),fillcolor=C["bull"]),
        decreasing=dict(line=dict(color=C["bear"],width=1),fillcolor=C["bear"]),
        whiskerwidth=0.5,visible=vis[0],showlegend=True,
    ),row=1,col=1)
    # 1: Close line
    fig.add_trace(go.Scatter(x=df.index,y=df["close"],name="Close",
        line=dict(color=C["blue"],width=1.7),visible=vis[1],
        hovertemplate="Close: ₹%{y:,.2f}<extra></extra>"),row=1,col=1)
    # 2: Open line
    fig.add_trace(go.Scatter(x=df.index,y=df["open"],name="Open",
        line=dict(color=C["purple"],width=1.7),visible=vis[2],
        hovertemplate="Open: ₹%{y:,.2f}<extra></extra>"),row=1,col=1)
    # 3: Adj Close line
    fig.add_trace(go.Scatter(x=df.index,y=df["adj_close"],name="Adj Close",
        line=dict(color=C["pink"],width=1.7),visible=vis[3],
        hovertemplate="Adj Close: ₹%{y:,.2f}<extra></extra>"),row=1,col=1)
    # 4: High (band top)
    fig.add_trace(go.Scatter(x=df.index,y=df["high"],name="High",
        line=dict(color=C["teal"],width=0.8),visible=vis[4],
        hovertemplate="High: ₹%{y:,.2f}<extra></extra>"),row=1,col=1)
    # 5: Low (band fill)
    fig.add_trace(go.Scatter(x=df.index,y=df["low"],name="Low",
        line=dict(color=C["teal"],width=0.8),fill="tonexty",
        fillcolor="rgba(128,203,196,0.15)",visible=vis[5],
        hovertemplate="Low: ₹%{y:,.2f}<extra></extra>"),row=1,col=1)
    # 6: Volume bars (always on)
    fig.add_trace(go.Bar(x=df.index,y=df["vs"]-pmin,base=pmin,name="Volume",
        marker=dict(color=vc,opacity=0.32),
        customdata=df["volume"]/1e7,
        hovertemplate="Vol: %{customdata:.2f} Cr<extra></extra>"),row=1,col=1)
    # 7: Vol MA-20 (always on)
    fig.add_trace(go.Scatter(x=df.index,y=df["vms"],name="Vol MA-20",
        line=dict(color=C["yellow"],width=1.0),
        customdata=df["vma20"]/1e7,
        hovertemplate="Vol MA20: %{customdata:.2f} Cr<extra></extra>"),row=1,col=1)
    # 8: Cum return, right axis (always on)
    fig.add_trace(go.Scatter(x=df.index,y=df["cumret"],name="Cum. Return %",
        line=dict(color=C["orange"],width=1.4,dash="dot"),yaxis="y3",
        hovertemplate="Cum Ret: %{y:.2f}%<extra></extra>"),row=1,col=1)
    # 9: Daily return bars (always on)
    fig.add_trace(go.Bar(x=df.index,y=df["ret"],name="Daily Return %",
        marker=dict(color=rc,opacity=0.95,
                    line=dict(color=rc,width=0)),
        width=24*60*60*1000*0.85,   # 85% of a day → solid bars on date axis
        hovertemplate="%{x|%d %b %Y}<br>Ret: %{y:.3f}%<extra></extra>"),row=2,col=1)

    last  = df["close"].iloc[-1]
    tcum  = df["cumret"].iloc[-1]
    sec   = SECTOR.get(symbol,"NSE")
    color = C["bull"] if tcum>=0 else C["bear"]
    sign  = "+" if tcum>=0 else ""

    # Build xaxis kwargs — embed the view window when caller specified one
    xaxis_kwargs = dict(
        type="date",gridcolor=C["grid"],linecolor=C["border2"],
        showspikes=True,spikecolor=C["spike"],
        spikethickness=1,spikedash="dash",spikemode="across",
        rangeselector=dict(
            x=1.0,y=1.0,xanchor="right",yanchor="bottom",
            bgcolor=C["btn"],bordercolor=C["border2"],
            font=dict(color=C["font"],size=10),
            buttons=[
                dict(count=1,label="1M",step="month",stepmode="backward"),
                dict(count=3,label="3M",step="month",stepmode="backward"),
                dict(count=6,label="6M",step="month",stepmode="backward"),
                dict(count=1,label="1Y",step="year", stepmode="backward"),
                dict(count=2,label="2Y",step="year", stepmode="backward"),
                dict(step="all",label="All"),
            ]),
        rangeslider=dict(visible=True,thickness=0.045,
                         bgcolor=C["hdr"],bordercolor=C["border2"]),
    )
    if view_start is not None and view_end is not None:
        xaxis_kwargs["range"] = [view_start, view_end]

    fig.update_layout(
        title=dict(
            text=(f"<b>{symbol}</b>"
                  f"<span style='color:#90a4ae;font-size:12px'>  {sec}</span>"
                  f"<span style='color:{C['blue']};font-size:13px'>  ₹{last:,.2f}</span>"
                  f"<span style='color:{color};font-size:12px'>  {sign}{tcum:.1f}% total</span>"),
            font=dict(size=14,color=C["font"]),x=0.5,xanchor="center",y=0.99),
        height=720, template="plotly_dark",
        paper_bgcolor=C["paper"], plot_bgcolor=C["bg"],
        hovermode="x unified",
        uirevision=symbol,    # persist legend / modebar state across rebuilds for same symbol
        hoverlabel=dict(bgcolor="#0f1923",bordercolor=C["spike"],
                        font=dict(color=C["font"],size=11),namelength=-1),
        legend=dict(orientation="h",x=0,y=1.045,
                    bgcolor="rgba(10,15,22,0.9)",bordercolor=C["border2"],
                    borderwidth=1,font=dict(color=C["font"],size=10)),
        margin=dict(l=65,r=85,t=100,b=40),
        xaxis=xaxis_kwargs,
        xaxis2=dict(type="date",gridcolor=C["grid"],linecolor=C["border2"],
                    showspikes=True,spikecolor=C["spike"],
                    spikethickness=1,spikedash="dash",spikemode="across"),
        yaxis=dict(title=dict(text="Price (₹)",font=dict(color=C["blue"],size=11)),
                   tickformat=",.0f",tickprefix="₹",gridcolor=C["grid"],
                   tickfont=dict(color=C["blue"]),
                   showspikes=True,spikecolor=C["spike"],spikethickness=1,
                   range=[pmin-prng*0.02,pmax+prng*0.06],zeroline=False),
        yaxis2=dict(title=dict(text="Return %",font=dict(color=C["font"],size=11)),
                    tickformat=".2f",ticksuffix="%",gridcolor=C["grid"],
                    tickfont=dict(color=C["font"]),
                    showspikes=True,spikecolor=C["spike"],spikethickness=1,
                    range=yaxis2_range,
                    zeroline=True,zerolinecolor=C["spike"],zerolinewidth=1),
        yaxis3=dict(title=dict(text="Cum. Return %",font=dict(color=C["orange"],size=11)),
                    overlaying="y",side="right",tickformat=".1f",ticksuffix="%",
                    showgrid=False,tickfont=dict(color=C["orange"]),zeroline=False,
                    range=yaxis3_range),
    )
    fig.add_hline(y=0,line_color=C["spike"],line_width=0.8,row=2,col=1)
    fig.add_hline(y=pmin+prng*VOL_FRAC,line_color=C["border"],
                  line_width=0.6,line_dash="dot",row=1,col=1)
    return fig

# ══════════════════════════════════════════════════════════════════
# DASH APP
# ══════════════════════════════════════════════════════════════════
generate_demo_csvs()
# Merge sectors from the universe CSV (universe takes precedence for new symbols,
# hardcoded SECTOR retains the short labels the dashboard already uses)
for _s, _i in UNIV_SECTORS.items():
    SECTOR.setdefault(_s, _i)
STOCKS = scan_stocks()
assert STOCKS, f"No CSVs found matching universe in '{DATA_DIR}/'"
print(f"  Loaded {len(STOCKS)} stocks  |  http://127.0.0.1:{APP_PORT}")

app = dash.Dash(__name__,title="QBEAST AI — Stock Cycler",update_title=None)

# ── Dark theme CSS for dcc.Dropdown and compact UI ───────────────
DARK_CSS = f"""
* {{ box-sizing: border-box; margin:0; padding:0; }}
body {{ background:{C['bg']}; color:{C['font']}; font-family:'JetBrains Mono',
       'Fira Code','Courier New',monospace; overflow-x:hidden; }}
/* Dropdown container */
.dd-dark .Select-control {{
    background:{C['btn']} !important; border:1px solid {C['border2']} !important;
    border-radius:5px !important; min-height:28px !important;
    height:28px !important; cursor:pointer !important;
}}
.dd-dark .Select-control:hover {{ border-color:{C['blue']} !important; }}
.dd-dark .Select-value {{
    line-height:28px !important; padding-left:8px !important;
}}
.dd-dark .Select-value-label {{ color:{C['font']} !important; font-size:12px !important; }}
.dd-dark .Select-placeholder {{ color:{C['sub']} !important;
    font-size:12px !important; line-height:28px !important; padding-left:8px; }}
.dd-dark .Select-input > input {{ color:{C['font']} !important;
    font-size:12px !important; background:transparent !important; }}
.dd-dark .Select-arrow-zone {{ padding-top:4px; }}
.dd-dark .Select-arrow {{ border-color:{C['sub']} transparent transparent !important; }}
.dd-dark .is-open .Select-control {{ border-color:{C['blue']} !important;
    border-radius:5px 5px 0 0 !important; }}
.dd-dark .Select-menu-outer {{
    background:{C['btn']} !important; border:1px solid {C['blue']} !important;
    border-top:none !important; border-radius:0 0 6px 6px !important;
    box-shadow:0 6px 24px rgba(0,0,0,0.6) !important; z-index:9999 !important;
}}
.dd-dark .Select-option {{
    background:{C['btn']} !important; color:{C['font']} !important;
    font-size:12px !important; padding:7px 10px !important; cursor:pointer;
}}
.dd-dark .Select-option:hover, .dd-dark .Select-option.is-focused {{
    background:{C['act_blue']} !important; color:white !important;
}}
.dd-dark .Select-option.is-selected {{
    background:#1e3a5f !important; color:{C['blue']} !important;
}}
/* Scrollbar */
::-webkit-scrollbar {{ width:5px; height:5px; }}
::-webkit-scrollbar-track {{ background:{C['bg']}; }}
::-webkit-scrollbar-thumb {{ background:{C['border2']}; border-radius:3px; }}
"""

app.index_string = f"""<!DOCTYPE html>
<html><head>
  {{%metas%}}<title>{{%title%}}</title>{{%favicon%}}{{%css%}}
  <style>{DARK_CSS}</style>
</head>
<body>{{%app_entry%}}
<footer>{{%config%}}{{%scripts%}}{{%renderer%}}</footer>
</body></html>"""

# ── Shared styles ─────────────────────────────────────────────────
HDR  = {"display":"flex","alignItems":"flex-end","gap":"10px",
        "background":C["hdr"],"padding":"8px 14px 7px",
        "borderBottom":f"1px solid {C['border']}","flexWrap":"nowrap",
        "minHeight":"52px"}
LBL  = {"color":C["sub"],"fontSize":"10px","marginBottom":"3px",
        "letterSpacing":"0.5px","textTransform":"uppercase"}
VDIV = {"width":"1px","height":"32px","background":C["border"],
        "alignSelf":"center","flexShrink":"0"}

def pbtn(label, bid, color=None, title=""):
    """Compact playback button."""
    bg = color or C["btn"]
    return html.Button(label, id=bid, n_clicks=0, title=title, style={
        "background":bg,"color":C["font"],"border":f"1px solid {C['border2']}",
        "borderRadius":"5px","padding":"5px 11px","cursor":"pointer",
        "fontSize":"12px","fontFamily":"inherit","whiteSpace":"nowrap",
        "transition":"background 0.15s",
    })

def spd_btn(opt, i, active):
    return html.Button(opt["label"],
        id={"type":"spd","index":i}, n_clicks=0, style={
        "background":C["act_purple"] if active else C["btn"],
        "color":C["font"],"border":f"1px solid {C['border2']}",
        "borderRadius":"4px","padding":"4px 8px","cursor":"pointer",
        "fontSize":"11px","fontFamily":"inherit",
        "transition":"background 0.15s",
    })

def dd(id_, options, value, width, className="dd-dark"):
    return dcc.Dropdown(id=id_, options=options, value=value,
        clearable=False, searchable=False, className=className,
        style={"width":width,"fontSize":"12px"})

# ── Layout ────────────────────────────────────────────────────────
app.layout = html.Div(style={"background":C["bg"],"minHeight":"100vh"}, children=[

    # State stores
    dcc.Store(id="s-idx",     data=0),
    dcc.Store(id="s-playing", data=False),
    dcc.Store(id="s-speed",   data=2000),
    dcc.Interval(id="iv",interval=2000,n_intervals=0,disabled=True),

    # ══ SINGLE HEADER ROW ════════════════════════════════════════
    html.Div(style=HDR, children=[

        # Logo
        html.Div("QBEAST AI", style={
            "color":C["orange"],"fontSize":"15px","fontWeight":"bold",
            "letterSpacing":"1px","paddingBottom":"2px","flexShrink":"0",
        }),

        html.Div(style=VDIV),

        # Stock
        html.Div([html.Div("Stock", style=LBL),
                  dd("dd-stock",[{"label":s,"value":s} for s in STOCKS],
                     STOCKS[0],"170px","dd-dark")]),

        # Price Type dropdown
        html.Div([html.Div("Price Type", style=LBL),
                  dd("dd-price", PRICE_OPTS, "ohlc", "148px")]),

        # Style dropdown
        html.Div([html.Div("Style", style=LBL),
                  dd("dd-style", STYLE_OPTS, "candle", "128px")]),

        html.Div(style=VDIV),

        # Playback
        html.Div([
            html.Div("Playback", style=LBL),
            html.Div(style={"display":"flex","gap":"5px"}, children=[
                pbtn("⏮", "btn-prev", title="Previous"),
                pbtn("▶  Play", "btn-play", color=C["act_green"], title="Play / Pause"),
                pbtn("⏭", "btn-next", title="Next"),
            ]),
        ]),

        html.Div(style=VDIV),

        # Speed
        html.Div([
            html.Div("Speed / stock", style=LBL),
            html.Div(id="spd-row", style={"display":"flex","gap":"4px"},
                     children=[spd_btn(o,i,o["value"]==2000)
                                for i,o in enumerate(SPEED_OPTS)]),
        ]),

        html.Div(style=VDIV),

        # Progress (right-aligned)
        html.Div(id="prog-txt", style={
            "marginLeft":"auto","textAlign":"right","flexShrink":"0",
            "lineHeight":"1.55","paddingBottom":"2px",
        }),
    ]),

    # ══ PROGRESS BAR ══════════════════════════════════════════════
    html.Div(style={"background":C["hdr"],"padding":"0 14px 5px"}, children=[
        html.Div(style={"background":C["btn"],"borderRadius":"3px",
                         "height":"4px","overflow":"hidden"}, children=[
            html.Div(id="prog-bar",style={
                "background":f"linear-gradient(90deg,{C['act_blue']},{C['blue']})",
                "height":"100%","borderRadius":"3px","transition":"width 0.35s ease",
                "width":f"{100/max(len(STOCKS),1):.1f}%",
            }),
        ]),
    ]),

    # ══ CHART ════════════════════════════════════════════════════
    html.Div(style={"padding":"0 6px"}, children=[
        dcc.Graph(id="chart",
                  config={"displayModeBar":True,"responsive":True,
                          "displaylogo":False,
                          "modeBarButtonsToRemove":["lasso2d","select2d"]},
                  style={"height":"748px"}),
    ]),

    # ══ FOOTER ═══════════════════════════════════════════════════
    html.Div(style={"textAlign":"center","padding":"6px",
                    "color":C["sub"],"fontSize":"10px",
                    "borderTop":f"1px solid {C['border']}"},
             children=[f"QBEAST AI  •  {len(STOCKS)} stocks  •  "
                       "Drag chart slider to zoom  •  "
                       "Hover for unified crosshair values"]),
])

# ══════════════════════════════════════════════════════════════════
# CALLBACKS
# ══════════════════════════════════════════════════════════════════

# ── A: Play / Pause ───────────────────────────────────────────────
@app.callback(
    Output("iv","disabled"), Output("iv","interval"),
    Output("btn-play","children"), Output("btn-play","style"),
    Output("s-playing","data"),
    Input("btn-play","n_clicks"),
    State("s-playing","data"), State("s-speed","data"),
    prevent_initial_call=True,
)
def toggle_play(_, playing, speed):
    new = not playing
    lbl = "⏸  Pause" if new else "▶  Play"
    bg  = "#c62828" if new else C["act_green"]
    sty = {"background":bg,"color":C["font"],
           "border":f"1px solid {C['border2']}","borderRadius":"5px",
           "padding":"5px 11px","cursor":"pointer","fontSize":"12px",
           "fontFamily":"inherit","whiteSpace":"nowrap"}
    return (not new), speed, lbl, sty, new

# ── B: Speed buttons ──────────────────────────────────────────────
@app.callback(
    Output("s-speed","data"),
    Output("iv","interval",allow_duplicate=True),
    Output("spd-row","children"),
    Input({"type":"spd","index":ALL},"n_clicks"),
    prevent_initial_call=True,
)
def set_speed(clicks):
    trig = callback_context.triggered[0]["prop_id"]
    idx  = json.loads(trig.split(".")[0])["index"]
    ms   = SPEED_OPTS[idx]["value"]
    btns = [spd_btn(o,i,i==idx) for i,o in enumerate(SPEED_OPTS)]
    return ms, ms, btns

# ── C: Main — stock navigation + chart rebuild ────────────────────
@app.callback(
    Output("chart","figure"),
    Output("s-idx","data"),
    Output("dd-stock","value"),
    Output("prog-txt","children"),
    Output("prog-bar","style"),
    Input("btn-prev","n_clicks"),
    Input("btn-next","n_clicks"),
    Input("iv","n_intervals"),
    Input("dd-stock","value"),
    Input("dd-price","value"),
    Input("dd-style","value"),
    State("s-idx","data"),
    State("s-playing","data"),
)
def update(prev_n, next_n, n_iv, dd_sym, price_type, chart_style, idx, playing):
    trig = callback_context.triggered[0]["prop_id"]
    n    = len(STOCKS)

    # Determine new stock index
    if "dd-stock"  in trig: new_idx = STOCKS.index(dd_sym) if dd_sym in STOCKS else idx
    elif "btn-prev" in trig: new_idx = (idx-1)%n
    elif "btn-next" in trig or "iv" in trig: new_idx = (idx+1)%n
    else: new_idx = idx   # price_type or style changed — redraw same stock

    sym = STOCKS[new_idx]
    df  = load_stock(sym)
    fig = build_figure(df, sym, price_type or "ohlc", chart_style or "candle")

    # Progress UI
    pct = (new_idx+1)/n*100
    state_label = "▶ Playing" if playing else "⏸ Paused"
    prog = html.Div([
        html.Div(f"Stock  {new_idx+1:>4d} / {n}",
                 style={"color":C["blue"],"fontWeight":"bold","fontSize":"12px"}),
        html.Div(f"{state_label}  •  {sym}",
                 style={"color":C["sub"],"fontSize":"11px"}),
    ])
    bar = {"background":f"linear-gradient(90deg,{C['act_blue']},{C['blue']})",
           "height":"100%","borderRadius":"3px",
           "transition":"width 0.35s ease","width":f"{pct:.1f}%"}

    return fig, new_idx, sym, prog, bar

# ── D: Autoscale Y on x-zoom (rangeselector / box / rangeslider) ──
@app.callback(
    Output("chart","figure", allow_duplicate=True),
    Input("chart","relayoutData"),
    State("dd-stock","value"),
    State("dd-price","value"),
    State("dd-style","value"),
    prevent_initial_call=True,
)
def autoscale_on_xzoom(relayout, sym, price_type, chart_style):
    """Rebuild the figure for the visible x-window so price, return,
    cum-return y-axes — and volume-bar scaling — track the zoom."""
    if not relayout or not sym:
        raise PreventUpdate

    view_start = view_end = None

    # rangeselector buttons (1M/3M/6M/1Y/2Y) and box-zoom
    if "xaxis.range[0]" in relayout and "xaxis.range[1]" in relayout:
        view_start = pd.to_datetime(relayout["xaxis.range[0]"])
        view_end   = pd.to_datetime(relayout["xaxis.range[1]"])
    # rangeslider drag
    elif ("xaxis.range" in relayout
          and isinstance(relayout["xaxis.range"], (list, tuple))
          and len(relayout["xaxis.range"]) == 2):
        view_start = pd.to_datetime(relayout["xaxis.range"][0])
        view_end   = pd.to_datetime(relayout["xaxis.range"][1])
    # "All" / double-click reset → full range (view_start/end stay None)
    elif relayout.get("xaxis.autorange") is True:
        pass
    else:
        # legend toggles, hover events, modebar clicks, etc. → ignore
        raise PreventUpdate

    df  = load_stock(sym)
    fig = build_figure(df, sym, price_type or "ohlc", chart_style or "candle",
                       view_start=view_start, view_end=view_end)
    return fig

# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print(f"\n{'═'*54}")
    print(f"  QBEAST AI — Stock Cycler  v2")
    print(f"  Stocks : {len(STOCKS)}  |  http://127.0.0.1:{APP_PORT}")
    print(f"{'═'*54}\n")
    app.run(host="0.0.0.0", debug=False,port=APP_PORT)