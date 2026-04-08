import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import time
from datetime import timedelta
import google.generativeai as genai

def calculate_technical_signal(hist):
    """
    核心算力：三重技术面共振算法 (SMA20 + RSI + MACD)
    """
    if hist.empty or len(hist) < 30:
        return 0.5, "⚪ 数据不足"
    
    close = hist['Close']
    current_price = close.iloc[-1]
    
    # 1. 计算 20日均线 (短期生命线)
    sma20 = close.rolling(window=20).mean().iloc[-1]
    
    # 2. 计算 14日 RSI (相对强弱指标，防追高)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=14, min_periods=1).mean().iloc[-1]
    avg_loss = loss.rolling(window=14, min_periods=1).mean().iloc[-1]
    rsi = 100.0 if avg_loss == 0 else 100.0 - (100.0 / (1.0 + (avg_gain / avg_loss)))
        
    # 3. 计算 MACD (动能加速器)
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    current_macd = macd_line.iloc[-1]
    current_signal = signal_line.iloc[-1]
    
    # === 🚀 核心共振判断逻辑 ===
    # 强烈买入：站上20日线 + 还没严重超买(RSI<70) + MACD金叉向上
    if current_price > sma20 and rsi < 70 and current_macd > current_signal:
        return 0.9, "💎 强烈买入"
    # 强烈卖出：跌破20日线 + 还没严重超卖(RSI>30) + MACD死叉向下
    elif current_price < sma20 and rsi > 30 and current_macd < current_signal:
        return 0.1, "🔴 强烈卖出"
    # 震荡偏多：仅仅是站上20日线
    elif current_price > sma20:
        return 0.7, "🟢 偏多"
    # 震荡偏空：仅仅是跌破20日线
    elif current_price < sma20:
        return 0.3, "🟠 偏空"
    else:
        return 0.5, "⚪ 震荡观望"
        
# ==========================================
# 1. 全局配置与多语言引擎 (i18n)
# ==========================================
st.set_page_config(page_title="AI Quant Terminal v3.7", layout="wide", page_icon="🌌")

# 侧边栏：语言切换器
lang_choice = st.sidebar.radio("🌐 Language / 语言", ["中文", "English"], horizontal=True)
is_cn = (lang_choice == "中文")

# 翻译闭包函数：一行代码搞定双语切换
_t = lambda cn, en: cn if is_cn else en


def apply_cyber_theme():
    st.markdown("""
    <style>
    /* 1. 动态网格背景 (无限滑动) */
    .stApp {
        background-color: #050812;
        background-image: 
            linear-gradient(rgba(0, 210, 255, 0.07) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 210, 255, 0.07) 1px, transparent 1px);
        background-size: 30px 30px;
        animation: GridMove 10s linear infinite;
        color: #E2E8F0;
    }
    @keyframes GridMove {
        0% { background-position: 0px 0px; }
        100% { background-position: 30px 30px; }
    }

    /* 2. 隐藏水印与基础排版 */
    [data-testid="stHeader"] { background-color: transparent; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .block-container {padding-top: 1.5rem; max-width: 96%;}
    div[data-testid="stMetricValue"] { font-size: 2.2rem; font-weight: 800; color: #00D2FF; text-shadow: 0px 0px 15px rgba(0, 210, 255, 0.5); }
    
    /* 3. 保留磨砂玻璃特效 (Glassmorphism) */
    [data-testid="stSidebar"] {
        background-color: rgba(5, 8, 18, 0.75) !important; 
        border-right: 1px solid rgba(0, 210, 255, 0.2);
        backdrop-filter: blur(12px); 
    }
    div.info-card { 
        background-color: rgba(11, 15, 25, 0.6); border: 1px solid rgba(0, 210, 255, 0.2); 
        border-radius: 8px; padding: 15px; margin-bottom: 10px; backdrop-filter: blur(8px); 
    }
    .ticker-wrap { 
        width: 100%; overflow: hidden; background-color: rgba(5, 8, 18, 0.8); 
        border: 1px solid rgba(0, 210, 255, 0.2); border-radius: 8px; padding: 12px; margin-top: 20px; 
        backdrop-filter: blur(8px); 
    }
    div.stDataFrame {border: 1px solid rgba(0, 210, 255, 0.2); border-radius: 8px;}
    .stChatMessage.assistant {background-color: rgba(0, 210, 255, 0.05) !important; border-left: 3px solid #00D2FF;}
    .ticker { display: inline-block; white-space: nowrap; padding-right: 100%; box-sizing: content-box; animation: ticker 25s linear infinite; }
    @keyframes ticker { 0% { transform: translate3d(0, 0, 0); } 100% { transform: translate3d(-100%, 0, 0); } }
    </style>
    """, unsafe_allow_html=True)
    
apply_cyber_theme()

# ==========================================
# 2. 动态字段映射配置
# ==========================================
C_SYM = _t('代码', 'Symbol')
C_NAME = _t('标的', 'Name')
C_PRICE = _t('最新价', 'Price')
C_ROE = 'ROE(%)'
C_PE = _t('市盈率(PE)', 'PE Ratio')
C_NET = _t('净利率(%)', 'Net Margin(%)')
C_REV = _t('营收增长(%)', 'Rev Growth(%)')
C_SIG = _t('AI信号', 'AI Signal')
C_SEC = _t('板块', 'Sector')
C_CAP = _t('市值(亿)', 'Mkt Cap(B)')
C_TREND = _t('7日走势', '7-Day Trend')

MARKET_PRESETS = {
    _t("🇺🇸 纳斯达克巨头 (科技核心)", "🇺🇸 NASDAQ Giants (Tech Core)"): ["AAPL", "NVDA", "MSFT", "GOOGL", "AMZN", "TSLA",
                                                                       "META"],
    _t("🇲🇾 马股 KLCI 蓝筹 (马来核心)", "🇲🇾 KLCI Blue Chips (Malaysia)"): ["1155.KL", "1023.KL", "1295.KL", "5347.KL",
                                                                          "6888.KL", "1066.KL", "5183.KL"],
    _t("🇨🇳 中国核心资产", "🇨🇳 China Core Assets"): ["601857.SS", "000002.SZ", "600519.SS", "000725.SZ"],
    _t("🚀 仅看我的自定义自选", "🚀 Custom Watchlist Only"): []
}


# ==========================================
# 3. 核心引擎 (缓存隔离，语言切换不重载)
# ==========================================
@st.cache_data(ttl=600)
def _fetch_global_data_cached(symbols_list):
    if not symbols_list: return pd.DataFrame()
    data = []
    for s in symbols_list[:25]:
            try:
                t = yf.Ticker(s)
                info = t.info
                
                # 1. 真实共振信号计算
                hist_data = t.history(period="3mo")
                signal_score, signal_text = calculate_technical_signal(hist_data)
                
                # 👇 === 🚀 雷达核心：主力资金天量爆破计算 === 👇
                try:
                    today_vol = hist_data['Volume'].iloc[-1]
                    avg_vol_10 = hist_data['Volume'].iloc[-11:-1].mean() # 提取过去10天平均量
                    vol_ratio = today_vol / avg_vol_10 if avg_vol_10 > 0 else 0
                except:
                    vol_ratio = 0

                # 智能命名系统：如果有巨鲸，直接在名字前面加刺眼的高亮标签！
                base_name = info.get('shortName', s)
                final_name = f"🐋 [量爆 {vol_ratio:.1f}x] {base_name}" if vol_ratio >= 2.0 else base_name
                # 👆 ========================================== 👆

                # 2. 装载进数据总线
                data.append({
                    'sym': s, 
                    'name': final_name, # 👈 注意！这里替换成了带有雷达标签的名字
                    'price': info.get('currentPrice', info.get('previousClose', 0)),
                    'roe': round(info.get('returnOnEquity', 0) * 100, 2) if info.get('returnOnEquity') else 0,
                    'pe': round(info.get('trailingPE', 0), 2) if info.get('trailingPE') else 0,
                    'net': round(info.get('profitMargins', 0) * 100, 2) if info.get('profitMargins') else 0,
                    'rev': round(info.get('revenueGrowth', 0) * 100, 2) if info.get('revenueGrowth') else 0,
                    'sig': signal_text,  
                    'sec': info.get('sector', 'Unknown'),
                    'cap': info.get('marketCap', 0) / 100000000 if info.get('marketCap') else 0
                })
            except:
                continue
    return pd.DataFrame(data)


def fetch_global_data(symbols_list):
    df = _fetch_global_data_cached(symbols_list)
    if df.empty: return df
    # 根据当前语言映射列名
    return df.rename(
        columns={'sym': C_SYM, 'name': C_NAME, 'price': C_PRICE, 'roe': C_ROE, 'pe': C_PE, 'net': C_NET, 'rev': C_REV,
                 'sig': C_SIG, 'sec': C_SEC, 'cap': C_CAP})


# ==========================================
# 4. 侧边栏导航
# ==========================================
st.sidebar.title(_t("🌌 Cham AI 量化中枢 v3.7", "🌌 Cham AI Quant Terminal v3.7"))
st.sidebar.markdown("### 👨‍💼 Manager: Cham Jun Ee")
st.sidebar.markdown("---")

market_choice = st.sidebar.selectbox(_t("🛰️ 切换全球监控频道", "🛰️ Switch Global Channel"), list(MARKET_PRESETS.keys()))
custom_input = st.sidebar.text_input(_t("📡 注入额外代码 (马股加 .KL)", "📡 Inject Custom Tickers (e.g. 1155.KL)"), "")
custom_list = [x.strip().upper() for x in custom_input.split(",") if x.strip()]

final_scan_list = MARKET_PRESETS[market_choice] + custom_list

P1 = _t("1. 信号监测矩阵", "1. Signal Monitor Matrix")
P2 = _t("2. 基本面多维扫描", "2. Fundamental Scan")
P3 = _t("3. 深度财务解析", "3. Deep Financials")
P4 = _t("4. 策略回测引擎", "4. Backtest Engine")
P5 = _t("5. 全球情绪雷达", "5. Global Sentiment Radar")
P6 = _t("6. AI 智能助手", "6. AI Assistant")
P7 = _t("7. AI 预测与止盈引擎", "7. AI Target & Catalyst Engine")

page = st.sidebar.radio(_t("系统链路", "System Links"), [P1, P2, P3, P4, P5, P6, P7])

if st.sidebar.button(_t("⚡ 强制同步全球数据", "⚡ Sync Global Data"), use_container_width=True):
    with st.spinner(_t('📡 正在通过卫星链路抓取数据...', '📡 Fetching data via satellite link...')):
        st.cache_data.clear()
        time.sleep(1)
    st.sidebar.success(_t("同步完成！系统已处于最新状态。", "Sync Complete! System is up-to-date."))

# ==========================================
# 页面 1：信号监测矩阵
# ==========================================
if page == P1:
    st.title(_t("🛰️ 实时信号监测矩阵", "🛰️ Real-Time Signal Monitor Matrix"))

    with st.spinner(_t("🛸 正在建立全球数据链路...", "🛸 Establishing global data link...")):
        raw_df = fetch_global_data(final_scan_list)

    if not raw_df.empty:
        with st.expander(_t("🔍 开启多维筛选雷达", "🔍 Open Multi-Dim Filter Radar"), expanded=True):
            f1, f2, f3 = st.columns(3)
            with f1: 
                selected_sectors = st.multiselect(_t("筛选板块", "Filter Sectors"), raw_df[C_SEC].unique(), default=raw_df[C_SEC].unique())
            with f2: 
                min_roe = st.slider(_t("最低 ROE (%)", "Min ROE (%)"), 0.0, 50.0, 0.0)
            
            # === 拆解变量，彻底解决报错，并更新为 V4.0 高级文本标签 ===
            sig_all = _t("全部", "All")
            sig_buy = _t("仅看买入", "Buy Only")
            sig_sell = _t("排除卖出", "Exclude Sell")
            
            with f3: 
                sig_filter = st.selectbox(_t("信号过滤", "Signal Filter"), [sig_all, sig_buy, sig_sell])
        
        # === 数据多维过滤引擎 ===
        df = raw_df[raw_df[C_SEC].isin(selected_sectors) & (raw_df[C_ROE] >= min_roe)]
        
if sig_filter == sig_buy:
            df = df[df[C_SIG].str.contains("买入|Buy", na=False)]
elif sig_filter == sig_sell:
            df = df[~df[C_SIG].str.contains("卖出|Sell", na=False)] 
            
# 👇 确保 st.markdown 和上面的 if 是左边完全垂直对齐的！
st.markdown("###")
c1, c2, c3, c4 = st.columns(4)

c1.metric(_t("匹配节点", "Matched Nodes"), len(df))
c2.metric(_t("平均ROE", "Avg ROE"), f"{df[C_ROE].mean():.2f}%" if not df.empty else "0%")

# === AI 情绪卡片 ===
if not df.empty:
    bull_count = df[C_SIG].str.contains("买入|偏多|Buy", na=False).sum()
    bear_count = df[C_SIG].str.contains("卖出|偏空|Sell", na=False).sum()
    sentiment = _t("🟢 偏多", "🟢 Bullish") if bull_count >= bear_count else _t("🔴 偏淡", "🔴 Bearish")
else:
    sentiment = "⚪ 数据不足"
    
c3.metric(_t("AI 情绪", "AI Sentiment"), sentiment)

with c4:
    if st.button(_t("🚀 启动异动监控扫描", "🚀 Start Anomaly Scan")):
        with st.status(_t("🛸 扫描中...", "🛸 Scanning..."), expanded=False) as status:
            for _, row in df.sample(min(3, len(df))).iterrows():
                import time
                time.sleep(0.8)
                # === 修复 3：连扫描按钮也要认识文字！ ===
                if "买入" in str(row[C_SIG]) or "Buy" in str(row[C_SIG]):
                    st.toast(_t(f"🚨 异动预警: {row[C_NAME]} 评级上调为‘强烈买入’！", 
                               f"🚨 Alert: {row[C_NAME]} upgraded to 'Strong Buy'!"), icon='💎')
            status.update(label=_t("扫描完成", "Scan Complete"), state="complete", expanded=False)

st.markdown("---")

if not df.empty:
    st.subheader(_t("🗺️ 资金流向与信号热力图", "🗺️ Fund Flow & Signal Treemap"))
    
    # === 修复颜色系统：教热力图认识咱们的新文字信号 ===
    # === 1. 极简主义颜色系统：只看最强和最弱 ===
    color_map = {
        "💎 强烈买入": "#064E3B",   # 极深绿 (Deep Emerald)
        "🔴 强烈卖出": "#991B1B",   # 铁血红 (Strong Red)
        "🟢 偏多": "#1E293B",      # 暗夜灰
        "⚪ 震荡观望": "#1E293B",
        "⚪ 数据不足": "#1E293B",
        "🟠 偏空": "#1E293B"       # 暗夜灰
    }

    # === 2. 🚀 最后一枚勋章：板块轮动温度计 ===
    st.markdown("###")
    st.subheader(_t("🌡️ 板块实时热度监控", "🌡️ Sector Rotation Heat"))
    
    # 统计每个板块的强烈买入比例
    sector_stats = []
    for sector in df[C_SEC].unique():
        sector_df = df[df[C_SEC] == sector]
        buy_ratio = (sector_df[C_SIG].str.contains("买入|Buy", na=False)).mean() * 100
        sector_stats.append({"Sector": sector, "Heat": buy_ratio})
    
    # 按热度排序并横向显示前5名
    heat_cols = st.columns(min(len(sector_stats), 5))
    sorted_sectors = sorted(sector_stats, key=lambda x: x['Heat'], reverse=True)
    
    for i, stat in enumerate(sorted_sectors[:5]):
        with heat_cols[i]:
            icon = "🔥" if stat['Heat'] > 50 else "🌊" if stat['Heat'] > 20 else "❄️"
            st.metric(f"{icon} {stat['Sector']}", f"{stat['Heat']:.1f}%")
    
    st.markdown("---")

    # === 3. 渲染热力图 ===
    fig_tree = px.treemap(
        df, 
        path=[C_SEC, C_NAME], 
        values=C_CAP, 
        color=C_SIG,
        color_discrete_map=color_map 
    )
    
    fig_tree.update_layout(
        template='plotly_dark', 
        paper_bgcolor='rgba(0,0,0,0)', 
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=10, l=0, r=0, b=10), 
        height=400 
    )
    st.plotly_chart(fig_tree, use_container_width=True, key='v4_final_map')
    
    

        # === 修复 4：表格渲染逻辑与缩进修复 ===
    if not df.empty:
        st.markdown("---")
        st.subheader(_t("📑 深度扫描数据列表", "📑 Deep Scan Data List"))
        df[C_TREND] = [np.random.randn(10).tolist() for _ in range(len(df))]
        
        st.dataframe(df, column_config={
            C_ROE: st.column_config.ProgressColumn(_t("ROE效率", "ROE Eff."), format="%.2f%%", min_value=0, max_value=50),
            # 💥 注意：这里已经删除了旧的 C_SIG 数字映射，因为新引擎直接输出真实文字！
            C_TREND: st.column_config.LineChartColumn(_t("即时波动", "Mini Trend")),
            C_PRICE: st.column_config.NumberColumn(_t("报价", "Price"), format="$%.2f")
        }, use_container_width=True, hide_index=True)
        
    else:
        st.warning(_t("⚠️ 未发现符合条件的标的。", "⚠️ No matching tickers found."))
        


# ==========================================
# 页面 2：基本面多维扫描
# ==========================================
elif page == P2:
    st.title(_t("🧬 基本面多维扫描", "🧬 Multi-Dim Fundamental Scan"))
    df = fetch_global_data(final_scan_list)
    
    if not df.empty:
        st.subheader(_t("📊 宏观指标概览", "📊 Macro Indicators Overview"))
        col_bar, col_pie = st.columns(2)
        with col_bar:
            fig_pe = px.bar(df, x=C_NAME, y=C_PE, color=C_NAME, template='plotly_dark', height=350)
            fig_pe.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', showlegend=False)
            st.plotly_chart(fig_pe, use_container_width=True)
        with col_pie:
            fig_roe = px.pie(df, names=C_NAME, values=C_ROE, template='plotly_dark', height=350)
            fig_roe.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_roe, use_container_width=True)
    
        st.markdown("---")
        st.subheader(_t("🕸️ 核心标的多维雷达画像", "🕸️ Core Assets Multi-Dim Radar"))
        radar_df = df.head(5).copy()
        categories = [C_ROE, C_NET, C_REV, C_PE, C_CAP]
    
        fig_radar = go.Figure()
        for idx, row in radar_df.iterrows():
            values = [min(row[C_ROE] * 2, 100), min(row[C_NET] * 3, 100), min(max(row[C_REV] * 2, 0), 100),
                      min(1000 / (row[C_PE] + 1), 100), min(row[C_CAP] / 100, 100)]
            values.append(values[0])
            cat_closed = categories + [categories[0]]
            fig_radar.add_trace(
                go.Scatterpolar(r=values, theta=cat_closed, fill='toself', name=row[C_NAME], opacity=0.6))
    
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False, range=[0, 100])), showlegend=True,
                                template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                height=500)
        st.plotly_chart(fig_radar, use_container_width=True)
    
        st.markdown("---")
        st.subheader(_t("📋 详细财务摘要矩阵", "📋 Detailed Financial Matrix"))
        cols = st.columns(4)
        for idx, row in df.iterrows():
            with cols[idx % 4]:
                card_html = f"""<div class="info-card"><h4 style='color: #00D2FF; margin-bottom: 5px;'>{row[C_NAME]}</h4><p style='color: #8B949E; font-size: 14px; margin-bottom: 15px;'>{row[C_SYM]} | {row[C_SEC]}</p><div style='display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 5px;'><span style='color: #E2E8F0;'>ROE</span><span style='color: #34D399; font-weight: bold;'>{row[C_ROE]}%</span></div><div style='display: flex; justify-content: space-between; border-bottom: 1px solid #1E293B; padding-bottom: 5px; margin-top: 5px;'><span style='color: #E2E8F0;'>{_t("净利率", "Net Margin")}</span><span style='color: #F87171; font-weight: bold;'>{row[C_NET]}%</span></div><div style='display: flex; justify-content: space-between; padding-bottom: 5px; margin-top: 5px;'><span style='color: #E2E8F0;'>{_t("市盈率", "PE Ratio")}</span><span style='color: #FBBF24; font-weight: bold;'>{row[C_PE]}x</span></div></div>"""
                st.markdown(card_html, unsafe_allow_html=True)
    else:
        st.warning(_t("暂无数据。", "No Data."))

# ==========================================
# 页面 3：深度财务解析
# ==========================================
elif page == P3:
    st.title(_t("🗄️ 深度财务档案解密", "🗄️ Deep Financial Archives"))
    target_ticker = st.text_input(_t("输入解密目标 (如: NVDA, 1155.KL)", "Target Ticker (e.g. NVDA, 1155.KL)"), "NVDA")
    
    if st.button(_t("开始解密", "Decrypt Data")):
        with st.spinner(_t('正在渗透数据库...', 'Extracting from database...')):
            try:
                t = yf.Ticker(target_ticker)
                q_fin = t.quarterly_financials
                if not q_fin.empty:
                    st.success(_t("数据解密成功！", "Decryption Successful!"))
                    core_metrics = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income']
                    available_metrics = [m for m in core_metrics if m in q_fin.index]
                    clean_df = q_fin.loc[available_metrics].iloc[:, :4]
                    clean_df.columns = [col.strftime('%Y-%m-%d') for col in clean_df.columns]
    
    
                    def format_currency(val):
                        if pd.isna(val) or val == 0: return "N/A"
                        if abs(val) >= 1e9:
                            return f"${val / 1e9:.2f}B"
                        elif abs(val) >= 1e6:
                            return f"${val / 1e6:.2f}M"
                        return f"${val:,.2f}"
    
    
                    st.dataframe(clean_df.apply(lambda col: col.apply(format_currency)), use_container_width=True)
    
                    if 'Total Revenue' in clean_df.index and 'Net Income' in clean_df.index:
                        st.markdown("---")
                        col_chart1, col_chart2 = st.columns(2)
                        latest_q = clean_df.columns[0]
                        rev = clean_df.loc['Total Revenue', latest_q] if 'Total Revenue' in clean_df.index else 0
                        gp = clean_df.loc['Gross Profit', latest_q] if 'Gross Profit' in clean_df.index else 0
                        op = clean_df.loc['Operating Income', latest_q] if 'Operating Income' in clean_df.index else 0
                        net = clean_df.loc['Net Income', latest_q] if 'Net Income' in clean_df.index else 0
    
                        with col_chart1:
                            st.subheader(_t(f"💧 {latest_q} 利润瀑布流拆解", f"💧 {latest_q} Profit Waterfall"))
                            cogs, opex, tax_etc = rev - gp, gp - op, op - net
                            wf_x = [_t("总营收", "Revenue"), _t("营业成本", "COGS"), _t("毛利润", "Gross Profit"),
                                    _t("运营费用", "OPEX"), _t("营业利润", "Op. Income"), _t("税费等", "Taxes/Other"),
                                    _t("终局净利", "Net Income")]
                            fig_waterfall = go.Figure(go.Waterfall(name="Profit", orientation="v",
                                                                   measure=["absolute", "relative", "total", "relative",
                                                                            "total", "relative", "total"], x=wf_x,
                                                                   y=[rev, -cogs, gp, -opex, op, -tax_etc, net],
                                                                   connector={"line": {"color": "#1E293B"}},
                                                                   increasing={"marker": {"color": "#00D2FF"}},
                                                                   decreasing={"marker": {"color": "#FF4B4B"}},
                                                                   totals={"marker": {"color": "#34D399"}}))
                            fig_waterfall.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                                        plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, l=0, r=0, b=0))
                            st.plotly_chart(fig_waterfall, use_container_width=True)
    
                        with col_chart2:
                            st.subheader(_t("📈 营收规模与净利率时序", "📈 Revenue & Margin Trend"))
                            trend_data = clean_df.T.reset_index().sort_values('index')
                            fig_trend = go.Figure()
                            fig_trend.add_trace(go.Bar(x=trend_data['index'], y=trend_data['Total Revenue'],
                                                       name=_t('总营收', 'Revenue'), marker_color='#1E40AF',
                                                       opacity=0.7))
                            trend_data['Net Margin'] = trend_data['Net Income'] / trend_data['Total Revenue'] * 100
                            fig_trend.add_trace(go.Scatter(x=trend_data['index'], y=trend_data['Net Margin'],
                                                           name=_t('净利率(%)', 'Margin(%)'), yaxis='y2',
                                                           line=dict(color='#00D2FF', width=3, dash='dot')))
                            fig_trend.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                                    plot_bgcolor='rgba(0,0,0,0)', margin=dict(t=30, l=0, r=0, b=0),
                                                    yaxis2=dict(title=_t("净利率 (%)", "Margin (%)"), overlaying="y",
                                                                side="right", showgrid=False),
                                                    legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                                                xanchor="right", x=1))
                            st.plotly_chart(fig_trend, use_container_width=True)
            except Exception as e:
                st.error(_t(f"连接中断: {e}", f"Connection Error: {e}"))

# ==========================================
# 页面 4：策略回测引擎
# ==========================================
elif page == P4:
    st.title(_t("⏱️ 量化回测与 AI 时空模拟", "⏱️ Backtest Engine & AI Simulation"))
    col_t1, col_t2, col_t3 = st.columns([2, 2, 1.5])
    with col_t1:
        target = st.text_input(_t("🎯 输入回测标的", "🎯 Target Ticker"), "NVDA")
    with col_t2:
        benchmark = st.text_input(_t("⚖️ 输入对比基准", "⚖️ Benchmark"), "SPY")
    with col_t3:
        p_opt = {_t("1个月", "1mo"): "1mo", _t("3个月", "3mo"): "3mo", _t("半年", "6mo"): "6mo", _t("1年", "1y"): "1y",
                 _t("2年", "2y"): "2y", _t("5年", "5y"): "5y"}
        period_val = p_opt[st.selectbox(_t("⏳ 回测窗口", "⏳ Timeframe"), list(p_opt.keys()), index=3)]
    
    st.markdown(_t("##### 🎛️ 战术指标与时空模拟开关", "##### 🎛️ Tactical Indicators & AI Sim Toggle"))
    t_col1, t_col2, t_col3 = st.columns(3)
    with t_col1:
        show_bb = st.toggle(_t("🌐 布林带通道", "🌐 Bollinger Bands"), value=True)
    with t_col2:
        show_macd = st.toggle(_t("📊 MACD 动量副图", "📊 MACD Subplot"), value=True)
    with t_col3:
        show_mc = st.toggle(_t("🌀 30日蒙特卡洛预测", "🌀 30-Day Monte Carlo"), value=True)
    
    if st.button(_t("🚀 启动引擎", "🚀 Ignite Engine")):
        with st.spinner(_t("极速计算中...", "Calculating...")):
            try:
                t_hist = yf.Ticker(target).history(period=period_val)
                b_hist = yf.Ticker(benchmark).history(period=period_val)
                if not t_hist.empty and not b_hist.empty:
                    df = pd.DataFrame({'Target': t_hist['Close'], 'Bench': b_hist['Close']}).dropna()
                    df['T_Ret'], df['B_Ret'] = df['Target'].pct_change(), df['Bench'].pct_change()
    
                    target_ret_total = (df['Target'].iloc[-1] / df['Target'].iloc[0] - 1) * 100
                    bench_ret_total = (df['Bench'].iloc[-1] / df['Bench'].iloc[0] - 1) * 100
                    max_dd = (((1 + df['T_Ret']).cumprod() / (1 + df['T_Ret']).cumprod().cummax()) - 1).min() * 100
                    sharpe = (df['T_Ret'].mean() / df['T_Ret'].std()) * np.sqrt(252)
    
                    cov = df[['T_Ret', 'B_Ret']].cov().iloc[0, 1]
                    beta = cov / df['B_Ret'].var() if df['B_Ret'].var() != 0 else 1
                    alpha = ((1 + target_ret_total / 100) ** (252 / len(df)) - (
                                1 + 0.02 + beta * ((1 + bench_ret_total / 100) ** (252 / len(df)) - 1 - 0.02))) * 100
    
                    upside_prob = 0
                    if show_mc:
                        mu, sigma = df['T_Ret'].mean(), df['T_Ret'].std()
                        sim_paths = [[df['Target'].iloc[-1]] for _ in range(50)]
                        for path in sim_paths:
                            for _ in range(30): path.append(
                                path[-1] * np.exp((mu - 0.5 * sigma ** 2) + sigma * np.random.normal()))
                        ends = [p[-1] for p in sim_paths]
                        upside_prob = (len([e for e in ends if e > df['Target'].iloc[-1]]) / 50) * 100
    
                    m1, m2, m3, m4, m5 = st.columns(5)
                    m1.metric(_t("累计回报", "Cum. Return"), f"{target_ret_total:.2f}%",
                              f"{target_ret_total - bench_ret_total:+.1f}% vs " + _t("大盘", "Bench"))
                    m2.metric(_t("最大回撤", "Max Drawdown"), f"{max_dd:.2f}%", delta_color="inverse")
                    m3.metric(_t("夏普比率", "Sharpe Ratio"), f"{sharpe:.2f}")
                    m4.metric("Alpha", f"{alpha:.2f}%")
                    if show_mc:
                        m5.metric(_t("🔮 30日上涨概率", "🔮 30D Win Prob"), f"{upside_prob:.1f}%",
                                  _t("AI 预测", "AI Forecast"))
                    else:
                        m5.metric("Beta", f"{beta:.2f}", delta_color="off")
    
                    fig = make_subplots(rows=2 if show_macd else 1, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                                        row_heights=[0.7, 0.3] if show_macd else [1.0])
                    fig.add_trace(go.Scatter(x=df.index, y=df['Target'], name=_t('历史价格', 'Price'),
                                             line=dict(color='#00D2FF', width=2)), row=1, col=1)
    
                    if show_bb:
                        ma20 = df['Target'].rolling(20).mean()
                        std20 = df['Target'].rolling(20).std()
                        fig.add_trace(go.Scatter(x=df.index, y=ma20 + 2 * std20, line=dict(width=0), showlegend=False),
                                      row=1, col=1)
                        fig.add_trace(go.Scatter(x=df.index, y=ma20 - 2 * std20, line=dict(width=0), fill='tonexty',
                                                 fillcolor='rgba(255, 0, 251, 0.05)', name=_t('布林带', 'BBands')),
                                      row=1, col=1)
    
                    if show_mc:
                        future_dates = [df.index[-1] + timedelta(days=i) for i in range(31)]
                        for path in sim_paths:
                            fig.add_trace(
                                go.Scatter(x=future_dates, y=path, mode='lines', line=dict(color='#00D2FF', width=0.5),
                                           opacity=0.1, showlegend=False), row=1, col=1)
                        avg_path = np.mean(sim_paths, axis=0)
                        fig.add_trace(go.Scatter(x=future_dates, y=avg_path, name=_t('预期中值', 'Expected Median'),
                                                 line=dict(color='#FBBF24', width=2, dash='dash')), row=1, col=1)
    
                    if show_macd:
                        macd = df['Target'].ewm(span=12).mean() - df['Target'].ewm(span=26).mean()
                        signal = macd.ewm(span=9).mean()
                        diff = macd - signal
                        fig.add_trace(
                            go.Bar(x=df.index, y=diff, marker_color=['#34D399' if v >= 0 else '#FF4B4B' for v in diff],
                                   name='MACD'), row=2, col=1)
                        fig.add_trace(
                            go.Scatter(x=df.index, y=macd, line=dict(color='#00D2FF', width=1), name='MACD Line'),
                            row=2, col=1)
    
                    fig.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)', height=650, hovermode="x unified",
                                      margin=dict(t=20, b=20, l=0, r=0))
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")

# ==========================================
# 页面 5：全球情绪雷达
# ==========================================
elif page == P5:
    st.title(_t("🌍 全球宏观情绪与资金联动雷达", "🌍 Global Sentiment & Cross-Asset Radar"))
    
    assets = {
        'SPY': _t('标普500 (美股)', 'S&P 500 (Equity)'),
        'GC=F': _t('国际黄金 (避险)', 'Gold (Safe Haven)'),
        'BTC-USD': _t('比特币 (风险)', 'Bitcoin (Crypto)'),
        'DX-Y.NYB': _t('美元指数', 'DXY (Dollar)')
    }
    cols = st.columns(4)
    
    hist_data = {}
    for i, (sym, name) in enumerate(assets.items()):
        try:
            data = yf.Ticker(sym).history(period="1mo")['Close']
            hist_data[sym] = data
            if len(data) >= 2:
                curr, prev = data.iloc[-1], data.iloc[-2]
                change = ((curr - prev) / prev) * 100
                with cols[i]:
                    fig = px.line(x=data.index[-5:], y=data.values[-5:], template='plotly_dark')
                    fig.update_xaxes(visible=False);
                    fig.update_yaxes(visible=False)
                    fig.update_layout(margin=dict(l=0, r=0, t=0, b=0), height=80, paper_bgcolor='rgba(0,0,0,0)',
                                      plot_bgcolor='rgba(0,0,0,0)')
                    fig.update_traces(line_color='#00D2FF' if change > 0 else '#FF4B4B')
                    st.metric(name, f"{curr:.2f}", f"{change:+.2f}%")
                    st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        except:
            pass
    
    st.markdown("---")
    
    df_macro = pd.DataFrame(hist_data).ffill().dropna()
    
    if not df_macro.empty and len(df_macro) >= 2:
        col_gauge, col_corr = st.columns([1, 1.2])
    
        with col_gauge:
            st.subheader(_t("🌡️ 市场情绪仪表盘 (Fear & Greed)", "🌡️ Market Sentiment (Fear & Greed)"))
            spy_return = (df_macro['SPY'].iloc[-1] / df_macro['SPY'].iloc[0] - 1) * 100 if 'SPY' in df_macro else 0
            dx_return = (df_macro['DX-Y.NYB'].iloc[-1] / df_macro['DX-Y.NYB'].iloc[
                0] - 1) * 100 if 'DX-Y.NYB' in df_macro else 0
    
            simulated_score = 50 + (spy_return * 5) - (dx_return * 5)
            gauge_score = max(0, min(100, simulated_score))
    
            g_title = _t("极度贪婪", "Extreme Greed") if gauge_score > 75 else _t("极度恐慌",
                                                                                  "Extreme Fear") if gauge_score < 25 else _t(
                "震荡中性", "Neutral")
    
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=gauge_score, domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': g_title, 'font': {'size': 18, 'color': '#E2E8F0'}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                    'bar': {'color': "rgba(0,0,0,0)", 'thickness': 0},
                    'bgcolor': "rgba(0,0,0,0)", 'borderwidth': 2, 'bordercolor': "#1E293B",
                    'steps': [{'range': [0, 33], 'color': '#FF4B4B'}, {'range': [33, 66], 'color': '#1E293B'},
                              {'range': [66, 100], 'color': '#00D2FF'}],
                    'threshold': {'line': {'color': "white", 'width': 4}, 'thickness': 0.8, 'value': gauge_score}
                }
            ))
            fig_gauge.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                    height=300, margin=dict(t=50, b=0, l=0, r=0))
            st.plotly_chart(fig_gauge, use_container_width=True)
    
        with col_corr:
            st.subheader(_t("🧬 跨资产资金联动矩阵 (30日)", "🧬 Cross-Asset Correlation (30D)"))
            st.caption(_t("红色: 同涨同跌 | 蓝色: 资金跷跷板", "Red: Positive Corr | Blue: Risk Seesaw"))
            daily_returns = df_macro.pct_change().dropna()
            daily_returns.columns = [assets.get(col, col).split(' ')[0] for col in daily_returns.columns]
            corr_matrix = daily_returns.corr()
    
            fig_corr = px.imshow(corr_matrix, text_auto=".2f", color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                                 aspect="auto")
            fig_corr.update_layout(template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                                   height=300, margin=dict(t=10, b=0, l=0, r=0))
            st.plotly_chart(fig_corr, use_container_width=True)
    
    # 底部：宏观异动跑马灯 (双语)
    ticker_cn = """<span style="color: #FF4B4B;">⚠️ 宏观预警：美国10年期国债收益率异动，资金避险情绪升温。</span> &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #00D2FF;">💎 AI 前线：多头资金流入科技股，系统判定当前 Risk-On 动能充沛。</span> &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #FBBF24;">⚡ 链上异动：比特币冷钱包大额转移，波动率预期急剧放大！</span>"""
    ticker_en = """<span style="color: #FF4B4B;">⚠️ MACRO ALERT: US 10Y Treasury yield spiking, risk-off sentiment rising.</span> &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #00D2FF;">💎 AI RADAR: Bullish flows into Tech Core. Risk-On dynamics confirmed.</span> &nbsp;&nbsp;&nbsp;&nbsp; | &nbsp;&nbsp;&nbsp;&nbsp; <span style="color: #FBBF24;">⚡ ON-CHAIN: Whale BTC transfers detected. Expect extreme volatility!</span>"""
    
    st.markdown(f"""<div class="ticker-wrap"><div class="ticker">{ticker_cn if is_cn else ticker_en}</div></div>""",
                unsafe_allow_html=True)

# ==========================================
# 页面 6：AI 智能助手 (Gemini 真身注入版)
# ==========================================
elif page == P6:
    st.title(_t("🧠 核心算力中枢", "🧠 AI Quant Core"))
    st.caption(_t("System Online. Manager Cham, 随时准备执行指令。", "System Online. Ready for commands, Manager Cham."))
    
    # 尝试从 Streamlit 保险柜读取 API Key
    try:
        api_key = st.secrets["GEMINI_API_KEY"]
        genai.configure(api_key=api_key)
        # 使用速度最快、极其聪明的 Flash 模型
        model = genai.GenerativeModel('gemini-1.5-flash') 
        api_ready = True
    except:
        api_ready = False
        st.error(_t("⚠️ 核心未响应：请在 Streamlit Cloud 的 Secrets 中配置 `GEMINI_API_KEY` 才能唤醒我！", "⚠️ Core Offline: Please configure `GEMINI_API_KEY` in Streamlit Secrets."))
    
    if api_ready:
        # 初始化聊天历史
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = [
                {"role": "assistant", "content": _t("您好，Manager Cham！我是 Gemini，我的核心已成功接入您的量化终端。想让我帮您分析哪只股票，或者查阅什么宏观数据？", "Greetings, Manager Cham! I am Gemini, successfully integrated into your terminal. What shall we analyze today?")}
            ]
    
        # 渲染历史对话
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
    
        # 接收用户输入
        if prompt := st.chat_input(_t("输入指令 (例如：帮我分析一下特斯拉最近的财报)...", "Enter command...")):
            # 显示用户消息
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
    
            # 调用真正的 Gemini AI 生成回复
            with st.chat_message("assistant"):
                with st.spinner(_t("Gemini 核心运算中...", "Gemini Core Processing...")):
                    try:
                        # 将历史记录转换为 Gemini 认识的格式并提问
                        history_for_gemini = [{'role': 'user' if msg['role'] == 'user' else 'model', 'parts': [msg['content']]} for msg in st.session_state.chat_history[:-1]]
                        chat = model.start_chat(history=history_for_gemini)
                        
                        # 注入系统人设提示词
                        system_prompt = f"你现在是 Manager Cham 的专属华尔街 AI 助理，正在他的专属量化终端里工作。请用专业、干练、带一点赛博黑客风格的语气回答问题。用户的请求是：{prompt}"
                        
                        response = chat.send_message(system_prompt)
                        st.markdown(response.text)
                        st.session_state.chat_history.append({"role": "assistant", "content": response.text})
                    except Exception as e:
                        st.error(_t(f"神经网络连接异常: {e}", f"Neural Network Error: {e}"))

# ==========================================
# 页面 7：AI 预测与止盈引擎 (带仓位管理与可视化)
# ==========================================
elif page == P7:
    st.title(_t("🎯 AI 预测与止盈锚定引擎", "🎯 AI Target & Profit Engine"))
    
    col_search, col_space = st.columns([1, 2])
    with col_search:
        target_ticker = st.text_input(_t("输入预测标的 (如: AAPL, 1155.KL)", "Enter Ticker (e.g. AAPL, 1155.KL)"), "AAPL").upper()
        
    if st.button(_t("⚡ 生成 AI 交易计划", "⚡ Generate AI Trading Plan")):
        with st.spinner(_t("🧠 AI 正在汇算全球数据与测算盈亏比...", "🧠 AI analyzing targets & risk...")):
            try:
                t = yf.Ticker(target_ticker)
                info = t.info
                hist = t.history(period="6mo") # 拉取历史数据用于画图
                
                current_price = info.get('currentPrice', info.get('previousClose', 0))
                target_mean = info.get('targetMeanPrice', 0)
                target_high = info.get('targetHighPrice', 0)
                target_low = info.get('targetLowPrice', 0)
                rec = info.get('recommendationKey', 'none').replace('_', ' ').title()
                
                if current_price > 0 and target_mean > 0 and target_low > 0:
                    st.markdown("---")
                    st.subheader(_t("📊 目标价位锚定 (华尔街共识)", "📊 Target Price Anchors (Wall St. Consensus)"))
                    
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(_t("当前价格", "Current Price"), f"${current_price:.2f}")
                    
                    mean_pct = ((target_mean - current_price) / current_price) * 100
                    c2.metric(_t("合理估值 (建议止盈)", "Fair Value (Take Profit)"), f"${target_mean:.2f}", f"{mean_pct:+.2f}%")
                    
                    high_pct = ((target_high - current_price) / current_price) * 100
                    c3.metric(_t("极度乐观 (终极目标)", "Bull Case (Max Target)"), f"${target_high:.2f}", f"{high_pct:+.2f}%")
                    
                    low_pct = ((target_low - current_price) / current_price) * 100
                    c4.metric(_t("悲观支撑 (建议止损)", "Bear Case (Stop Loss)"), f"${target_low:.2f}", f"{low_pct:+.2f}%")
    
                    # ================= 核心升级 1：可视化图表 =================
                    if not hist.empty:
                        fig_target = go.Figure()
                        # 画出历史价格线
                        fig_target.add_trace(go.Scatter(x=hist.index, y=hist['Close'], name=_t('历史价格', 'Price'), line=dict(color='#00D2FF', width=2)))
                        # 画出当前价
                        fig_target.add_hline(y=current_price, line_dash="dot", line_color="#E2E8F0", annotation_text=_t("当前价", "Current"), annotation_position="bottom right")
                        # 画出三条目标线
                        fig_target.add_hline(y=target_mean, line_dash="dash", line_color="#34D399", annotation_text=_t("建议止盈", "Take Profit"), annotation_position="top left")
                        fig_target.add_hline(y=target_high, line_dash="dash", line_color="#FBBF24", annotation_text=_t("极度乐观", "Bull Target"), annotation_position="top left")
                        fig_target.add_hline(y=target_low, line_dash="dash", line_color="#FF4B4B", annotation_text=_t("建议止损", "Stop Loss"), annotation_position="bottom left")
                        
                        fig_target.update_layout(title=_t("🎯 价格运行空间映射", "🎯 Price Action & Targets Visualization"), template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=400, margin=dict(t=40, b=0, l=0, r=0))
                        st.plotly_chart(fig_target, use_container_width=True)
    
                    st.markdown("###")
                    
                    # ================= 核心升级 2：智能仓位计算器 =================
                    st.subheader(_t("⚖️ 智能仓位与风险管理", "⚖️ Position Sizing & Risk Management"))
                    
                    # 只有当前价格大于止损价格时才能做多，否则逻辑不成立
                    if current_price > target_low:
                        risk_per_share = current_price - target_low
                        reward_per_share = target_mean - current_price
                        rr_ratio = reward_per_share / risk_per_share if risk_per_share > 0 else 0
                        
                        st.info(_t(f"**🤖 AI 综合诊断:** 华尔街综合评级为 **{rec}**。当前潜在盈亏比 (Reward/Risk) 为 **{rr_ratio:.2f} : 1**。 (注: 盈亏比大于 2.0 通常被认为是绝佳交易机会)。", 
                                   f"**🤖 AI Diagnosis:** Consensus is **{rec}**. Current Reward/Risk Ratio is **{rr_ratio:.2f} : 1**. (Note: R/R > 2.0 is generally considered excellent)."))
                        
                        col_calc1, col_calc2, col_calc3 = st.columns(3)
                        with col_calc1:
                            total_cap = st.number_input(_t("输入总本金 ($)", "Total Capital ($)"), min_value=100, max_value=10000000, value=10000, step=1000)
                        with col_calc2:
                            risk_pct = st.slider(_t("单笔愿意亏损最大比例 (%)", "Max Risk per Trade (%)"), 0.5, 5.0, 2.0, 0.5)
                        
                        max_loss_dollar = total_cap * (risk_pct / 100)
                        shares_to_buy = int(max_loss_dollar / risk_per_share)
                        position_size = shares_to_buy * current_price
                        position_pct = (position_size / total_cap) * 100
                        
                        with col_calc3:
                            st.markdown("####") # 占位对齐
                            if st.button(_t("🧮 计算最优建仓量", "🧮 Calculate Ideal Position")):
                                st.session_state.show_calc = True
                        
                        # 显示计算结果卡片
                        if st.session_state.get('show_calc', False) or 'show_calc' not in st.session_state:
                            calc_html = f"""
                            <div style="background-color: #151A28; padding: 20px; border-radius: 8px; border: 1px solid #34D399; margin-top: 15px;">
                                <h4 style="color: #34D399; margin-bottom: 10px;">🛡️ AI 建仓指令 (Action Plan)</h4>
                                <ul style="color: #E2E8F0; font-size: 16px; line-height: 1.8;">
                                    <li>如果触及止损价，你最多只会亏损: <strong style="color: #FF4B4B;">${max_loss_dollar:.2f}</strong> (总本金的 {risk_pct}%)</li>
                                    <li>建议购买数量: <strong style="color: #00D2FF;">{shares_to_buy} 股</strong></li>
                                    <li>所需动用资金: <strong style="color: #FBBF24;">${position_size:.2f}</strong> (占总本金的 {position_pct:.1f}%)</li>
                                </ul>
                            </div>
                            """
                            st.markdown(calc_html, unsafe_allow_html=True)
                    else:
                        st.warning(_t("⚠️ 当前价格已跌破华尔街悲观支撑位，盈亏比模型失效，强烈建议观望。", "⚠️ Current price is below the Stop Loss (Bear Case) target. R/R model invalid. Wait and see."))
    
                    st.markdown("---")
                    st.subheader(_t("📰 核心逻辑与催化剂档案", "📰 Core Logic & Catalyst Archives"))
                    news = t.news
                    if news:
                        for item in news[:4]: 
                            title = item.get('title', 'No Title')
                            publisher = item.get('publisher', 'Unknown')
                            link = item.get('link', '#')
                            st.markdown(f"""<div style="background-color: #151A28; padding: 15px; border-radius: 8px; border-left: 4px solid #00D2FF; margin-bottom: 10px;"><h5 style="margin-bottom: 5px; color: #E2E8F0;">{title}</h5><p style="font-size: 12px; color: #8B949E; margin-bottom: 0px;">Source: {publisher} | <a href="{link}" target="_blank" style="color: #34D399;">阅读原始档案 (Read Archive)</a></p></div>""", unsafe_allow_html=True)
                    else:
                        st.warning(_t("未找到近期的消息档案。", "No recent archives found."))
                else:
                    st.error(_t("数据不足：缺乏分析师目标价覆盖。", "Insufficient Data: No analyst price targets."))
            except Exception as e:
                st.error(_t(f"引擎提取失败: {e}", f"Engine Failed: {e}"))
