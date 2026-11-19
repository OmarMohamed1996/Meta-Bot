
import os
import json
import requests
import pandas as pd
import streamlit as st
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor
from google import genai
from google.genai import types

# ================= 1. الإعدادات والمفاتيح =================
METABASE_URL = os.getenv("METABASE_URL", "https://metabase.2segypt.com")
SESSION_TOKEN = os.getenv("METABASE_SESSION", "")
GEMINI_API_KEY = ""

client = genai.Client(api_key=GEMINI_API_KEY, http_options={"api_version": "v1beta"})
DASHBOARD_ID = 1467

MIVIDA_CARDS_CATALOG = """
1. Branch Sales:
   - 7458: Total Revenue (إجمالي المبيعات)
   - 7440: Daily Sales Target (تارجت المبيعات اليومي)
   - 7457: Revenue vs Target (المبيعات مقارنة بالتارجت)
   - 7445: Target per branch (تارجت الفروع)
   - 7461: Rank in Today's Sales (ترتيب الفرع في مبيعات اليوم)
   - 7432: Number POS Orders (عدد فواتير الكاشير)
   - 7438: Quantity (عدد القطع المباعة)
   - 7436: Sales per Season (المبيعات حسب السيزون)
   - 7448: POS Timeline (المبيعات بالساعة/الوقت)
   - 7442: Like for Like (مقارنة المبيعات بنفس الفترة)
2. UPT & ATV:
   - 7450: UPT & ATV Report (تقرير متوسط القطع ومتوسط الفاتورة)
   - 7452: UPT & ATV Report per Employee (أداء الموظفين في UPT و ATV)
   - 7460: UPT Timeline | 7462: ATV Timeline
3. Replenishment & Stock:
   - 8510: Replenishment – Mivida (تقرير تعويض النواقص - ريبلينشمنت ميفيدا)
   - 8566: Stock Movement Analysis – Mivida (تحليل حركة المخزون)
   - 7434: Slow Moving Stock – Mivida (الأصناف بطيئة الحركة / الراكدة)
   - 8207: Category Stock Flow | 8623: Season Stock Flow | 8624: Category Stock Flow - Model Detail
4. Product Analysis:
   - 7455: Top-Selling Models (أكثر الموديلات مبيعاً)
   - 7456: Top Colors (أكثر الألوان مبيعاً)
   - 7449: Top Sizes (أكثر المقاسات مبيعاً)
   - 7453: Top Categories | 7443: Quantity per Category | 7435: Amount per Category
   - 7437: Product Weekly Sales | 7439: Category Composition by Department | 7454: Factory Performance
5. Productivity & Operations:
   - 7459: Branch Productivity per Employee - Mivida (إنتاجية الموظف)
   - 7433: Branch Productivity per Sq. Meter - Mivida (إنتاجية المتر المربع)
   - 7444: Store Pickup – Fulfillment Rate | 7451: Store Pickup – Requested Items
6. Customer Satisfaction:
   - 7441: Customer Survey Score per branch | 7446: Number of Bad Reviews | 7447: Surveys to orders ratio
"""

# ================= 2. محرك الاتصال + Caching + Parallel Execution =================
@st.cache_resource
def get_mb_session():
    s = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=15, pool_maxsize=15)
    s.mount("https://", adapter)
    s.headers.update({
        "X-Metabase-Session": SESSION_TOKEN,
        "Content-Type": "application/json",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    })
    return s

@st.cache_data(ttl=3600, show_spinner=False)
def get_dashboard_mappings():
    session = get_mb_session()
    res = session.get(f"{METABASE_URL.rstrip('/')}/api/dashboard/{DASHBOARD_ID}", timeout=20)
    card_map = {}
    if res.status_code == 200:
        data = res.json()
        param_meta = {p["id"]: p for p in data.get("parameters", [])}
        for dc in data.get("dashcards", []):
            card = dc.get("card")
            if card and card.get("id"):
                cid = card["id"]
                mappings = {}
                for m in dc.get("parameter_mappings", []):
                    pid = m.get("parameter_id")
                    slug = param_meta.get(pid, {}).get("slug")
                    if slug:
                        mappings[slug] = {
                            "id": pid,
                            "type": param_meta.get(pid, {}).get("type", "string/="),
                            "target": m.get("target")
                        }
                card_map[cid] = {"dashcard_id": dc["id"], "mappings": mappings}
    return card_map

@st.cache_data(ttl=300, show_spinner=False)
def query_metabase_card(
    card_id: int,
    date_filter: str = "thisday",
    season: str = "",
    category: str = "",
    model_no: str = "",
    factory: str = ""
) -> pd.DataFrame:
    session = get_mb_session()
    card_map = get_dashboard_mappings()
    info = card_map.get(card_id)

    filter_values = {
        "date": date_filter,
        "season": season,
        "category": category,
        "model_no.": model_no,
        "factory": factory,
    }

    parameters = []
    if info:
        for slug, val in filter_values.items():
            if val and slug in info["mappings"]:
                m = info["mappings"][slug]
                param_type = "date/all-options" if slug == "date" else m["type"]
                param_val = val if slug == "date" else [val]
                parameters.append({
                    "id": m["id"],
                    "type": param_type,
                    "value": param_val,
                    "target": m["target"]
                })
        url = f"{METABASE_URL.rstrip('/')}/api/dashboard/{DASHBOARD_ID}/dashcard/{info['dashcard_id']}/card/{card_id}/query/json"
        res = session.post(url, json={"parameters": parameters}, timeout=20)
    else:
        url = f"{METABASE_URL.rstrip('/')}/api/card/{card_id}/query/json"
        res = session.post(url, timeout=20)

    if res.status_code == 200:
        data = res.json()
        return pd.DataFrame(data) if isinstance(data, list) else pd.DataFrame()
    return pd.DataFrame()

def fetch_cards_parallel(card_ids: list[int], date_filter: str = "thisday", season: str = "", category: str = "") -> dict[int, pd.DataFrame]:
    with ThreadPoolExecutor(max_workers=6) as executor:
        futures = {
            cid: executor.submit(query_metabase_card, cid, date_filter, season, category)
            for cid in card_ids
        }
        return {cid: fut.result() for cid, fut in futures.items()}

# ================= 3. أداة الإيجنت السريعة =================
def get_mivida_reports(
    card_ids: list[int],
    date_filter: str = "thisday",
    season: str = "",
    category: str = "",
    model_no: str = "",
    factory: str = ""
) -> str:
    """
    Fetches live filtered data for one or multiple Metabase Card IDs in parallel for Mivida branch.
    """
    results = {}
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            cid: executor.submit(query_metabase_card, cid, date_filter, season, category, model_no, factory)
            for cid in card_ids
        }
        for cid, fut in futures.items():
            df = fut.result()
            if df.empty:
                results[str(cid)] = "لا توجد بيانات"
            else:
                results[str(cid)] = df.head(60).to_dict(orient="records")
    return json.dumps(results, ensure_ascii=False)

# ================= 4. تنسيق الواجهة الاحترافي (Custom CSS) =================
st.set_page_config(page_title="2S Mivida AI & Dashboard", page_icon="📊", layout="wide")

st.markdown("""
<style>
    /* تحسين شكل كروت الـ KPI */
    div[data-testid="metric-container"] {
        background-color: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 14px 18px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    div[data-testid="metric-container"] label {
        font-size: 14px !important;
        font-weight: 600 !important;
        color: #475569 !important;
    }
    div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
        font-size: 26px !important;
        font-weight: 800 !important;
        color: #0f172a !important;
    }
</style>
""", unsafe_allow_html=True)

head_col1, head_col2 = st.columns([4, 1])
with head_col1:
    st.title("📊 2S Home Wear | Mivida Live AI & Dashboard")
with head_col2:
    st.write("")
    if st.button("🔄 تحديث الداتا الآن", use_container_width=True):
        st.cache_data.clear()
        st.rerun()

# --- شريط الفلاتر ---
f1, f2, f3, f4 = st.columns([1.3, 1.3, 1, 1])
with f1:
    period_mode = st.selectbox(
        "الفترة الزمنية:",
        options=["thisday", "past1days", "past7days", "thismonth", "past30days", "custom_day", "custom_range"],
        format_func=lambda x: {
            "thisday": "النهاردة (Today)",
            "past1days": "إمبارح (Yesterday)",
            "past7days": "آخر 7 أيام",
            "thismonth": "الشهر الحالي",
            "past30days": "آخر 30 يوم",
            "custom_day": "تاريخ يوم محدد 📅",
            "custom_range": "من تاريخ إلى تاريخ 📆"
        }[x]
    )
with f2:
    if period_mode == "custom_day":
        picked_date = st.date_input("اختر اليوم:")
        active_date_filter = picked_date.strftime("%Y-%m-%d")
    elif period_mode == "custom_range":
        picked_range = st.date_input("اختر الفترة:", value=[])
        if isinstance(picked_range, (list, tuple)) and len(picked_range) == 2:
            active_date_filter = f"{picked_range[0].strftime('%Y-%m-%d')}~{picked_range[1].strftime('%Y-%m-%d')}"
        else:
            active_date_filter = "thisday"
    else:
        active_date_filter = period_mode
        st.text_input("الفلتر النشط:", value=active_date_filter, disabled=True)

with f3:
    filter_category = st.text_input("القسم (Category):", placeholder="اختياري...")
with f4:
    filter_season = st.text_input("السيزون (Season):", placeholder="اختياري...")

# --- القسم الأول: الداشبورد ---
with st.expander("📈 لوحة مؤشرات فرع ميفيدا الحية (Live KPIs)", expanded=True):
    dash_data = fetch_cards_parallel([7458, 7432, 7438, 7461, 7455, 7452], active_date_filter, filter_season, filter_category)

    df_rev = dash_data[7458]
    df_orders = dash_data[7432]
    df_qty = dash_data[7438]
    df_rank = dash_data[7461]
    df_emp_raw = dash_data[7452]

    # استخراج UPT و ATV بدقة من صف Grand Total في جدول الموظفين
    upt_val, atv_val, asp_val = "—", "—", "—"
    if not df_emp_raw.empty and "UPT" in df_emp_raw.columns and "ATV" in df_emp_raw.columns:
        total_rows = df_emp_raw[df_emp_raw.astype(str).apply(lambda r: r.str.contains("Grand Total|Month Total", case=False).any(), axis=1)]
        target_row = total_rows.iloc[-1] if not total_rows.empty else df_emp_raw.iloc[-1]
        upt_val = f"{float(target_row['UPT']):.2f}"
        atv_val = f"{float(target_row['ATV']):,.0f} EGP"
        if "ASP" in df_emp_raw.columns:
            asp_val = f"{float(target_row['ASP']):,.0f}"

    # صف الـ 5 KPIs الرئيسية
    kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
    with kpi1:
        st.metric("💰 إجمالي المبيعات", f"{df_rev.iloc[0, 0]:,.0f} EGP" if not df_rev.empty else "—")
    with kpi2:
        st.metric("🧾 عدد الفواتير", f"{df_orders.iloc[0, 0]:,.0f}" if not df_orders.empty else "—")
    with kpi3:
        st.metric("🛍️ القطع المباعة", f"{df_qty.iloc[0, 0]:,.0f} قطعة" if not df_qty.empty else "—")
    with kpi4:
        st.metric("📊 UPT | ATV", f"{upt_val} | {atv_val}")
    with kpi5:
        st.metric("🏆 ترتيب ميفيدا", f"#{df_rank.iloc[0, 0]}" if not df_rank.empty else "—")

    st.write("")

    # صف الرسومات البيانية والجداول المنظفة
    chart1, chart2 = st.columns([1.05, 1.15])

    with chart1:
        df_top = dash_data[7455]
        if not df_top.empty:
            st.subheader("🔥 أكثر 10 موديلات مبيعاً")
            # تحديد عمود الاسم وعمود الكمية/المبيعات بذكاء بعيداً عن Model No
            numeric_cols = [c for c in df_top.select_dtypes(include="number").columns if "model" not in c.lower() and "id" not in c.lower()]
            y_col = next((c for c in numeric_cols if "qty" in c.lower() or "quant" in c.lower()), numeric_cols[0] if numeric_cols else df_top.columns[-1])
            name_col = next((c for c in df_top.columns if "product" in c.lower() or "name" in c.lower()), df_top.columns[0])

            df_plot = df_top.head(10).copy()
            df_plot = df_plot.sort_values(by=y_col, ascending=True)

            fig1 = px.bar(
                df_plot,
                x=y_col,
                y=name_col,
                orientation="h",
                text_auto=True,
                color=y_col,
                color_continuous_scale="Blues"
            )
            fig1.update_layout(
                height=380,
                margin=dict(l=10, r=20, t=10, b=10),
                coloraxis_showscale=False,
                xaxis_title="الكمية / القيمة",
                yaxis_title=""
            )
            st.plotly_chart(fig1, use_container_width=True)

    with chart2:
        if not df_emp_raw.empty:
            st.subheader("👥 أداء فريق ميفيدا (UPT & ATV)")
            df_clean = df_emp_raw.copy()
            # استبعاد صفوف الإجمالي المكررة من الجدول لأنها معروضة فوق في الـ KPIs
            mask_total = df_clean.astype(str).apply(lambda r: r.str.contains("Grand Total|Month Total", case=False).any(), axis=1)
            df_clean = df_clean[~mask_total]

            # اختيار الأعمدة المهمة فقط وتنظيفها
            wanted_cols = [c for c in ["sales_person", "Quantity", "Amount", "Order Count", "UPT", "ATV", "ASP"] if c in df_clean.columns]
            if wanted_cols:
                df_clean = df_clean[wanted_cols].sort_values(by="Amount" if "Amount" in wanted_cols else wanted_cols[1], ascending=False)
                df_clean = df_clean.rename(columns={
                    "sales_person": "الموظف",
                    "Quantity": "القطع",
                    "Amount": "المبيعات",
                    "Order Count": "الفواتير"
                })
            st.dataframe(
                df_clean,
                use_container_width=True,
                hide_index=True,
                height=380
            )

st.divider()

# --- القسم الثاني: الشات السريع مع الإيجنت ---
st.subheader("💬 اسأل الإيجنت في الداتا لحظياً")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = [
        {"role": "assistant", "content": "أهلاً يا عمر! الداشبورد اتظبطت وجاهزة. اسألني أي سؤال عن مبيعات ميفيدا، التارجت، الموظفين، أو النواقص."}
    ]

for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("اكتب سؤالك هنا..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("بيسحب الداتا..."):
            system_prompt = f"""
            أنت محلل بيانات سريع ودقيق لمدير فرع ميفيدا (عمر) في شركة 2S Home Wear.
            تاريخ اليوم هو: 2026-11-16.
            معاك أداة `get_mivida_reports` اللي بتسحب قائمة كروت (`card_ids`) بالتوازي في نفس اللحظة.
            خريطة التقارير:
            {MIVIDA_CARDS_CATALOG}

            قواعد السرعة والدقة:
            1. لو السؤال محتاج أكتر من تقرير، ابعت كل أرقام الكروت في قائمة واحدة `card_ids=[...]` في استدعاء واحد للأداة.
            2. صيغ `date_filter`: 'thisday'، 'past1days'، 'past7days'، 'thismonth'، 'YYYY-MM-DD'، أو 'YYYY-MM-DD~YYYY-MM-DD'.
            3. رد مباشرة بالأرقام في نقاط أو جدول واضح وباللهجة المصرية العملية.
            """
            try:
                response = client.models.generate_content(
                    model="gemini-2.5-flash",
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=system_prompt,
                        tools=[get_mivida_reports],
                        temperature=0.1,
                    ),
                )
                answer = response.text
            except Exception as e:
                answer = f"حصلت مشكلة: {e}"

            st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})