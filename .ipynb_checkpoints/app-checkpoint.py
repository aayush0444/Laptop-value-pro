import streamlit as st
import pandas as pd
from predict_price import predict_price

# ---------- PAGE CONFIG ----------ru
st.set_page_config(
    page_title="Laptop Valuer Pro",
    page_icon="💻",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- PREMIUM DARK CSS ----------
st.markdown(
    """
<style>
.stApp{
  background: radial-gradient(1200px 600px at 15% 10%, rgba(255,75,75,0.14), rgba(0,0,0,0) 55%),
              radial-gradient(900px 520px at 85% 0%, rgba(0,255,204,0.10), rgba(0,0,0,0) 55%),
              linear-gradient(180deg, #0b1220 0%, #0a0f1a 100%);
  color: #e9eefc;
}
section.main > div.block-container{
  max-width: 1120px;
  padding-top: 2.2rem;
  padding-bottom: 3rem;
}
.lp-card{
  background: rgba(255,255,255,0.06);
  border: 1px solid rgba(255,255,255,0.10);
  border-radius: 16px;
  padding: 18px 18px;
  box-shadow: 0 10px 30px rgba(0,0,0,0.22);
}
.lp-title{ font-size: 2.25rem; line-height: 1.1; margin: 0; }
.lp-subtitle{ margin-top: 0.35rem; opacity: 0.82; font-size: 1.02rem; }
.lp-badge{
  display:inline-block;
  padding:6px 10px;
  border-radius:999px;
  border:1px solid rgba(255,255,255,0.14);
  background: rgba(255,255,255,0.06);
  font-size:0.85rem;
  opacity:0.9;
}
label, .stMarkdown p{ color: rgba(233,238,252,0.90) !important; }

.stButton > button, div[data-testid="stFormSubmitButton"] > button{
  background: linear-gradient(90deg, #ff4b4b 0%, #ff6b6b 55%, #ff8080 100%) !important;
  color: #0b1220 !important;
  border: 0 !important;
  border-radius: 14px !important;
  padding: 0.85rem 1.05rem !important;
  font-weight: 750 !important;
  letter-spacing: 0.3px !important;
  box-shadow: 0 10px 22px rgba(255,75,75,0.18) !important;
}
.stButton > button:hover, div[data-testid="stFormSubmitButton"] > button:hover{
  transform: translateY(-1px);
  box-shadow: 0 14px 28px rgba(255,75,75,0.24) !important;
}
section[data-testid="stSidebar"]{
  background: rgba(255,255,255,0.04);
  border-right: 1px solid rgba(255,255,255,0.08);
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------- SIDEBAR ----------
st.sidebar.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)  # top spacing
st.sidebar.title("Laptop Valuer Pro")

st.sidebar.markdown(
    """
<div class="lp-card" style="padding:14px;">
  <div style="font-size:0.95rem; opacity:0.85; margin-bottom:6px;">Model confidence</div>
  <div style="font-size:1.6rem; font-weight:800; color:#00c853;">86.5%</div>
</div>

<div class="lp-card" style="padding:14px;">
  <div style="font-size:0.95rem; opacity:0.85; margin-bottom:6px;">MAE</div>
  <div style="font-size:1.6rem; font-weight:800; color:#00c853;">INR 8,665</div>
</div>
""",
    unsafe_allow_html=True,
)

st.sidebar.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)


# ---------- HEADER ----------
st.markdown(
    """
<div class="lp-card">
  <div style="display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;">
    <div>
      <h1 class="lp-title">Laptop Valuer Pro</h1>
      <div class="lp-subtitle">A data science–powered laptop valuation project.</div>
    </div>
    <div class="lp-badge">Gradient Boosting • Market estimate</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

st.write("")

# ---------- MAIN ----------
tab_val, tab_info = st.tabs(["Valuation", "Model & notes"])

with tab_val:
    st.markdown('<div class="lp-card">', unsafe_allow_html=True)

    with st.form("valuation_form", border=False):
        c1, c2 = st.columns(2, gap="large")

        with c1:
            st.subheader("Core identity")
            company = st.selectbox(
                "Manufacturer",
                ["Dell", "HP", "Lenovo", "Asus", "Acer", "MSI", "Apple", "Toshiba",
                 "Samsung", "Razer", "Huawei", "Microsoft", "Google", "LG", "Xiaomi"],
            )
            type_name = st.selectbox("Category", ["Notebook", "Ultrabook", "Gaming", "Workstation", "Netbook"])
            os_sys = st.selectbox(
                "Operating system",
                ["Windows 11","Windows 10", "Windows 7", "macOS", "Mac OS X", "Linux", "Chrome OS", "No OS"],
            )
            ram = st.select_slider("Memory (RAM, GB)", options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)

        with c2:
            st.subheader("Hardware specs")
            sub1, sub2 = st.columns(2, gap="medium")
            with sub1:
                inches = st.number_input("Screen size (inches)", 10.0, 18.0, 15.6, 0.1)
                weight = st.number_input("Weight (kg)", 0.5, 5.0, 2.0, 0.1)
            with sub2:
                resolution = st.text_input("Resolution", "1920x1080")
                storage = st.text_input("Storage", "256GB SSD")

            with st.expander("Advanced (optional)", expanded=False):
                cpu = st.text_input("Processor (CPU)", "Intel Core i5 8250U 1.6GHz")
                gpu = st.text_input("Graphics (GPU)", "Intel HD Graphics 620")

        submitted = st.form_submit_button("Calculate market value", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        user_input = {
            "Company": company,
            "TypeName": type_name,
            "Inches": inches,
            "Ram": ram,
            "Weight": weight,
            "cpu": cpu if "cpu" in locals() else "Unknown",
            "ScreenResolution": resolution,
            "Gpu": gpu if "gpu" in locals() else "Unknown",
            "OpSys": os_sys,
            "Memory": storage,
        }

        try:
            with st.spinner("Estimating value..."):
                predicted_price = predict_price(user_input)

            st.write("")
            st.markdown(
                f"""
<div class="lp-card" style="border: 1px solid rgba(255,75,75,0.35);">
  <div style="display:flex; align-items:flex-start; justify-content:space-between; gap:14px; flex-wrap:wrap;">
    <div>
      <div style="opacity:0.78; font-size:0.95rem;">Estimated valuation</div>
      <div style="font-size:2.4rem; font-weight:800; margin-top:6px;">₹{predicted_price:,.2f}</div>
      <div style="margin-top:8px; opacity:0.75; font-size:0.95rem;">Adjust one field at a time for better comparisons.</div>
    </div>
    <div style="min-width:220px;">
      <div class="lp-badge" style="display:block; text-align:center;">Prediction ready</div>
    </div>
  </div>
</div>
""",
                unsafe_allow_html=True,
            )

        except Exception as e:
            st.error(f"Prediction engine error: {e}")

with tab_info:
    st.markdown(
        """
<div class="lp-card">
  <h3 style="margin-top:0;">Model & notes</h3>
  <ul style="margin-bottom:0; opacity:0.9;">
    <li>This is a data science project using structured laptop specs to estimate market value.</li>
    <li>Use Advanced fields only when needed to keep the interface clean.</li>
  </ul>
</div>
""",
        unsafe_allow_html=True,
    )
