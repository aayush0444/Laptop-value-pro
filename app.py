import streamlit as st
import pandas as pd
from predict_price import predict_price

# ---------- PAGE CONFIG ----------
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
st.sidebar.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
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
      <div class="lp-subtitle">A data science-powered laptop valuation project.</div>
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
                 "Samsung", "Razer", "Huawei", "Microsoft", "Google", "LG", "Xiaomi", "Chuwi"],
            )
            type_name = st.selectbox("Category", ["Notebook", "Ultrabook", "Gaming", "Workstation", "Netbook", "2 in 1 Convertible"])
            os_sys = st.selectbox(
                "Operating system",
                ["Windows 11", "Windows 10", "Windows 7", "macOS", "Mac OS X", "Linux", "Chrome OS", "No OS"],
            )
            ram = st.select_slider("Memory (RAM, GB)", options=[2, 4, 6, 8, 12, 16, 24, 32, 64], value=8)

        with c2:
            st.subheader("Display & Build")
            inches = st.select_slider(
                "Screen size (inches)", 
                options=[10.1, 11.6, 12.0, 12.5, 13.3, 14.0, 15.6, 17.3, 18.4],
                value=15.6
            )
            
            resolution_type = st.selectbox(
                "Resolution quality",
                ["Standard (1366x768)", "Full HD (1920x1080)", "Quad HD (2560x1440)", 
                 "Quad HD+ (3200x1800)", "4K Ultra HD (3840x2160)"]
            )
            
            touchscreen = st.checkbox("Touchscreen")
            ips_panel = st.checkbox("IPS Panel")
            retina_display = st.checkbox("Retina Display")
            
            weight = st.number_input("Weight (kg)", 0.5, 5.0, 2.0, 0.1)

        st.markdown("---")
        
        # CPU Section
        st.subheader("⚙️ Processor (CPU)")
        cpu_col1, cpu_col2, cpu_col3, cpu_col4 = st.columns(4)
        
        with cpu_col1:
            cpu_company = st.selectbox(
                "CPU Brand",
                ["Intel", "AMD", "Samsung", "Apple M-Series", "Other"]
            )
        
        with cpu_col2:
            if cpu_company == "Intel":
                cpu_line = st.selectbox(
                    "Processor Line",
                    ["Core i9", "Core i7", "Core i5", "Core i3", "Xeon", 
                     "Pentium", "Celeron", "Core M", "Atom"]
                )
            elif cpu_company == "AMD":
                cpu_line = st.selectbox(
                    "Processor Line",
                    ["Ryzen 9", "Ryzen 7", "Ryzen 5", "Ryzen 3",
                     "A12-Series", "A10-Series", "A9-Series", "A8-Series", 
                     "A6-Series", "A4-Series", "E-Series"]
                )
            else:
                cpu_line = st.text_input("Processor Line", "Unknown")
        
        with cpu_col3:
            cpu_generation = st.selectbox(
                "Generation",
                [14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
                index=6  # Default to 8th gen
            )
        
        with cpu_col4:
            cpu_type_suffix = st.selectbox(
                "Type",
                ["HK (High Performance Mobile)", 
                 "HQ (High Performance Quad Core)",
                 "H (High Performance)",
                 "HS (High Performance Slim)",
                 "U (Ultra-Low Power)",
                 "Y (Extremely Low Power)",
                 "M (Mobile)",
                 "T (Power Optimized)"],
                index=4  # Default to U
            )
        
        cpu_clock_speed = st.slider("Base Clock Speed (GHz)", 1.0, 5.0, 2.5, 0.1)

        st.markdown("---")
        
        # GPU Section
        st.subheader("🎮 Graphics (GPU)")
        gpu_col1, gpu_col2, gpu_col3 = st.columns(3)
        
        with gpu_col1:
            gpu_company = st.selectbox(
                "GPU Brand",
                ["Intel", "Nvidia", "AMD", "ARM", "Apple", "Other"]
            )
        
        with gpu_col2:
            if gpu_company == "Intel":
                gpu_series = st.selectbox(
                    "GPU Series",
                    ["UHD Graphics", "Iris Xe Graphics", "HD Graphics", 
                     "Iris Plus Graphics", "Iris Pro Graphics", "Iris Graphics"]
                )
            elif gpu_company == "Nvidia":
                gpu_series = st.selectbox(
                    "GPU Series",
                    ["RTX 50 Series", "RTX 40 Series", "RTX 30 Series", "RTX 20 Series",
                     "GTX 16 Series", "GTX 10 Series", "GTX 9 Series", 
                     "GTX 8 Series", "GTX 7 Series", "MX Series", 
                     "Quadro", "GeForce"]
                )
            elif gpu_company == "AMD":
                gpu_series = st.selectbox(
                    "GPU Series",
                    ["Radeon RX 7000 Series", "Radeon RX 6000 Series",
                     "Radeon RX 5000 Series", "Radeon Pro", "Radeon RX", 
                     "Radeon R7", "Radeon R5", "Radeon", "FirePro"]
                )
            elif gpu_company == "ARM":
                gpu_series = st.selectbox("GPU Series", ["Mali"])
            else:
                gpu_series = st.text_input("GPU Series", "Integrated")
        
        with gpu_col3:
            gpu_model = st.text_input("GPU Model (e.g., 3060, 620, 4090)", "")

        st.markdown("---")
        
        # Storage Section
        st.subheader("💾 Storage")
        storage_col1, storage_col2 = st.columns(2)
        
        with storage_col1:
            st.markdown("**Primary Storage**")
            primary_storage_type = st.selectbox(
                "Primary Type",
                ["SSD", "HDD", "Hybrid", "Flash Storage", "None"],
                key="primary_type"
            )
            primary_storage_size = st.selectbox(
                "Primary Size (GB)",
                [0, 128, 256, 512, 1024, 2048],
                index=2,
                key="primary_size"
            )
        
        with storage_col2:
            st.markdown("**Secondary Storage (Optional)**")
            secondary_storage_type = st.selectbox(
                "Secondary Type",
                ["None", "SSD", "HDD", "Hybrid", "Flash Storage"],
                key="secondary_type"
            )
            secondary_storage_size = st.selectbox(
                "Secondary Size (GB)",
                [0, 128, 256, 512, 1024, 2048],
                index=0,
                key="secondary_size"
            )

        submitted = st.form_submit_button("Calculate market value", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    if submitted:
        # Extract suffix code from selection
        suffix_map = {
            "HK (High Performance Mobile)": "HK",
            "HQ (High Performance Quad Core)": "HQ",
            "H (High Performance)": "H",
            "HS (High Performance Slim)": "HS",
            "U (Ultra-Low Power)": "U",
            "Y (Extremely Low Power)": "Y",
            "M (Mobile)": "M",
            "T (Power Optimized)": "T"
        }
        
        # Extract resolution dimensions
        resolution_map = {
            "Standard (1366x768)": ("Standard", 1366, 768),
            "Full HD (1920x1080)": ("Full HD", 1920, 1080),
            "Quad HD (2560x1440)": ("Quad HD", 2560, 1440),
            "Quad HD+ (3200x1800)": ("Quad HD+", 3200, 1800),
            "4K Ultra HD (3840x2160)": ("4K Ultra HD", 3840, 2160)
        }
        
        res_type, res_width, res_height = resolution_map[resolution_type]
        
        # Calculate storage
        storage_dict = {
            "HDD": 0,
            "SSD": 0,
            "Hybrid": 0,
            "Flash_Storage": 0
        }
        
        if primary_storage_type != "None":
            if primary_storage_type == "Flash Storage":
                storage_dict["Flash_Storage"] += primary_storage_size
            else:
                storage_dict[primary_storage_type] += primary_storage_size
        
        if secondary_storage_type != "None":
            if secondary_storage_type == "Flash Storage":
                storage_dict["Flash_Storage"] += secondary_storage_size
            else:
                storage_dict[secondary_storage_type] += secondary_storage_size
        
        user_input = {
            "Company": company,
            "TypeName": type_name,
            "Inches": inches,
            "Ram": ram,
            "Weight": weight,
            "OpSys": os_sys,
            # CPU features
            "cpu_company": cpu_company,
            "cpu_line": cpu_line,
            "cpu_generation": cpu_generation,
            "cpu_type_suffix": suffix_map.get(cpu_type_suffix, "U"),
            "cpu_clock_speed": cpu_clock_speed,
            # Screen features
            "resolution_type": res_type,
            "resolution_width": res_width,
            "resolution_height": res_height,
            "touchscreen": 1 if touchscreen else 0,
            "ips_panel": 1 if ips_panel else 0,
            "retina_display": 1 if retina_display else 0,
            # GPU features
            "gpu_company": gpu_company,
            "gpu_series": gpu_series,
            "gpu_model": gpu_model if gpu_model else "Unknown",
            # Storage
            "HDD": storage_dict["HDD"],
            "SSD": storage_dict["SSD"],
            "Hybrid": storage_dict["Hybrid"],
            "Flash_Storage": storage_dict["Flash_Storage"]
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
            with st.expander("Debug info"):
                st.write(user_input)

with tab_info:
    st.markdown(
        """
<div class="lp-card">
  <h3 style="margin-top:0;">Model & notes</h3>
  <ul style="margin-bottom:0; opacity:0.9;">
    <li>This is a data science project using structured laptop specs to estimate market value.</li>
    <li>All inputs are dropdown-based for accuracy and ease of use.</li>
    <li>The model uses Gradient Boosting trained on 2,500+ laptops from 2017-2020.</li>
    <li>Model accuracy: R² = 86.5%, Mean Absolute Error = ₹8,665</li>
    <li>Select specifications that match your laptop as closely as possible.</li>
    <li>For best results, ensure all hardware specs are accurately selected.</li>
  </ul>
  
  <h4 style="margin-top:1.5rem; margin-bottom:0.5rem;">Coverage</h4>
  <div style="opacity:0.85; font-size:0.9rem;">
    <strong>CPU:</strong> Intel 6th-14th gen, AMD Ryzen 1000-7000 series<br>
    <strong>GPU:</strong> GTX 7/8/9/10/16 series, RTX 20/30/40/50 series, Radeon RX 5000/6000/7000<br>
    <strong>Price Range:</strong> ₹15,000 - ₹400,000
  </div>
</div>
""",
        unsafe_allow_html=True,
    )