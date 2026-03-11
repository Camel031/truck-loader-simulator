import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.font_manager as fm
import platform
import copy
import time
import os

# ==========================================
# 0. 系統設定 & 字體與圖表修正
# ==========================================
def get_font_and_labels():
    """
    偵測中文字體，若失敗則回傳英文標籤，避免亂碼方塊。
    """
    # 常見中文字體路徑與名稱
    possible_fonts = [
        '/mnt/c/Windows/Fonts/msjh.ttc', # WSL Windows Font
        'C:\\Windows\\Fonts\\msjh.ttc',  # Windows Local
        'Microsoft JhengHei', 'SimHei', 'PingFang TC', 'Heiti TC', 'Noto Sans CJK JP'
    ]
    
    found_font = None
    for f in possible_fonts:
        try:
            # 如果是路徑，檢查是否存在
            if '/' in f or '\\' in f:
                if os.path.exists(f):
                    fm.fontManager.addfont(f)  # 確保註冊到 FontManager，避免找不到字體導致亂碼
                    prop = fm.FontProperties(fname=f)
                    font_name = prop.get_name()
                    plt.rcParams['font.family'] = font_name
                    plt.rcParams['font.sans-serif'] = [font_name]
                    found_font = font_name
                    break
            else:
                # 如果是名稱，嘗試設定並驗證
                try:
                    fm.findfont(f, fallback_to_default=False)
                    plt.rcParams['font.family'] = f
                    plt.rcParams['font.sans-serif'] = [f]
                    found_font = f
                    break
                except:
                    pass
        except:
            continue
            
    plt.rcParams['axes.unicode_minus'] = False

    # 定義標籤字典
    if found_font:
        return {
            'top_title': '俯視圖 (Top View)', 'side_title': '側視圖 (Side View)', 'rear_title': '後視圖 (Rear View)',
            'w': '車寬 (X)', 'l': '車長 (Y)', 'h': '車高 (Z)',
            'cab': '車頭 (Cab)', 'door': '車尾 (Door)', 'cab_arrow': '車頭',
            'use_en': False
        }
    else:
        return {
            'top_title': 'Top View', 'side_title': 'Side View', 'rear_title': 'Rear View',
            'w': 'Width (X)', 'l': 'Length (Y)', 'h': 'Height (Z)',
            'cab': 'Cab', 'door': 'Door', 'cab_arrow': 'Cab',
            'use_en': True
        }

LABELS = get_font_and_labels()

# ==========================================
# 1. CSS 樣式注入
# ==========================================
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] { min-width: 400px; max-width: 500px; }
    button[data-baseweb="tab"] { font-size: 16px; font-weight: bold; }
    /* Metric 放大 */
    [data-testid="stMetricValue"] { font-size: 32px; }
    [data-testid="stMetricLabel"] { font-size: 14px; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ==========================================
# 2. 演算法核心
# ==========================================

class Item:
    def __init__(self, item_id, name, width, length, height, color, max_stack=1, is_orphan=False):
        self.id = item_id 
        self.name = name
        self.width = int(width)
        self.length = int(length)
        self.height = int(height)
        self.color = color
        self.max_stack = int(max_stack)
        self.position = None
        self.placed_dim = (self.width, self.length, self.height)
        self.orig_long_side = max(self.width, self.length)
        self.is_orphan = is_orphan

class TruckPacker:
    def __init__(self, truck_w, truck_l, truck_h, gap=1.0, resolution=2):
        self.truck_w = int(truck_w)
        self.truck_l = int(truck_l)
        self.truck_h = int(truck_h)
        self.gap = float(gap)
        self.resolution = int(resolution) 
        self.packed_items = []
        self.unpacked_items = []
        self.grid_shape = (
            int(self.truck_w / self.resolution) + 1,
            int(self.truck_l / self.resolution) + 1,
            int(self.truck_h / self.resolution) + 1
        )
        self.grid = np.zeros(self.grid_shape, dtype=np.int16)
        self.valid_z_levels = {0} 
        self.max_occupied_y = 0

    def _check_fit(self, item, dim_w, dim_l, dim_h, x, y, z, allow_mix_stacking):
        eff_w = dim_w + self.gap
        eff_l = dim_l + self.gap
        
        if (x + eff_w > self.truck_w or y + eff_l > self.truck_l or z + dim_h > self.truck_h):
            return False
        
        res = self.resolution
        gx, gy, gz = int(x/res), int(y/res), int(z/res)
        gw, gl, gh = int(eff_w/res), int(eff_l/res), int(dim_h/res)
        
        if gx + gw > self.grid_shape[0] or gy + gl > self.grid_shape[1] or gz + gh > self.grid_shape[2]:
            return False

        region = self.grid[gx:gx+gw, gy:gy+gl, gz:gz+gh]
        if np.any(region != 0):
            return False
        
        if z > 0:
            support_z = max(0, gz - 1)
            # 支撐檢查只看實際佔用 (不含 gap) 以避免緩衝區阻擋疊放
            support_w = max(1, int(np.ceil(dim_w / self.resolution)))
            support_l = max(1, int(np.ceil(dim_l / self.resolution)))
            support_region = self.grid[gx:gx+support_w, gy:gy+support_l, support_z]
            if np.any(support_region == 0):
                return False
            if not allow_mix_stacking and not np.all(support_region == item.id):
                return False
        return True

    def _mark_grid(self, item_id, dim_w, dim_l, dim_h, x, y, z):
        eff_w = dim_w + self.gap
        eff_l = dim_l + self.gap
        res = self.resolution
        gx, gy, gz = int(x/res), int(y/res), int(z/res)
        gw, gl, gh = int(eff_w/res), int(eff_l/res), int(dim_h/res)
        
        self.grid[gx:gx+gw, gy:gy+gl, gz:gz+gh] = item_id
        
        new_top_z = z + dim_h
        if new_top_z < self.truck_h:
            self.valid_z_levels.add(new_top_z)
        
        new_far_y = y + eff_l
        if new_far_y > self.max_occupied_y:
            self.max_occupied_y = new_far_y

    def pack(self, items_input, mode='strict', progress_cb=None):
        items = copy.deepcopy(items_input)
        self.packed_items = []
        self.unpacked_items = []
        self.grid = np.zeros(self.grid_shape, dtype=np.int16)
        self.valid_z_levels = {0}
        self.max_occupied_y = 0
        total_items = len(items)

        allow_mix_stacking = False
        
        if mode == 'strict':
            items.sort(key=lambda x: (x.is_orphan, x.id, -x.height, -x.width * x.length))
            allow_mix_stacking = False
        elif mode == 'mixed':
            items.sort(key=lambda x: (x.is_orphan, -x.height, -x.width * x.length))
            allow_mix_stacking = False
        elif mode == 'extreme':
            items.sort(key=lambda x: (-x.height, -x.width * x.length))
            allow_mix_stacking = True

        step_xy = self.resolution

        for idx, item in enumerate(items):
            placed = False
            w, l, h = item.width, item.length, item.height

            if mode in ['strict', 'mixed']:
                orientations = [(w, l, h), (l, w, h)]
            else:
                orientations = [
                    (w, l, h), (l, w, h), 
                    (w, h, l), (h, w, l), 
                    (l, h, w), (h, l, w)
                ]
            
            search_limit_y = min(self.truck_l, int(self.max_occupied_y + max(w, l) + 50)) 
            
            for y in range(0, search_limit_y, step_xy):
                if placed: break
                sorted_z_levels = sorted(list(self.valid_z_levels))
                for z in sorted_z_levels:
                    if placed: break
                    current_layer = int(z / item.height) + 1
                    if current_layer > item.max_stack: continue
                    
                    for x in range(0, self.truck_w, step_xy):
                        for (dw, dl, dh) in orientations:
                            if self._check_fit(item, dw, dl, dh, x, y, z, allow_mix_stacking):
                                item.position = (x, y, z)
                                item.placed_dim = (dw, dl, dh)
                                self.packed_items.append(item)
                                self._mark_grid(item.id, dw, dl, dh, x, y, z)
                                placed = True
                                break 
                        if placed: break
            if not placed:
                self.unpacked_items.append(item)
            if progress_cb:
                progress_cb(idx + 1, total_items, item)
        return len(self.packed_items), len(self.unpacked_items)


def get_cube_wireframe(x, y, z, dx, dy, dz):
    X = [x, x+dx, x+dx, x, x,  x, x+dx, x+dx, x, x,  x, x, x+dx, x+dx, x+dx, x+dx]
    Y = [y, y, y+dy, y+dy, y,  y, y, y+dy, y+dy, y,  y, y, y, y, y+dy, y+dy]
    Z = [z, z, z, z, z,        z+dz, z+dz, z+dz, z+dz, z+dz, z, z+dz, z, z+dz, z, z+dz]
    return X, Y, Z

def draw_truck_3d(packer):
    truck_w, truck_l, truck_h = packer.truck_w, packer.truck_l, packer.truck_h
    fig = go.Figure()

    fig.add_trace(go.Mesh3d(
        x=[0, truck_w, truck_w, 0], y=[0, 0, 0, 0], z=[0, 0, truck_h, truck_h],
        color='white', opacity=0.1, name='車頭'
    ))
    
    lines_x = [0, truck_w, truck_w, 0, 0, 0, truck_w, truck_w, 0, 0, 0, 0, truck_w, truck_w, truck_w, truck_w]
    lines_y = [0, 0, truck_l, truck_l, 0, 0, 0, truck_l, truck_l, 0, 0, truck_l, truck_l, 0, 0, truck_l]
    lines_z = [0, 0, 0, 0, 0, truck_h, truck_h, truck_h, truck_h, truck_h, 0, 0, 0, 0, truck_h, truck_h]
    fig.add_trace(go.Scatter3d(x=lines_x, y=lines_y, z=lines_z, mode='lines', line=dict(color='white', width=4), hoverinfo='none'))

    fig.add_trace(go.Scatter3d(
        x=[truck_w/2, truck_w/2], y=[0, truck_l], z=[truck_h*1.1, truck_h*1.1],
        mode='text', text=[LABELS['cab'], LABELS['door']],
        textfont=dict(color="white", size=14)
    ))

    for item in packer.packed_items:
        x, y, z = item.position
        dx, dy, dz = item.placed_dim
        
        fig.add_trace(go.Mesh3d(
            x=[x, x+dx, x+dx, x, x, x+dx, x+dx, x],
            y=[y, y, y+dy, y+dy, y, y, y+dy, y+dy],
            z=[z, z, z, z, z+dz, z+dz, z+dz, z+dz],
            i=[7, 0, 0, 0, 4, 4, 6, 6, 4, 0, 3, 2],
            j=[3, 4, 1, 2, 5, 6, 5, 2, 0, 1, 6, 3],
            k=[0, 7, 2, 3, 6, 7, 1, 1, 5, 5, 7, 6],
            opacity=1.0, color=item.color, flatshading=True,
            name=item.name, hoverinfo='text',
            text=f"{item.name}<br>{dx}x{dy}x{dz}"
        ))
        wx, wy, wz = get_cube_wireframe(x, y, z, dx, dy, dz)
        fig.add_trace(go.Scatter3d(x=wx, y=wy, z=wz, mode='lines', line=dict(color='black', width=2), hoverinfo='skip'))

    fig.update_layout(
        scene=dict(
            xaxis=dict(title='X', range=[0, max(truck_w, truck_l)], visible=True, color='white'),
            yaxis=dict(title='Y', range=[0, max(truck_w, truck_l)], visible=True, color='white'),
            zaxis=dict(title='Z', range=[0, max(truck_w, truck_l)], visible=True, color='white'),
            aspectmode='data', bgcolor='rgba(0,0,0,0)'
        ),
        margin=dict(l=0, r=0, b=0, t=0), height=600, showlegend=False, paper_bgcolor='rgba(0,0,0,0)'
    )
    return fig

def draw_2d_views(packer):
    truck_w, truck_l, truck_h = packer.truck_w, packer.truck_l, packer.truck_h
    packed_items = packer.packed_items
    
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 7))

    # --- 1. Top View (俯視) ---
    ax1.add_patch(patches.Rectangle((0, 0), truck_w, truck_l, fill=False, edgecolor='k', lw=2))
    # V7.5: 反轉 X 軸 (左右互換) 以匹配 3D 視角
    ax1.set_xlim(truck_w+10, -10) 
    ax1.set_ylim(truck_l+20, -20) # 0在上(Cab), L在下(Door)
    ax1.set_aspect('equal')
    ax1.set_title(LABELS['top_title'], fontsize=14)
    ax1.set_xlabel(LABELS['w'])
    ax1.set_ylabel(LABELS['l'])
    
    # 標示一次即可：置於右上/右下框外，避免重複與亂碼
    ax1.annotate(f"{LABELS['cab']}", xy=(truck_w, -5), xytext=(truck_w+25, -5),
                 ha='left', va='center', color='red', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7), clip_on=False)
    ax1.annotate(f"{LABELS['door']}", xy=(truck_w, truck_l+5), xytext=(truck_w+25, truck_l+5),
                 ha='left', va='center', color='red', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7), clip_on=False)

    for item in packed_items:
        x, y, z = item.position
        dx, dy, dz = item.placed_dim
        ax1.add_patch(patches.Rectangle((x, y), dx, dy, edgecolor='k', facecolor=item.color, alpha=0.8))

    # --- 2. Side View (側視) ---
    ax2.add_patch(patches.Rectangle((0, 0), truck_l, truck_h, fill=False, edgecolor='k', lw=2))
    ax2.set_xlim(-20, truck_l+20)
    ax2.set_ylim(-10, truck_h+10)
    ax2.set_aspect('equal')
    ax2.set_title(LABELS['side_title'], fontsize=14)
    ax2.set_xlabel(LABELS['l'])
    ax2.set_ylabel(LABELS['h'])
    
    # 側視車頭標示：簡單文字置於左外側
    ax2.annotate(f"{LABELS['cab_arrow']}", xy=(0, truck_h/2), xytext=(-20, truck_h/2),
                 va='center', ha='right', color='red', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7), clip_on=False)

    for item in packed_items:
        x, y, z = item.position
        dx, dy, dz = item.placed_dim
        ax2.add_patch(patches.Rectangle((y, z), dy, dz, edgecolor='k', facecolor=item.color, alpha=0.8))

    # --- 3. Rear View (後視) ---
    ax3.add_patch(patches.Rectangle((0, 0), truck_w, truck_h, fill=False, edgecolor='k', lw=2))
    # V7.5: 反轉 X 軸 (左右互換)
    ax3.set_xlim(truck_w+10, -10)
    ax3.set_ylim(-10, truck_h+10)
    ax3.set_aspect('equal')
    ax3.set_title(LABELS['rear_title'], fontsize=14)
    ax3.set_xlabel(LABELS['w'])
    ax3.set_ylabel(LABELS['h'])

    sorted_items = sorted(packed_items, key=lambda i: i.position[1]) 
    for item in sorted_items:
        x, y, z = item.position
        dx, dy, dz = item.placed_dim
        ax3.add_patch(patches.Rectangle((x, z), dx, dz, edgecolor='k', facecolor=item.color, alpha=0.9))

    plt.tight_layout()
    return fig

# ==========================================
# 4. Streamlit 介面
# ==========================================

st.set_page_config(page_title="Pro Truck Loader V7.5", layout="wide", initial_sidebar_state="expanded")

if 'results' not in st.session_state: st.session_state.results = {} 
if 'is_dirty' not in st.session_state: st.session_state.is_dirty = False 
if 'truck_presets' not in st.session_state:
    st.session_state.truck_presets = {
        "預設 3.5噸": (220, 450, 200),
        "預設 15噸": (240, 850, 250)
    }
if 'preset_input_val' not in st.session_state: st.session_state.preset_input_val = ""
if 'cargo_data' not in st.session_state:
    st.session_state.cargo_data = pd.DataFrame([
        {"名稱": "ACME XA-1000", "寬": 50, "長": 65, "高": 70, "數量": 8, "最大堆疊層數": 2, "顏色": "#FF6B6B"},
        {"名稱": "ACME TB-1230", "寬": 100, "長": 30, "高": 30, "數量": 15, "最大堆疊層數": 5, "顏色": "#4ECDC4"},
        {"名稱": "JOLLY Q4 Plus", "寬": 40, "長": 40, "高": 50, "數量": 12, "最大堆疊層數": 3, "顏色": "#FFE66D"},
        {"名稱": "長線箱 Cable", "寬": 60, "長": 110, "高": 60, "數量": 4, "最大堆疊層數": 1, "顏色": "#1A535C"},
    ])

def set_dirty():
    st.session_state.is_dirty = True

st.title("Pro 貨車疊貨模擬器 V7.5")

# --- 側邊欄 ---
with st.sidebar:
    st.header("貨車參數")
    
    preset_names = list(st.session_state.truck_presets.keys())
    selected_preset = st.selectbox("選擇車型", ["自訂尺寸"] + preset_names)
    if 'prev_preset' not in st.session_state:
        st.session_state.prev_preset = selected_preset
    # 取得預設值
    if selected_preset != "自訂尺寸":
        p_w, p_l, p_h = st.session_state.truck_presets[selected_preset]
    else:
        p_w, p_l, p_h = st.session_state.get('truck_dims', (220, 450, 200))

    # 若切換車型，覆寫輸入框的 state，避免舊值殘留
    if selected_preset != st.session_state.prev_preset:
        st.session_state["w_in"] = p_w
        st.session_state["l_in"] = p_l
        st.session_state["h_in"] = p_h
        st.session_state.prev_preset = selected_preset
        set_dirty()

    truck_w = st.number_input("寬 (Width)", value=st.session_state.get("w_in", p_w), step=10, on_change=set_dirty, key="w_in")
    truck_l = st.number_input("長 (Length)", value=st.session_state.get("l_in", p_l), step=10, on_change=set_dirty, key="l_in")
    truck_h = st.number_input("高 (Height)", value=st.session_state.get("h_in", p_h), step=10, on_change=set_dirty, key="h_in")
    st.session_state.truck_dims = (truck_w, truck_l, truck_h)
    st.session_state.truck_dims = (truck_w, truck_l, truck_h)
    
    with st.expander("預設管理", expanded=False):
        new_preset_name = st.text_input("預設名稱", value=st.session_state.preset_input_val, key="preset_input_key")
        
        col_p1, col_p2 = st.columns(2)
        with col_p1:
            if st.button("儲存", use_container_width=True):
                if new_preset_name:
                    st.session_state.truck_presets[new_preset_name] = (truck_w, truck_l, truck_h)
                    st.session_state.preset_input_val = "" 
                    st.success(f"已儲存")
                    st.rerun()
        with col_p2:
            if st.button("刪除", use_container_width=True):
                if selected_preset != "自訂尺寸":
                    del st.session_state.truck_presets[selected_preset]
                    st.success(f"已刪除")
                    st.rerun()
        if st.button("重命名目前項目", use_container_width=True):
            if selected_preset != "自訂尺寸" and new_preset_name:
                st.session_state.truck_presets[new_preset_name] = st.session_state.truck_presets[selected_preset]
                del st.session_state.truck_presets[selected_preset]
                st.session_state.preset_input_val = ""
                st.success(f"已重命名")
                st.rerun()

    st.divider()
    gap = st.number_input("緩衝 (Gap)", value=1.0, step=0.1, format="%.1f", on_change=set_dirty)
    resolution = st.select_slider("精度 (cm)", [1, 2, 5], value=2, on_change=set_dirty)

# --- 貨物清單 ---
if st.session_state.is_dirty and st.session_state.results:
    st.warning("資料已變更，請重新計算。")

st.subheader("貨物清單")

required_cols = ["名稱", "寬", "長", "高", "數量", "最大堆疊層數", "顏色"]
up_col, dl_col = st.columns([2.4, 1.1])
with up_col:
    st.markdown("**上傳貨物清單 (CSV)**")
    uploaded_file = st.file_uploader("上傳 CSV 貨物清單", type=["csv"], label_visibility="collapsed")
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            if not all(col in df_upload.columns for col in required_cols):
                missing = [c for c in required_cols if c not in df_upload.columns]
                st.error(f"CSV 缺少欄位: {missing}")
            else:
                st.session_state.cargo_data = df_upload
                set_dirty()
                st.success("已載入 CSV，請確認並重新計算。")
        except Exception as e:
            st.error(f"讀取失敗: {e}")
with dl_col:
    st.markdown("**下載 CSV 範本**")
    template_df = pd.DataFrame(columns=required_cols)
    template_csv = template_df.to_csv(index=False).encode('utf-8-sig')
    st.download_button("下載 CSV 範本", data=template_csv, file_name="cargo_template.csv", mime="text/csv", use_container_width=True)
    st.caption(f"欄位：{', '.join(required_cols)}")

edited_df = st.data_editor(
    st.session_state.cargo_data,
    num_rows="dynamic",
    column_config={
        "顏色": st.column_config.TextColumn("顏色", validate="^#[0-9a-fA-F]{6}$"),
        "最大堆疊層數": st.column_config.NumberColumn(min_value=1, max_value=10)
    },
    use_container_width=True,
    on_change=set_dirty
)
st.markdown("</div>", unsafe_allow_html=True)

# --- 計算按鈕與功能函數 ---
def prepare_items():
    raw_items = []
    idx_counter = 1
    for index, row in edited_df.iterrows():
        try:
            c = int(row["數量"])
            if c > 0:
                max_stack = int(row["最大堆疊層數"])
                remainder = c % max_stack
                full_stacks_count = c - remainder
                for _ in range(full_stacks_count):
                    raw_items.append(Item(idx_counter, row["名稱"], row["寬"], row["長"], row["高"], row["顏色"], row["最大堆疊層數"], is_orphan=False))
                for _ in range(remainder):
                    raw_items.append(Item(idx_counter, row["名稱"], row["寬"], row["長"], row["高"], row["顏色"], row["最大堆疊層數"], is_orphan=True))
                idx_counter += 1
        except: pass
    return raw_items

# V7.5: 單獨計算某個模式的函數
def run_packer(mode, items, progress_slot=None):
    mode_label = {
        'strict': "正常 (Strict)",
        'mixed': "混合 (Mixed)",
        'extreme': "極限 (Extreme)"
    }.get(mode, mode)

    progress_bar = None
    if progress_slot is not None:
        progress_bar = progress_slot.progress(0, text=f"計算中：{mode_label}")

    def _progress(done, total, item):
        if progress_bar:
            progress_bar.progress(
                done / total,
                text=f"計算中：{mode_label} ({done}/{total}) - {item.name}"
            )

    packer = TruckPacker(truck_w, truck_l, truck_h, gap, resolution)
    packer.pack(items, mode=mode, progress_cb=_progress if progress_bar else None)

    if progress_slot is not None:
        progress_slot.empty()
    return packer

# 主計算按鈕
if st.button("開始計算 (Start Calculation)", type="primary", use_container_width=True):
    items = prepare_items()
    if not items:
        st.error("清單為空")
    else:
        st.session_state.results = {} # 重置
        progress_placeholder = st.empty()
        # 先算 Strict
        st.session_state.results['strict'] = run_packer('strict', items, progress_placeholder)
        # 若還有未裝載，自動續算 Mixed
        if st.session_state.results['strict'].unpacked_items:
            st.session_state.results['mixed'] = run_packer('mixed', items, progress_placeholder)
        # 若 Mixed 仍有未裝載，自動續算 Extreme
        if 'mixed' in st.session_state.results and st.session_state.results['mixed'].unpacked_items:
            st.session_state.results['extreme'] = run_packer('extreme', items, progress_placeholder)
        elif 'mixed' not in st.session_state.results and st.session_state.results['strict'].unpacked_items:
            # 如果未跑混合但 strict 仍有剩餘，直接跑 extreme
            st.session_state.results['extreme'] = run_packer('extreme', items, progress_placeholder)
        st.session_state.is_dirty = False
        st.rerun()

# --- 結果顯示 ---
if st.session_state.results:
    
    # 建立 Tabs，無論是否有結果都顯示 Tab
    tab_labels = ["正常 (Strict)", "混合 (Mixed)", "極限 (Extreme)"]
    tabs = st.tabs(tab_labels)
    
    modes = ['strict', 'mixed', 'extreme']
    
    items = prepare_items() # 準備資料以備手動計算使用

    for i, mode in enumerate(modes):
        with tabs[i]:
            # 檢查該模式是否已計算
            if mode in st.session_state.results:
                packer = st.session_state.results[mode]
                
                # 結果顯示
                leftover = len(packer.unpacked_items)
                if leftover == 0:
                    st.success(f"全部裝載完成 (剩餘 0 箱)")
                else:
                    st.error(f"還有 {leftover} 箱裝不下")
                
                st.markdown("### 裝載摘要")
                c1, c2, c3 = st.columns(3)
                total_vol = packer.truck_w * packer.truck_l * packer.truck_h
                used_vol = sum([i.width * i.length * i.height for i in packer.packed_items])
                
                c1.metric("空間利用率", f"{(used_vol/total_vol)*100:.1f}%")
                c2.metric("成功裝載", f"{len(packer.packed_items)} 箱")
                c3.metric("未裝載", f"{len(packer.unpacked_items)} 箱")
                
                sub_t1, sub_t2, sub_t3 = st.tabs(["3D 視圖", "工程三視圖", "清單"])
                with sub_t1:
                    st.plotly_chart(draw_truck_3d(packer), use_container_width=True)
                    used_names = sorted(list(set([item.name for item in packer.packed_items])))
                    cols_legend = st.columns(len(used_names) if len(used_names)>0 else 1)
                    for idx, name in enumerate(used_names):
                        color = next((x.color for x in packer.packed_items if x.name == name), "#000")
                        cols_legend[idx%len(cols_legend)].markdown(f"<span style='color:{color}'>■</span> {name}", unsafe_allow_html=True)
                with sub_t2:
                    st.pyplot(draw_2d_views(packer))
                with sub_t3:
                    data = []
                    for item in packer.packed_items:
                        direction = "直" if item.placed_dim[1] == item.orig_long_side else "橫"
                        data.append({
                            "名稱": item.name, "X": item.position[0], "Y": item.position[1], "Z": item.position[2], 
                            "方向": direction, "屬性": "落單" if item.is_orphan else "成套"
                        })
                    st.dataframe(pd.DataFrame(data), use_container_width=True)
            
            else:
                # 尚未計算該模式
                st.info(f"尚未計算【{tab_labels[i]}】模式。")
                if st.button(f"計算 {tab_labels[i]}", key=f"btn_{mode}"):
                    progress_placeholder = st.empty()
                    st.session_state.results[mode] = run_packer(mode, items, progress_placeholder)
                    st.rerun()
