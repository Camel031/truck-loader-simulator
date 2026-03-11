# 🚛 Pro 貨車疊貨模擬器

A 3D truck loading simulator built with Streamlit. Input cargo dimensions and quantities to visualize optimal packing arrangements with interactive 3D and engineering 2D views.

**Live Demo:** *(deploy to Streamlit Community Cloud and paste URL here)*

---

## Features

- **3D Interactive View** — rotate and inspect the packing result via Plotly
- **Engineering 2D Views** — top / side / rear orthographic drawings via Matplotlib
- **Three Packing Modes**
  - 🟢 **Strict** — same-type stacking only, grouped by item ID
  - 🟡 **Mixed** — height-first sorting, no cross-type stacking
  - 🔴 **Extreme** — all 6 orientations allowed, cross-type stacking enabled
- **Auto Escalation** — automatically runs Mixed → Extreme if items remain unpacked
- **CSV Import / Export** — upload custom cargo lists; download blank template
- **Truck Presets** — save, rename, and delete custom truck size presets

## Screenshot

> *(add screenshot here)*

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

Open http://localhost:8501 in your browser.

## CSV Format

Upload a cargo list with the following columns:

| Column | Type | Description |
|--------|------|-------------|
| 名稱 | string | Item name |
| 寬 | int (cm) | Width |
| 長 | int (cm) | Length |
| 高 | int (cm) | Height |
| 數量 | int | Quantity |
| 最大堆疊層數 | int (1–10) | Max stack layers |
| 顏色 | hex string | Display color, e.g. `#FF6B6B` |

Sample CSV files are included in the repo:
- `real_cases.csv` — typical AV equipment scenario
- `complex_test.csv` — 18-item stress test

## Deploy to Streamlit Community Cloud

1. Fork or push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo, set main file to `app.py`
4. Click **Deploy**

## Tech Stack

- [Streamlit](https://streamlit.io) — web UI framework
- [Plotly](https://plotly.com/python/) — interactive 3D visualization
- [Matplotlib](https://matplotlib.org) — 2D engineering drawings
- [NumPy](https://numpy.org) — 3D occupancy grid for collision detection
- [Pandas](https://pandas.pydata.org) — cargo data management

## Algorithm

Uses a greedy 3D bin-packing approach:

1. Items are sorted by priority (orphan flag → height → footprint area)
2. For each item, candidate positions are scanned in `(Y, Z, X)` order
3. A voxel grid (`numpy` array, configurable resolution) tracks occupied space
4. Stacking constraints check both support surface and allowed mix policy
5. A configurable `gap` parameter adds clearance between items
