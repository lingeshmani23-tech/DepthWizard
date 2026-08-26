"""
DEPTHWIZARD — SIH HTML Report Generator
Generates outputs/demo_report/FINAL_REPORT.html with dark mode aesthetic,
embedded annotated figures, metrics tables, video player, and JSON traceability.
"""

import os
import json
from pathlib import Path
from typing import Dict, Any

def generate_html_report(results_data: Dict[str, Any], report_dir: str) -> str:
    """
    Builds and saves outputs/demo_report/FINAL_REPORT.html.
    """
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, "FINAL_REPORT.html")

    ref_info = results_data.get("reference", {})
    tgt_info = results_data.get("target", {})
    depth_info = results_data.get("depth", {})
    calib_info = results_data.get("calibration", {})
    height_info = results_data.get("height_estimation", {})
    eval_info = results_data.get("evaluation", {})
    pc_info = results_data.get("pointcloud", {})

    est_h = height_info.get("estimated_height_m", 0.0)
    known_h = tgt_info.get("known_height_m", 0.0)
    abs_err = eval_info.get("absolute_error_m", 0.0)
    pct_err = eval_info.get("percentage_error", 0.0)

    pc_bounds = pc_info.get("bounds", {})
    min_b = pc_bounds.get("min", [0.0, 0.0, 0.0])
    max_b = pc_bounds.get("max", [0.0, 0.0, 0.0])

    xmin, ymin, zmin = min_b[0], min_b[1], min_b[2]
    xmax, ymax, zmax = max_b[0], max_b[1], max_b[2]

    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>DEPTHWIZARD — SIH Demonstration Report</title>
    <link rel="preconnect" href="https://fonts.googleapis.com">
    <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
    <style>
        :root {{
            --bg-dark: #0B0F19;
            --bg-card: rgba(30, 41, 59, 0.75);
            --border-color: rgba(56, 189, 248, 0.25);
            --primary: #38BDF8;
            --primary-gradient: linear-gradient(135deg, #38BDF8 0%, #818CF8 100%);
            --accent: #4F46E5;
            --text-main: #F3F4F6;
            --text-muted: #9CA3AF;
            --success: #34D399;
        }}
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{
            background-color: var(--bg-dark);
            color: var(--text-main);
            font-family: 'Inter', system-ui, sans-serif;
            line-height: 1.6;
            padding: 2rem 1rem;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            text-align: center;
            padding: 2.5rem 1rem;
            margin-bottom: 2rem;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            backdrop-filter: blur(12px);
        }}
        .badge {{
            display: inline-block;
            background: rgba(56, 189, 248, 0.15);
            border: 1px solid var(--border-color);
            color: var(--primary);
            padding: 0.4rem 1rem;
            border-radius: 50px;
            font-size: 0.85rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-bottom: 1rem;
        }}
        .header h1 {{
            font-size: 2.8rem;
            font-weight: 800;
            background: var(--primary-gradient);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem;
        }}
        .header p {{
            color: var(--text-muted);
            font-size: 1.1rem;
        }}
        
        /* Metric Overview Cards */
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
            gap: 1.2rem;
            margin-bottom: 2.5rem;
        }}
        .metric-card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 14px;
            padding: 1.5rem;
            text-align: center;
        }}
        .metric-val {{
            font-size: 2.2rem;
            font-weight: 800;
            color: var(--primary);
            font-family: 'JetBrains Mono', monospace;
        }}
        .metric-lbl {{
            font-size: 0.85rem;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.05em;
            margin-top: 0.4rem;
        }}

        /* Section Layout */
        .section {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 16px;
            padding: 2rem;
            margin-bottom: 2rem;
        }}
        .section-title {{
            font-size: 1.4rem;
            font-weight: 700;
            color: #FFF;
            margin-bottom: 1.2rem;
            display: flex;
            align-items: center;
            gap: 10px;
            border-bottom: 1px solid rgba(255, 255, 255, 0.1);
            padding-bottom: 0.6rem;
        }}
        .grid-2 {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 1.8rem;
        }}
        @media (max-width: 800px) {{ .grid-2 {{ grid-template-columns: 1fr; }} }}

        .report-img {{
            width: 100%;
            border-radius: 12px;
            border: 1px solid var(--border-color);
            display: block;
        }}
        
        .code-box {{
            background: #05070D;
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 1rem;
            font-family: 'JetBrains Mono', monospace;
            font-size: 0.9rem;
            color: #38BDF8;
            white-space: pre-wrap;
            margin-top: 0.8rem;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 1rem;
        }}
        th, td {{
            padding: 0.8rem 1rem;
            text-align: left;
            border-bottom: 1px solid rgba(255, 255, 255, 0.08);
        }}
        th {{
            color: var(--primary);
            font-size: 0.8rem;
            text-transform: uppercase;
        }}

        /* Pipeline Diagram */
        .pipeline-flow {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: #05070D;
            border: 1px solid var(--border-color);
            border-radius: 12px;
            padding: 1.2rem 1.5rem;
            margin-top: 1rem;
            overflow-x: auto;
        }}
        .flow-step {{
            text-align: center;
            font-weight: 700;
            font-size: 0.9rem;
            color: var(--primary);
        }}
        .flow-arrow {{
            color: var(--text-muted);
            font-size: 1.2rem;
        }}
    </style>
</head>
<body>
    <div class="container">
        <!-- Header -->
        <div class="header">
            <span class="badge"><i class="fa-solid fa-wand-magic-sparkles"></i> SIH Demonstration Report</span>
            <h1>DEPTHWIZARD</h1>
            <p>Single-View Metric Height Estimation & 3D Virtual Camera Flythrough</p>
        </div>

        <!-- Top Metrics Overview -->
        <div class="metrics-grid">
            <div class="metric-card">
                <div class="metric-val">{est_h:.2f} m</div>
                <div class="metric-lbl">Estimated Target Height</div>
            </div>
            <div class="metric-card">
                <div class="metric-val">{known_h:.2f} m</div>
                <div class="metric-lbl">Known Ground Truth</div>
            </div>
            <div class="metric-card">
                <div class="metric-val" style="color: var(--success);">{abs_err:.3f} m</div>
                <div class="metric-lbl">Absolute Metric Error</div>
            </div>
            <div class="metric-card">
                <div class="metric-val" style="color: var(--success);">{pct_err:.2f} %</div>
                <div class="metric-lbl">Percentage Error</div>
            </div>
        </div>

        <!-- 1. INPUT IMAGE -->
        <div class="section">
            <div class="section-title"><i class="fa-solid fa-image"></i> 1. Input RGB Image</div>
            <div class="grid-2">
                <img src="01_original.png" class="report-img" alt="Input Image">
                <div>
                    <p style="color: var(--text-muted);">
                        Outdoor single RGB photograph containing reference person and target building structure.
                    </p>
                    <div class="code-box">
Image File: sih_demo.jpg<br>
Resolution: {results_data.get('image_resolution', '640x512')}<br>
Channels: 3 (RGB)
                    </div>
                </div>
            </div>
        </div>

        <!-- 2. DEPTH ESTIMATION -->
        <div class="section">
            <div class="section-title"><i class="fa-solid fa-layer-group"></i> 2. Monocular Relative Depth Estimation</div>
            <div class="grid-2">
                <img src="02_depth.png" class="report-img" alt="Relative Depth Map">
                <div>
                    <p style="color: var(--text-muted);">
                        Inferred via <strong>Depth Anything V2 Metric Outdoor Model</strong>.
                    </p>
                    <div class="code-box">
Depth Minimum: {depth_info.get('min', 0.0):.2f}<br>
Depth Maximum: {depth_info.get('max', 0.0):.2f}<br>
Depth Mean: {depth_info.get('mean', 0.0):.2f}
                    </div>
                </div>
            </div>
        </div>

        <!-- 3. REFERENCE CALIBRATION -->
        <div class="section">
            <div class="section-title"><i class="fa-solid fa-ruler-vertical"></i> 3. Reference-Assisted Scale Calibration</div>
            <div class="grid-2">
                <img src="03_calibration.png" class="report-img" alt="Calibration Overlay">
                <div>
                    <table>
                        <tr><th>Parameter</th><th>Value</th></tr>
                        <tr><td>Reference Object</td><td>{ref_info.get('name', 'Person')}</td></tr>
                        <tr><td>Reference Real Height</td><td><strong>{ref_info.get('height_m', 1.70):.2f} m</strong></td></tr>
                        <tr><td>Reference Coordinates</td><td>Top {ref_info.get('top')}, Bot {ref_info.get('bottom')}</td></tr>
                        <tr><td>Reference Pixel Height</td><td>{calib_info.get('reference_pixel_height', 0.0):.1f} px</td></tr>
                        <tr><td>Calibrated Scale Factor (S)</td><td><strong>{calib_info.get('scale_factor', 1.0):.4f}</strong></td></tr>
                    </table>
                </div>
            </div>
        </div>

        <!-- 4. TARGET HEIGHT ESTIMATION & EVALUATION -->
        <div class="section">
            <div class="section-title"><i class="fa-solid fa-calculator"></i> 4. Target Height Estimation & Accuracy Benchmark</div>
            <div class="grid-2">
                <img src="04_height_estimation.png" class="report-img" alt="Height Estimation Overlay">
                <div>
                    <div style="background: rgba(56, 189, 248, 0.1); border: 2px solid var(--primary); padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1rem;">
                        <span style="font-size: 0.9rem; color: var(--text-muted); text-transform: uppercase;">Calibrated Estimated Target Height</span>
                        <div style="font-size: 2.8rem; font-weight: 800; color: var(--primary); font-family: 'JetBrains Mono';">{est_h:.2f} meters</div>
                    </div>
                    <table>
                        <tr><td>Target Object</td><td>{tgt_info.get('name', 'Building')}</td></tr>
                        <tr><td>Target Coordinates</td><td>Top {tgt_info.get('top')}, Bot {tgt_info.get('bottom')}</td></tr>
                        <tr><td>Target Pixel Height</td><td>{height_info.get('target_pixel_height', 0.0):.1f} px</td></tr>
                        <tr><td>Known Ground Truth Height</td><td>{known_h:.2f} m</td></tr>
                        <tr><td>Absolute Error</td><td><span style="color: var(--success);">{abs_err:.3f} m</span></td></tr>
                        <tr><td>Relative Percentage Error</td><td><span style="color: var(--success);">{pct_err:.2f} %</span></td></tr>
                    </table>
                </div>
            </div>
        </div>

        <!-- 5. 3D POINT CLOUD -->
        <div class="section">
            <div class="section-title"><i class="fa-solid fa-cube"></i> 5. Metric 3D Point Cloud Reconstruction</div>
            <div class="grid-2">
                <img src="05_pointcloud.png" class="report-img" alt="Point Cloud View">
                <div>
                    <div class="code-box">
Total 3D Points: {pc_info.get('num_points', 0):,} vertices<br>
Spatial Bounds:<br>
X range: {xmin:.2f}m to {xmax:.2f}m<br>
Y range: {ymin:.2f}m to {ymax:.2f}m<br>
Z range: {zmin:.2f}m to {zmax:.2f}m
                    </div>
                    <p style="margin-top: 1rem; color: var(--text-muted); font-size: 0.9rem;">
                        Exported 3D Point Cloud file ready: <code>scene.ply</code>
                    </p>
                </div>
            </div>
        </div>

        <!-- 6. VIRTUAL FLYTHROUGH -->
        <div class="section">
            <div class="section-title"><i class="fa-solid fa-film"></i> 6. 3D Virtual Camera Flythrough Video</div>
            <div style="border-radius: 12px; overflow: hidden; border: 1px solid var(--border-color); background: #000;">
                <video controls autoplay loop muted style="width: 100%; display: block;" poster="06_flythrough_preview.png">
                    <source src="flythrough.mp4" type="video/mp4">
                    Your browser does not support HTML5 video.
                </video>
            </div>
        </div>

        <!-- 7. PIPELINE FLOW -->
        <div class="section">
            <div class="section-title"><i class="fa-solid fa-diagram-project"></i> End-to-End Pipeline Architecture</div>
            <div class="pipeline-flow">
                <div class="flow-step">RGB IMAGE</div>
                <div class="flow-arrow">➔</div>
                <div class="flow-step">DEPTH ESTIMATION</div>
                <div class="flow-arrow">➔</div>
                <div class="flow-step">CALIBRATION</div>
                <div class="flow-arrow">➔</div>
                <div class="flow-step">HEIGHT SOLVER</div>
                <div class="flow-arrow">➔</div>
                <div class="flow-step">3D POINT CLOUD</div>
                <div class="flow-arrow">➔</div>
                <div class="flow-step">FLYTHROUGH</div>
            </div>
        </div>
    </div>
</body>
</html>
"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return report_path
