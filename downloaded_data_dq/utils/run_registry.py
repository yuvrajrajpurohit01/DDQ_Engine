"""
Downloaded Data DQ Engine — Run Registry
downloaded_data_dq/utils/run_registry.py

Maintains:
  reports/runs/run_index.json      — machine-readable run history
  reports/runs/run_registry.html   — interactive HTML viewer (regenerated each run)

Files are named with RUN_ID PREFIX: {RUN_ID}_ddq.log, {RUN_ID}_results.csv, etc.
Links in the registry use paths relative to the reports/runs/ directory.

Public API:
    make_run_id()                → str  (10-digit YYMMDDHHMM)
    register_run(root, meta)     → Path (run_registry.html)
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path


def make_run_id() -> str:
    """Return a 10-digit RUN_ID: YYMMDDHHMM"""
    return datetime.now().strftime("%y%m%d%H%M")


def _load_index(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {"runs": []}


def _save_index(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _rate_col(pct: float) -> str:
    if pct >= 95: return "#22c55e"
    if pct >= 80: return "#eab308"
    return "#ef4444"


def _dur(s: float) -> str:
    s = int(s)
    return f"{s//60}m {s%60}s" if s >= 60 else f"{s}s"


def _status_badge(run: dict) -> str:
    if run.get("hard_fails", 0) > 0:
        return '<span class="badge b-blocked">&#128683; BLOCKED</span>'
    if run.get("crit_fails", 0) > 0:
        return '<span class="badge b-warn">&#9888;&#65039; WARN</span>'
    return '<span class="badge b-ok">&#9989; CLEAN</span>'


def _build_html(runs: list[dict]) -> str:
    """Build the standalone run_registry.html from all run records.

    All file links are relative paths from reports/runs/ back to the files:
      ../../logs/{RUN_ID}_ddq.log
      ../../reports/scorecard/{RUN_ID}_results.csv
      ../../reports/html/{RUN_ID}_dq_command_centre.html
    """
    rows = ""
    for run in reversed(runs):
        rid    = run.get("run_id", "?")
        start  = run.get("start", "")
        dt_str = start[:16].replace("T", "  ") if start else "—"
        dur    = _dur(run.get("elapsed_s", 0))
        syms   = ", ".join(run.get("symbols", []))
        pr     = run.get("pass_rate", 0.0)
        wt     = run.get("wt_score",  0.0)
        p      = run.get("n_pass", 0)
        f      = run.get("n_fail", 0)
        sk     = run.get("n_skip", 0)
        rc     = _rate_col(pr)
        badge  = _status_badge(run)

        lf  = run.get("log_file", "")
        rf  = run.get("results_file", "")
        df  = run.get("dash_file", "")

        # Build relative links from reports/runs/ to each file
        log_link  = f"../../logs/{lf}"      if lf else "#"
        res_link  = f"../../reports/scorecard/{rf}" if rf else "#"
        dash_link = f"../../reports/html/{df}"  if df else "#"

        rows += f"""
    <tr>
      <td class="run-id" onclick="window.open('{dash_link}','_blank')"
          title="Click to open dashboard">{rid}</td>
      <td>{dt_str}</td>
      <td>{dur}</td>
      <td class="syms" title="{syms}">{syms[:40]}{"..." if len(syms)>40 else ""}</td>
      <td style="color:{rc};font-weight:700">{pr}%</td>
      <td style="color:#38bdf8">{wt}%</td>
      <td><span class="ct g">{p}</span>/<span class="ct r">{f}</span>/<span class="ct s">{sk}</span></td>
      <td>{badge}</td>
      <td class="links">
        <a href="{dash_link}" target="_blank" title="Command Centre Dashboard">&#128202;</a>
        <a href="{log_link}"  target="_blank" title="Debug Log">&#128196;</a>
        <a href="{res_link}"  target="_blank" title="Results CSV" download>&#128203;</a>
      </td>
    </tr>"""

    latest   = runs[-1] if runs else {}
    latest_id = latest.get("run_id", "—")
    latest_pr = latest.get("pass_rate", 0)
    latest_rc = _rate_col(latest_pr)
    latest_sym= ", ".join(latest.get("symbols", []))
    total_runs= len(runs)

    # Latest run file links
    lf = latest.get("log_file", "")
    rf = latest.get("results_file", "")
    df = latest.get("dash_file", "")
    lat_dash = f"../../reports/html/{df}" if df else "#"
    lat_log  = f"../../logs/{lf}"         if lf else "#"
    lat_res  = f"../../reports/scorecard/{rf}" if rf else "#"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>DDQ — Run Registry</title>
<style>
:root{{--bg:#0f172a;--surf:#1e293b;--surf2:#263349;--border:#334155;
  --text:#e2e8f0;--muted:#94a3b8;--accent:#38bdf8}}
*{{box-sizing:border-box;margin:0;padding:0}}
body{{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
  background:var(--bg);color:var(--text);font-size:14px;line-height:1.5}}
.hdr{{background:var(--surf);border-bottom:1px solid var(--border);
  padding:16px 28px;display:flex;justify-content:space-between;
  align-items:center;flex-wrap:wrap;gap:12px}}
.hdr-title{{font-size:20px;font-weight:800;color:var(--accent)}}
.hdr-sub{{font-size:12px;color:var(--muted);margin-top:3px}}
.kpis{{display:flex;gap:20px}}
.kpi-val{{font-size:24px;font-weight:900;line-height:1}}
.kpi-lbl{{font-size:10px;font-weight:700;text-transform:uppercase;
  letter-spacing:.8px;color:var(--muted);margin-top:2px;text-align:center}}
.main{{padding:28px}}
.card{{background:var(--surf);border:1px solid var(--border);
  border-radius:12px;padding:20px;margin-bottom:20px}}
.card-title{{font-size:11px;font-weight:700;text-transform:uppercase;
  letter-spacing:.8px;color:var(--muted);margin-bottom:14px}}
.latest{{background:linear-gradient(135deg,var(--surf),var(--surf2));
  border:1px solid var(--accent)44;border-radius:12px;padding:20px;
  margin-bottom:20px;display:flex;justify-content:space-between;
  align-items:center;flex-wrap:wrap;gap:16px}}
.latest-id{{font-size:32px;font-weight:900;color:var(--accent);font-family:monospace}}
.btn{{display:inline-block;border-radius:8px;padding:9px 18px;
  font-size:13px;font-weight:700;text-decoration:none;transition:.15s;white-space:nowrap}}
.btn-accent{{background:var(--accent);color:#0f172a}}
.btn-surf{{background:var(--surf2);color:var(--text);border:1px solid var(--border)}}
.btn-surf:hover,.btn-accent:hover{{filter:brightness(1.1)}}
.toolbar{{display:flex;gap:10px;margin-bottom:12px;align-items:center;flex-wrap:wrap}}
.toolbar input{{background:var(--surf);border:1px solid var(--border);
  border-radius:8px;padding:8px 14px;color:var(--text);font-size:13px;
  width:260px;outline:none}}
.toolbar input:focus{{border-color:var(--accent)}}
.toolbar input::placeholder{{color:var(--muted)}}
.tbl-wrap{{overflow-x:auto;border-radius:10px;border:1px solid var(--border)}}
table{{border-collapse:collapse;width:100%;font-size:13px}}
th{{background:var(--surf2);color:var(--muted);font-size:10px;font-weight:700;
  text-transform:uppercase;letter-spacing:.7px;padding:10px 12px;
  text-align:left;border-bottom:2px solid var(--border);white-space:nowrap;
  cursor:pointer;user-select:none}}
th:hover{{color:var(--text)}}
td{{padding:10px 12px;border-bottom:1px solid var(--border);vertical-align:middle}}
tr:hover td{{background:var(--surf2)}}
tr:last-child td{{border-bottom:none}}
.run-id{{font-family:monospace;font-weight:800;color:var(--accent);cursor:pointer;font-size:14px}}
.run-id:hover{{text-decoration:underline}}
.syms{{max-width:200px;overflow:hidden;text-overflow:ellipsis;
  white-space:nowrap;font-size:12px;color:var(--muted)}}
.links a{{font-size:18px;margin-right:8px;text-decoration:none;opacity:.8;transition:.15s}}
.links a:hover{{opacity:1;transform:scale(1.2)}}
.ct.g{{color:#22c55e;font-weight:700}}
.ct.r{{color:#ef4444;font-weight:700}}
.ct.s{{color:#94a3b8}}
.badge{{display:inline-block;border-radius:5px;padding:2px 8px;font-size:11px;font-weight:700}}
.b-ok{{background:#22c55e22;color:#22c55e;border:1px solid #22c55e44}}
.b-warn{{background:#eab30822;color:#eab308;border:1px solid #eab30844}}
.b-blocked{{background:#ef444422;color:#ef4444;border:1px solid #ef444444}}
.empty{{text-align:center;padding:40px;color:var(--muted)}}
</style>
</head>
<body>
<div class="hdr">
  <div>
    <div class="hdr-title">DDQ — Run Registry</div>
    <div class="hdr-sub">Every run · Click any RUN_ID to open its Command Centre</div>
  </div>
  <div class="kpis">
    <div>
      <div class="kpi-val" style="color:var(--accent)">{total_runs}</div>
      <div class="kpi-lbl">Total Runs</div>
    </div>
    <div>
      <div class="kpi-val" style="color:{latest_rc}">{latest_pr}%</div>
      <div class="kpi-lbl">Latest Pass%</div>
    </div>
    <div>
      <div class="kpi-val" style="color:var(--text);font-family:monospace">{latest_id}</div>
      <div class="kpi-lbl">Latest RUN_ID</div>
    </div>
  </div>
</div>

<div class="main">
{"" if not latest else f'''
  <div class="latest">
    <div>
      <div style="font-size:11px;font-weight:700;text-transform:uppercase;letter-spacing:.8px;color:var(--muted);margin-bottom:6px">Latest Run</div>
      <div class="latest-id">{latest_id}</div>
      <div style="font-size:13px;color:var(--muted);margin-top:6px">
        {latest.get("start","")[:16].replace("T","  ")} &nbsp;·&nbsp; {_dur(latest.get("elapsed_s",0))} &nbsp;·&nbsp; {latest_sym}
      </div>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap">
      <a href="{lat_dash}" target="_blank" class="btn btn-accent">&#128202; Open Dashboard</a>
      <a href="{lat_log}"  target="_blank" class="btn btn-surf">&#128196; Debug Log</a>
      <a href="{lat_res}"  target="_blank" class="btn btn-surf" download>&#128203; Results CSV</a>
    </div>
  </div>
'''}

  <div class="card">
    <div class="card-title">All Runs</div>
    <div class="toolbar">
      <input type="text" id="search" placeholder="Search RUN_ID, symbols..." oninput="filterTable()">
      <span style="font-size:12px;color:var(--muted)" id="countLbl"></span>
    </div>
    <div class="tbl-wrap">
      <table id="runTable">
        <thead><tr>
          <th onclick="sortTable(0)">RUN_ID &#8645;</th>
          <th onclick="sortTable(1)">Date / Time &#8645;</th>
          <th onclick="sortTable(2)">Duration &#8645;</th>
          <th>Symbols</th>
          <th onclick="sortTable(4)">Pass% &#8645;</th>
          <th onclick="sortTable(5)">Wt.Score &#8645;</th>
          <th>P / F / S</th>
          <th>Status</th>
          <th>Open</th>
        </tr></thead>
        <tbody id="runBody">
          {"<tr><td colspan='9' class='empty'>No runs recorded yet</td></tr>" if not rows else rows}
        </tbody>
      </table>
    </div>
  </div>
</div>

<script>
var sortCol=0,sortAsc=false;
function sortTable(col){{
  if(sortCol===col)sortAsc=!sortAsc;else{{sortCol=col;sortAsc=false;}}
  var tbody=document.getElementById("runBody");
  var rows=Array.from(tbody.querySelectorAll("tr"));
  rows.sort(function(a,b){{
    var av=a.cells[col]?a.cells[col].innerText:"";
    var bv=b.cells[col]?b.cells[col].innerText:"";
    var r=av.localeCompare(bv,undefined,{{numeric:true}});
    return sortAsc?r:-r;
  }});
  rows.forEach(function(r){{tbody.appendChild(r);}});
  updateCount();
}}
function filterTable(){{
  var q=document.getElementById("search").value.toLowerCase();
  document.querySelectorAll("#runBody tr").forEach(function(tr){{
    tr.style.display=tr.innerText.toLowerCase().includes(q)?"":"none";
  }});
  updateCount();
}}
function updateCount(){{
  var vis=document.querySelectorAll("#runBody tr:not([style*='none'])").length;
  var tot=document.querySelectorAll("#runBody tr").length;
  document.getElementById("countLbl").textContent=vis===tot?tot+" runs":vis+" of "+tot+" runs";
}}
window.onload=updateCount;
</script>
</body>
</html>"""
    return html


def register_run(project_root: Path, meta: dict) -> Path:
    """
    Append run record to run_index.json and regenerate run_registry.html.

    meta keys:
        run_id, start, stop, elapsed_s, symbols, mode,
        n_pass, n_fail, n_skip, pass_rate, wt_score, hard_fails, crit_fails,
        log_file, results_file, dash_file
    """
    runs_dir   = project_root / "reports" / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)
    index_path = runs_dir / "run_index.json"
    html_path  = runs_dir / "run_registry.html"

    data = _load_index(index_path)

    # Avoid duplicates
    existing = {r.get("run_id") for r in data["runs"]}
    if meta.get("run_id") not in existing:
        data["runs"].append(meta)
        _save_index(index_path, data)

    html_path.write_text(_build_html(data["runs"]), encoding="utf-8")
    return html_path
