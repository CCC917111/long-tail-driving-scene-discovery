#!/usr/bin/env python3
"""Neuron explorer: which frames fire a long-tail unit, and which units a frame fires.

The sparse autoencoder turns every frame into a handful of active units, and
the units of the long-tail subspace are where the method's explanations live.
This tool makes that inspectable in both directions:

* unit -> frames:  give a unit number, see the frames that activate it most,
* frame -> units:  give a sample token, see its six camera views and every
                   long-tail unit it activates, strongest first.

It reads the ``activations.npz`` written by the SAE training scripts (every
labelled frame, with labels and splits) or by ``scripts/screen.py`` (your own,
unlabelled frames), and the camera images under ``--samples-root``. It needs
numpy and Pillow only.

Commands
--------
serve    Local web interface with both views, linked to each other
         (http://127.0.0.1:8765 by default).
unit     Print the top frames of one unit; ``--save`` writes a contact sheet.
sample   Print the active units of one frame; ``--save`` writes its six views.
units    Print the units ranked by how many frames they fire on (and by
         long-tail purity when the run has labels).

Examples
--------
    python scripts/neuron_explorer.py serve \\
        --run output/sae_abstopk_tail_reward \\
        --samples-root /data/nuscenes/samples \\
        --glossary results/neuron_glossary.csv

    python scripts/neuron_explorer.py unit 3058 --run output/sae_abstopk_tail_reward \\
        --samples-root /data/nuscenes/samples --save unit3058.jpg

    python scripts/neuron_explorer.py sample <sample_token> \\
        --run output/sae_abstopk_tail_reward --samples-root /data/nuscenes/samples

Images are located through the ``images`` entry that scripts/extract.py writes
into meta.json. For features extracted before that entry existed, pass
``--nuscenes-meta`` (the directory holding sample_data.json) and the paths are
rebuilt from the nuScenes tables.
"""

from __future__ import annotations

import argparse
import io
import json
import sys
import threading
from collections import OrderedDict
from functools import partial
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tail_activations import (
    CAMERAS,
    DEFAULT_ETA,
    DEFAULT_NEURON_THRESHOLD,
    ActivationStore,
    load_glossary,
)

# Row-major layout of the six views, as a driver would see them.
SIX_VIEW_LAYOUT = (
    ("CAM_FRONT_LEFT", "CAM_FRONT", "CAM_FRONT_RIGHT"),
    ("CAM_BACK_LEFT", "CAM_BACK", "CAM_BACK_RIGHT"),
)


# ---------------------------------------------------------------------------
#  Image lookup
# ---------------------------------------------------------------------------

def images_from_nuscenes(meta_dir: Path) -> dict[str, dict[str, str]]:
    """``sample_token -> {camera: 'CAM_X/file.jpg'}`` from nuScenes sample_data.json."""
    path = Path(meta_dir) / "sample_data.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; --nuscenes-meta must be the directory "
                                f"of the nuScenes tables (e.g. v1.0-trainval)")
    with path.open("r", encoding="utf-8") as f:
        records = json.load(f)
    out: dict[str, dict[str, str]] = {}
    for rec in records:
        if not rec.get("is_key_frame"):
            continue
        parts = rec.get("filename", "").split("/")
        if len(parts) == 3 and parts[0] == "samples" and parts[1] in CAMERAS:
            out.setdefault(rec["sample_token"], {})[parts[1]] = f"{parts[1]}/{parts[2]}"
    return out


# ---------------------------------------------------------------------------
#  Queries
# ---------------------------------------------------------------------------

class Explorer:
    """Both lookup directions over one activation store."""

    def __init__(self, store: ActivationStore, samples_root: Path | None = None,
                 glossary: dict[int, str] | None = None, eta: float = DEFAULT_ETA,
                 neuron_threshold: float = DEFAULT_NEURON_THRESHOLD,
                 extra_images: dict[str, dict[str, str]] | None = None):
        self.store = store
        self.samples_root = Path(samples_root).resolve() if samples_root else None
        self.glossary = glossary or {}
        self.eta = eta
        self.neuron_threshold = neuron_threshold
        self.extra_images = extra_images or {}
        self._rows = np.repeat(np.arange(len(store)), np.diff(store.indptr))
        self._counts = store.counts(eta)

    # -- unit overview ------------------------------------------------------

    def units(self, top: int = 50) -> list[dict]:
        """Units ranked by how many frames they fire on above the neuron threshold.

        Named units come first. When the run has labels, the long-tail purity of
        each unit (share of its frames labelled long-tail) is reported too.
        """
        strong = np.abs(self.store.values) > self.neuron_threshold
        units = self.store.indices[strong]
        n_frames = np.bincount(units, minlength=self.store.tail_dim)
        n_tail = None
        if self.store.labels is not None:
            tail_rows = self.store.labels[self._rows[strong]]
            n_tail = np.bincount(units, weights=tail_rows, minlength=self.store.tail_dim)

        order = sorted(np.nonzero(n_frames)[0],
                       key=lambda j: (j not in self.glossary, -n_frames[j], j))
        named = [j for j in self.glossary if 0 <= j < self.store.tail_dim and n_frames[j] == 0]
        out = []
        for j in list(order) + named:
            entry = {"unit": int(j), "name": self.glossary.get(int(j), ""),
                     "frames": int(n_frames[j])}
            if n_tail is not None:
                entry["long_tail_frames"] = int(n_tail[j])
                entry["purity"] = float(n_tail[j] / n_frames[j]) if n_frames[j] else None
            out.append(entry)
        named_first = [e for e in out if e["name"]]
        rest = [e for e in out if not e["name"]]
        return named_first + rest[:max(top - len(named_first), 0)]

    def unit_stats(self, unit: int) -> dict:
        """Frames a unit fires on and, with labels, the share labelled long-tail."""
        rows, values = self.store.column(unit)
        rows = rows[np.abs(values) > self.neuron_threshold]
        stats = {"unit": unit, "name": self.glossary.get(unit, ""), "frames": int(len(rows))}
        if self.store.labels is not None:
            n_tail = int(self.store.labels[rows].sum())
            stats["long_tail_frames"] = n_tail
            stats["purity"] = n_tail / len(rows) if len(rows) else None
        return stats

    # -- unit -> frames -----------------------------------------------------

    def frames_of_unit(self, unit: int, top: int = 24,
                       min_activation: float | None = None) -> list[dict]:
        """The frames with the largest ``|z_t|`` on ``unit``, strongest first."""
        floor = self.neuron_threshold if min_activation is None else min_activation
        rows, values = self.store.column(unit)
        keep = np.abs(values) > floor
        rows, values = rows[keep], values[keep]
        order = np.argsort(-np.abs(values), kind="stable")[:top]
        return [self._frame(int(rows[i]), activation=float(values[i])) for i in order]

    # -- frame -> units -----------------------------------------------------

    def units_of_frame(self, token: str, top: int = 30) -> dict:
        """Every active long-tail unit of one frame, strongest first."""
        if token not in self.store.row_of:
            raise KeyError(f"sample token {token!r} is not in {self.store.path}")
        i = self.store.row_of[token]
        cols, vals = self.store.row(i)
        active = np.abs(vals) > self.eta
        cols, vals = cols[active], vals[active]
        order = np.argsort(-np.abs(vals), kind="stable")
        units = [{"unit": int(cols[k]), "name": self.glossary.get(int(cols[k]), ""),
                  "activation": float(vals[k]),
                  "strong": bool(abs(vals[k]) > self.neuron_threshold)}
                 for k in order[:top]]
        frame = self._frame(i)
        frame.update({"units": units, "total_active_units": int(active.sum())})
        return frame

    def search(self, query: str, limit: int = 20) -> list[dict]:
        """Frames whose sample token starts with, or whose scene contains, ``query``."""
        q = query.strip().lower()
        if not q:
            return []
        hits = []
        for i, s in enumerate(self.store.samples):
            if s["sample_token"].lower().startswith(q) or q in s.get("scene", "").lower():
                hits.append(self._frame(i))
                if len(hits) >= limit:
                    break
        return hits

    # -- helpers ------------------------------------------------------------

    def _frame(self, i: int, activation: float | None = None) -> dict:
        s = self.store.samples[i]
        cols, vals = self.store.row(i)
        frame = {
            "sample_token": s["sample_token"],
            "scene": s.get("scene", ""),
            "active_tail_units": int(self._counts[i]),
            "prediction": "long_tail" if self._counts[i] >= 1 else "normal",
            "tail_score": float(np.linalg.norm(vals)),
            "cameras": [c for c in CAMERAS if self.image_path(s["sample_token"], c)],
        }
        if activation is not None:
            frame["activation"] = activation
        if self.store.labels is not None:
            frame["label"] = "long_tail" if self.store.labels[i] == 1 else "normal"
        if self.store.splits is not None:
            frame["split"] = self.store.splits[i]
        return frame

    def image_path(self, token: str, camera: str) -> Path | None:
        """The image of one view, or None; never resolves outside --samples-root."""
        if self.samples_root is None or camera not in CAMERAS:
            return None
        i = self.store.row_of.get(token)
        rel = None
        if i is not None:
            rel = self.store.samples[i].get("images", {}).get(camera)
        rel = rel or self.extra_images.get(token, {}).get(camera)
        if not rel:
            return None
        path = (self.samples_root / rel).resolve()
        if self.samples_root not in path.parents or not path.is_file():
            return None
        return path


# ---------------------------------------------------------------------------
#  Rendering (contact sheets for the command line)
# ---------------------------------------------------------------------------

def _font(size: int):
    for name in ("DejaVuSans.ttf", "Arial.ttf", "Helvetica.ttc"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _thumb(path: Path | None, width: int) -> Image.Image:
    height = round(width * 9 / 16)
    if path is None:
        tile = Image.new("RGB", (width, height), (235, 238, 243))
        ImageDraw.Draw(tile).text((10, height // 2 - 8), "image not found",
                                  fill=(120, 128, 140), font=_font(14))
        return tile
    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((width, height))
        tile = Image.new("RGB", (width, height), (0, 0, 0))
        tile.paste(im, ((width - im.width) // 2, (height - im.height) // 2))
        return tile


def render_grid(tiles: list[tuple[Image.Image, str]], columns: int, title: str) -> Image.Image:
    """Tiles with a caption under each and a title bar on top."""
    if not tiles:
        raise ValueError("nothing to render")
    w, h = tiles[0][0].size
    caption_h, title_h, pad = 26, 44, 10
    rows = -(-len(tiles) // columns)
    sheet = Image.new("RGB", (columns * (w + pad) + pad,
                              title_h + rows * (h + caption_h + pad) + pad), (255, 255, 255))
    draw = ImageDraw.Draw(sheet)
    draw.rectangle([0, 0, sheet.width, title_h - 8], fill=(28, 52, 120))
    draw.text((pad + 4, 9), title, fill=(255, 255, 255), font=_font(18))
    for k, (tile, caption) in enumerate(tiles):
        r, c = divmod(k, columns)
        x = pad + c * (w + pad)
        y = title_h + r * (h + caption_h + pad)
        sheet.paste(tile, (x, y))
        draw.text((x + 2, y + h + 4), caption, fill=(60, 66, 76), font=_font(13))
    return sheet


def unit_sheet(explorer: Explorer, unit: int, frames: list[dict], camera: str,
               width: int = 320, columns: int = 4) -> Image.Image:
    name = explorer.glossary.get(unit, "")
    title = f"Unit {unit}" + (f" - {name}" if name else "") + f"  ({camera})"
    tiles = [(_thumb(explorer.image_path(f["sample_token"], camera), width),
              f"{f['scene']}  |z|={abs(f['activation']):.2f}") for f in frames]
    return render_grid(tiles, columns, title)


def frame_sheet(explorer: Explorer, frame: dict, width: int = 400) -> Image.Image:
    named = [u for u in frame["units"] if u["strong"]][:4]
    reasons = ", ".join(f"{u['unit']}" + (f" {u['name']}" if u["name"] else "")
                        for u in named)
    title = f"{frame['scene']}  {frame['sample_token'][:12]}  -  {frame['prediction']}"
    if reasons:
        title += f"  -  {reasons}"
    tiles = [(_thumb(explorer.image_path(frame["sample_token"], cam), width), cam)
             for row in SIX_VIEW_LAYOUT for cam in row]
    return render_grid(tiles, 3, title)


# ---------------------------------------------------------------------------
#  Web interface
# ---------------------------------------------------------------------------

PAGE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Neuron Explorer</title>
<style>
:root{--navy:#1c3478;--ink:#1f2933;--muted:#667085;--rule:#e4e7ec;--panel:#f6f8fb;
--tail:#c2410c;--tail-bg:#fff1e8;--ok:#1f7a4d;--ok-bg:#e9f6ef;}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;color:var(--ink);background:#fff}
header{background:var(--navy);color:#fff;padding:12px 20px;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
header h1{font-size:17px;margin:0;font-weight:650}header .meta{opacity:.8;font-size:12.5px}
header form{margin-left:auto;display:flex;gap:8px}
input,select,button{font:inherit;border-radius:6px;border:1px solid #c8d0dc;padding:6px 9px}
header input{width:230px;border:0}header button{background:#fff;color:var(--navy);border:0;font-weight:600;cursor:pointer}
.wrap{display:grid;grid-template-columns:300px 1fr;min-height:calc(100vh - 54px)}
aside{border-right:1px solid var(--rule);background:var(--panel);overflow:auto;max-height:calc(100vh - 54px);position:sticky;top:0}
aside h2{font-size:12px;letter-spacing:.06em;text-transform:uppercase;color:var(--muted);margin:16px 16px 8px}
.unit-row{display:block;padding:8px 16px;border-bottom:1px solid var(--rule);color:inherit;text-decoration:none}
.unit-row:hover,.unit-row.on{background:#e9eef8}.unit-row b{font-variant-numeric:tabular-nums}
.unit-row .nm{color:var(--navy);font-weight:600}.unit-row small{display:block;color:var(--muted)}
main{padding:20px 26px 40px;min-width:0}
main h2{font-size:21px;margin:0 0 4px}.sub{color:var(--muted);margin:0 0 16px}
.bar{display:flex;gap:10px;align-items:center;margin:0 0 16px;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(230px,1fr));gap:14px}
.card{border:1px solid var(--rule);border-radius:8px;overflow:hidden;background:#fff;cursor:pointer;text-decoration:none;color:inherit}
.card:hover{border-color:#9fb0d0;box-shadow:0 2px 8px rgba(28,52,120,.12)}
.card img,.ph{width:100%;aspect-ratio:16/9;object-fit:cover;display:block;background:#e9edf3}
.ph{display:flex;align-items:center;justify-content:center;color:var(--muted);font-size:12px}
.card .cap{padding:7px 9px;font-size:12.5px;display:flex;justify-content:space-between;gap:6px}
.chip{display:inline-block;font-size:11px;padding:1px 7px;border-radius:10px;font-weight:600}
.chip.long_tail{background:var(--tail-bg);color:var(--tail)}.chip.normal{background:var(--ok-bg);color:var(--ok)}
.views{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:6px 0 20px}
.views figure{margin:0}.views figcaption{font-size:11.5px;color:var(--muted);padding:3px 1px}
.views img,.views .ph{width:100%;aspect-ratio:16/9;object-fit:cover;border-radius:6px;display:block}
table{border-collapse:collapse;width:100%;max-width:760px}td,th{text-align:left;padding:6px 8px;border-bottom:1px solid var(--rule)}
th{font-size:12px;color:var(--muted);font-weight:600}td.num{font-variant-numeric:tabular-nums;width:90px}
.act{height:8px;border-radius:4px;background:var(--navy);display:inline-block;vertical-align:middle}
.act.weak{background:#aab6d3}a.u{color:var(--navy);font-weight:600;text-decoration:none}
.empty{color:var(--muted);padding:30px 0}
@media (max-width:820px){.wrap{grid-template-columns:1fr}aside{position:static;max-height:260px}.views{grid-template-columns:repeat(2,1fr)}header form{margin-left:0;width:100%}header input{flex:1}}
</style></head><body>
<header><h1>Neuron Explorer</h1><span class="meta" id="meta"></span>
<form id="find"><input id="q" placeholder="unit number, sample token or scene" autocomplete="off"><button>Go</button></form></header>
<div class="wrap"><aside><h2>Long-tail units</h2><div id="units"></div></aside><main id="main"></main></div>
<script>
const $=s=>document.querySelector(s), esc=s=>String(s).replace(/[&<>"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c]));
let S={};
const api=p=>fetch(p).then(r=>r.ok?r.json():r.json().then(e=>Promise.reject(e.error||r.statusText)));
function chip(p){return p?`<span class="chip ${p}">${p.replace('_','-')}</span>`:''}
function img(tok,cam,w){return `<img loading="lazy" src="img/${encodeURIComponent(tok)}/${cam}?w=${w||420}" onerror="this.outerHTML='<div class=ph>no image</div>'">`}
async function boot(){S=await api('api/summary');
 $('#meta').textContent=`${S.frames.toLocaleString()} frames · ${S.tail_dim} long-tail units · eta ${S.eta} · unit threshold ${S.neuron_threshold}`;
 const us=await api('api/units?top=80');
 $('#units').innerHTML=us.map(u=>`<a class="unit-row" data-u="${u.unit}" href="#/unit/${u.unit}"><b>#${u.unit}</b> ${u.name?`<span class="nm">${esc(u.name)}</span>`:''}<small>${u.frames} frame${u.frames==1?'':'s'}${u.purity!=null?` · ${(100*u.purity).toFixed(0)}% long-tail`:''}</small></a>`).join('')||'<p class="empty" style="padding:16px">No unit fires above the threshold.</p>';
 route();}
async function showUnit(u,cam){cam=cam||'CAM_FRONT';
 document.querySelectorAll('.unit-row').forEach(a=>a.classList.toggle('on',a.dataset.u==u));
 $('#main').innerHTML='<p class="sub">Loading…</p>';
 try{const d=await api(`api/unit/${u}?top=48`);
  $('#main').innerHTML=`<h2>Unit ${u}${d.name?' — '+esc(d.name):''}</h2>
  <p class="sub">Fires on ${d.n_frames} frames with |z<sub>t</sub>| &gt; ${S.neuron_threshold}${d.purity!=null?` · ${(100*d.purity).toFixed(1)}% of them labelled long-tail`:''} · showing the ${d.frames.length} strongest</p>
  <div class="bar">Camera <select id="cam">${S.cameras.map(c=>`<option ${c==cam?'selected':''}>${c}</option>`).join('')}</select></div>
  <div class="grid">${d.frames.map(f=>`<a class="card" href="#/sample/${encodeURIComponent(f.sample_token)}">${img(f.sample_token,cam,420)}
   <div class="cap"><span>${esc(f.scene||f.sample_token.slice(0,10))}</span><span>|z| ${Math.abs(f.activation).toFixed(2)} ${chip(f.label)}</span></div></a>`).join('')||'<p class="empty">No frame activates this unit above the threshold.</p>'}</div>`;
  $('#cam').onchange=e=>showUnit(u,e.target.value);
 }catch(e){$('#main').innerHTML=`<p class="empty">${esc(e)}</p>`}}
async function showSample(tok){
 $('#main').innerHTML='<p class="sub">Loading…</p>';
 try{const f=await api(`api/sample/${encodeURIComponent(tok)}`), mx=Math.max(1e-9,...f.units.map(u=>Math.abs(u.activation)));
  const views=[['CAM_FRONT_LEFT','CAM_FRONT','CAM_FRONT_RIGHT'],['CAM_BACK_LEFT','CAM_BACK','CAM_BACK_RIGHT']].flat();
  $('#main').innerHTML=`<h2>${esc(f.scene||'Frame')} ${chip(f.prediction)}</h2>
  <p class="sub">${esc(f.sample_token)} · ${f.total_active_units} active long-tail units (|z| &gt; ${S.eta}) · ||z<sub>t</sub>|| = ${f.tail_score.toFixed(2)}${f.label?` · label: ${f.label.replace('_','-')}`:''}${f.split?` · ${f.split} split`:''}</p>
  <div class="views">${views.map(c=>`<figure>${img(f.sample_token,c,560)}<figcaption>${c}</figcaption></figure>`).join('')}</div>
  <table><tr><th>Unit</th><th>Feature</th><th>|z<sub>t</sub>|</th><th></th></tr>${f.units.map(u=>`<tr><td><a class="u" href="#/unit/${u.unit}">#${u.unit}</a></td><td>${esc(u.name||'')}</td><td class="num">${Math.abs(u.activation).toFixed(3)}</td><td><span class="act ${u.strong?'':'weak'}" style="width:${Math.max(2,220*Math.abs(u.activation)/mx)}px"></span></td></tr>`).join('')||'<tr><td colspan=4 class="empty">No long-tail unit is active: this frame is predicted normal.</td></tr>'}</table>`;
 }catch(e){$('#main').innerHTML=`<p class="empty">${esc(e)}</p>`}}
async function showSearch(q){const hits=await api(`api/search?q=${encodeURIComponent(q)}`);
 $('#main').innerHTML=`<h2>Search</h2><p class="sub">${hits.length} frames match “${esc(q)}”</p><div class="grid">${hits.map(f=>`<a class="card" href="#/sample/${encodeURIComponent(f.sample_token)}">${img(f.sample_token,'CAM_FRONT',420)}<div class="cap"><span>${esc(f.scene||f.sample_token.slice(0,10))}</span>${chip(f.prediction)}</div></a>`).join('')}</div>`}
function route(){const h=decodeURIComponent(location.hash.slice(2)).split('/');
 if(h[0]==='unit')showUnit(+h[1]);else if(h[0]==='sample')showSample(h.slice(1).join('/'));else if(h[0]==='search')showSearch(h.slice(1).join('/'));
 else{const first=document.querySelector('.unit-row');if(first)location.hash='#/unit/'+first.dataset.u;else $('#main').innerHTML='<p class="empty">Search for a frame to begin.</p>'}}
$('#find').onsubmit=e=>{e.preventDefault();const q=$('#q').value.trim();if(!q)return;
 location.hash=/^\d+$/.test(q)?'#/unit/'+q:'#/search/'+encodeURIComponent(q)};
addEventListener('hashchange',route);boot();
</script></body></html>"""


class _Handler(BaseHTTPRequestHandler):
    server_version = "NeuronExplorer/1.0"

    def __init__(self, *args, explorer: Explorer, cache: "OrderedDict", lock, **kwargs):
        self.explorer = explorer
        self.cache = cache
        self.lock = lock
        super().__init__(*args, **kwargs)

    def log_message(self, fmt, *args):  # keep the console quiet
        pass

    def _send(self, body: bytes, ctype: str, status: HTTPStatus = HTTPStatus.OK) -> None:
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store" if ctype.startswith("application/json")
                         else "max-age=3600")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, payload, status: HTTPStatus = HTTPStatus.OK) -> None:
        self._send(json.dumps(payload).encode("utf-8"), "application/json; charset=utf-8",
                   status)

    def do_GET(self):  # noqa: N802 (http.server naming)
        url = urlparse(self.path)
        parts = [unquote(p) for p in url.path.strip("/").split("/") if p]
        query = {k: v[-1] for k, v in parse_qs(url.query).items()}
        ex = self.explorer
        try:
            if not parts:
                return self._send(PAGE.encode("utf-8"), "text/html; charset=utf-8")
            if parts == ["api", "summary"]:
                return self._json({"frames": len(ex.store), "tail_dim": ex.store.tail_dim,
                                   "eta": ex.eta, "neuron_threshold": ex.neuron_threshold,
                                   "has_labels": ex.store.labels is not None,
                                   "cameras": list(CAMERAS)})
            if parts == ["api", "units"]:
                return self._json(ex.units(top=int(query.get("top", 50))))
            if len(parts) == 3 and parts[:2] == ["api", "unit"]:
                unit = int(parts[2])
                frames = ex.frames_of_unit(unit, top=min(int(query.get("top", 24)), 200))
                stats = ex.unit_stats(unit)
                return self._json({**stats, "n_frames": stats["frames"], "frames": frames})
            if len(parts) >= 3 and parts[:2] == ["api", "sample"]:
                return self._json(ex.units_of_frame("/".join(parts[2:])))
            if parts == ["api", "search"]:
                return self._json(ex.search(query.get("q", "")))
            if len(parts) == 3 and parts[0] == "img":
                return self._image(parts[1], parts[2], int(query.get("w", 420)))
        except (KeyError, IndexError, ValueError) as err:
            return self._json({"error": str(err).strip("'\"")}, HTTPStatus.NOT_FOUND)
        return self._json({"error": "not found"}, HTTPStatus.NOT_FOUND)

    def _image(self, token: str, camera: str, width: int) -> None:
        width = max(64, min(width, 1600))
        key = (token, camera, width)
        with self.lock:
            body = self.cache.get(key)
            if body is not None:
                self.cache.move_to_end(key)
        if body is None:
            path = self.explorer.image_path(token, camera)
            if path is None:
                return self._json({"error": "image not found"}, HTTPStatus.NOT_FOUND)
            with Image.open(path) as im:
                im = im.convert("RGB")
                im.thumbnail((width, width))
                buf = io.BytesIO()
                im.save(buf, "JPEG", quality=85)
            body = buf.getvalue()
            with self.lock:
                self.cache[key] = body
                while len(self.cache) > 600:
                    self.cache.popitem(last=False)
        self._send(body, "image/jpeg")


def make_server(explorer: Explorer, host: str = "127.0.0.1", port: int = 8765):
    handler = partial(_Handler, explorer=explorer, cache=OrderedDict(), lock=threading.Lock())
    return ThreadingHTTPServer((host, port), handler)


# ---------------------------------------------------------------------------
#  Command line
# ---------------------------------------------------------------------------

def build_explorer(args) -> Explorer:
    store = ActivationStore(args.run)
    extra = images_from_nuscenes(args.nuscenes_meta) if args.nuscenes_meta else None
    glossary = load_glossary(args.glossary) if args.glossary else {}
    eta = args.eta
    if eta is None:
        metrics = Path(args.run) / "metrics.json"
        eta = DEFAULT_ETA
        if metrics.exists():
            with metrics.open("r", encoding="utf-8") as f:
                eta = json.load(f).get("decision", {}).get("eta", DEFAULT_ETA)
    return Explorer(store, args.samples_root, glossary, eta=eta,
                    neuron_threshold=args.neuron_threshold, extra_images=extra)


def parse_args(argv=None) -> argparse.Namespace:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--run", type=Path, required=True,
                        help="Directory with activations.npz (an SAE run or a screening run)")
    common.add_argument("--samples-root", type=Path, default=None,
                        help="nuScenes samples/ directory holding CAM_FRONT/, CAM_BACK/, ...")
    common.add_argument("--nuscenes-meta", type=Path, default=None,
                        help="nuScenes table directory (sample_data.json), only needed when "
                             "meta.json carries no image paths")
    common.add_argument("--glossary", type=Path, default=None,
                        help="CSV naming units, e.g. results/neuron_glossary.csv")
    common.add_argument("--eta", type=float, default=None,
                        help="Activity threshold of the decision rule (default: the run's)")
    common.add_argument("--neuron-threshold", type=float, default=DEFAULT_NEURON_THRESHOLD,
                        help="|z_t| above which a unit counts as firing on a frame")

    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    s = sub.add_parser("serve", parents=[common], help="Local web interface")
    s.add_argument("--host", default="127.0.0.1")
    s.add_argument("--port", type=int, default=8765)
    u = sub.add_parser("unit", parents=[common], help="Top frames of one unit")
    u.add_argument("unit", type=int)
    u.add_argument("--top", type=int, default=12)
    u.add_argument("--camera", default="CAM_FRONT", choices=CAMERAS)
    u.add_argument("--save", type=Path, default=None, help="Write a contact sheet (.jpg/.png)")
    f = sub.add_parser("sample", parents=[common], help="Active units of one frame")
    f.add_argument("token")
    f.add_argument("--top", type=int, default=15)
    f.add_argument("--save", type=Path, default=None, help="Write the six views (.jpg/.png)")
    o = sub.add_parser("units", parents=[common], help="Units ranked by frames fired on")
    o.add_argument("--top", type=int, default=30)
    return p.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    explorer = build_explorer(args)

    if args.command == "serve":
        if args.samples_root is None:
            print("note: no --samples-root given, frames will be listed without images",
                  file=sys.stderr)
        server = make_server(explorer, args.host, args.port)
        print(f"Neuron explorer on http://{args.host}:{args.port}  "
              f"({len(explorer.store)} frames, {explorer.store.tail_dim} long-tail units). "
              f"Ctrl+C to stop.")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()

    elif args.command == "unit":
        frames = explorer.frames_of_unit(args.unit, top=args.top)
        name = explorer.glossary.get(args.unit, "")
        print(f"unit {args.unit}" + (f" ({name})" if name else "") +
              f": {len(frames)} frames with |z_t| > {explorer.neuron_threshold}")
        for rank, f in enumerate(frames, start=1):
            label = f" {f['label']}" if "label" in f else ""
            print(f"  {rank:3d}  |z|={abs(f['activation']):6.3f}  {f['scene']:<12} "
                  f"{f['sample_token']}{label}")
        if args.save and frames:
            unit_sheet(explorer, args.unit, frames, args.camera).save(args.save)
            print(f"contact sheet: {args.save}")

    elif args.command == "sample":
        frame = explorer.units_of_frame(args.token, top=args.top)
        print(f"{frame['sample_token']}  {frame['scene']}  -> {frame['prediction']} "
              f"({frame['total_active_units']} active long-tail units, "
              f"||z_t|| = {frame['tail_score']:.3f})")
        for u in frame["units"]:
            mark = "*" if u["strong"] else " "
            print(f"  {mark} unit {u['unit']:5d}  |z|={abs(u['activation']):6.3f}  {u['name']}")
        if args.save:
            frame_sheet(explorer, frame).save(args.save)
            print(f"six views: {args.save}")

    elif args.command == "units":
        for u in explorer.units(top=args.top):
            purity = f"  purity {100 * u['purity']:5.1f}%" if u.get("purity") is not None else ""
            print(f"  unit {u['unit']:5d}  {u['frames']:6d} frame"
                  f"{'' if u['frames'] == 1 else 's'}{purity}  {u['name']}")


if __name__ == "__main__":
    main()
