"""Browser tool to inspect and correct the SAM 2 vessel labels.

    .venv-linux/bin/python vessel_ml/label_tool.py        # then open http://localhost:8765

Shows every frame with the current masks, lets you add / remove click prompts,
mark frame ranges (vessel absent, exclude from training, skip entirely), save
to prompts.json and re-run SAM 2 for one video. Standard library + OpenCV only.
"""
import json
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

import cv2
import numpy as np

HERE = Path(__file__).parent
ROOT = HERE.parent
DS = ROOT / "data" / "vessel_dataset"
PROMPTS = HERE / "prompts.json"
PORT = 8765
COLORS = {1: (0, 0, 255), 2: (255, 120, 0)}  # BGR: artery red, vein blue
job = {"proc": None, "vid": None, "log": HERE / "checkpoints" / "label_tool_sam.log"}
lock = threading.Lock()


def load_prompts():
    return json.loads(PROMPTS.read_text()) if PROMPTS.exists() else {}


def video_info(vid):
    files = sorted((DS / "frames" / vid).glob("*.jpg"))
    h, w = cv2.imread(str(files[0]), 0).shape
    areas = []
    csv = DS / "areas" / f"{vid}.csv"
    if csv.exists():
        for line in csv.read_text().splitlines()[1:]:
            areas.append([int(x) for x in line.split(",")])
    cfg = load_prompts().get(vid) or {"stride": 3 if len(files) > 600 else 1, "prompts": []}
    return {"vid": vid, "n": len(files), "w": w, "h": h, "areas": areas, "cfg": cfg}


def render(vid, i, overlay, scale):
    img = cv2.imread(str(DS / "frames" / vid / f"{i:05d}.jpg"))
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    mp = DS / "masks" / vid / f"{i:05d}.png"
    if overlay and mp.exists():
        m = cv2.resize(cv2.imread(str(mp), 0), (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
        for c, col in COLORS.items():
            cn, _ = cv2.findContours((m == c).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img, cn, -1, col, 2)
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def send(self, body, ctype="application/json", code=200):
        if not isinstance(body, bytes):
            body = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        parts = [unquote(p) for p in u.path.strip("/").split("/")]
        q = parse_qs(u.query)
        try:
            if u.path == "/":
                self.send(PAGE.encode(), "text/html; charset=utf-8")
            elif parts[:2] == ["api", "videos"]:
                vids = sorted(p.name for p in (DS / "frames").iterdir() if p.is_dir())
                labelled = {p.name for p in (DS / "masks").iterdir()} if (DS / "masks").exists() else set()
                self.send([{"vid": v, "labelled": v in labelled} for v in vids])
            elif parts[:2] == ["api", "video"]:
                self.send(video_info(parts[2]))
            elif parts[:2] == ["api", "job"]:
                p = job["proc"]
                tail = ""
                if job["log"].exists():
                    tail = job["log"].read_text(errors="replace").replace("\r", "\n").strip().splitlines()[-1:] or [""]
                    tail = tail[0][-160:]
                self.send({"running": p is not None and p.poll() is None, "vid": job["vid"],
                           "code": None if p is None else p.poll(), "tail": tail})
            elif parts[0] == "frame":
                self.send(render(parts[1], int(parts[2]), q.get("overlay", ["1"])[0] == "1",
                                 float(q.get("scale", ["1"])[0])), "image/jpeg")
            else:
                self.send({"error": "not found"}, code=404)
        except Exception as e:  # keep the server alive on a bad request
            self.send({"error": repr(e)}, code=500)

    def do_POST(self):
        parts = [unquote(p) for p in urlparse(self.path).path.strip("/").split("/")]
        body = self.rfile.read(int(self.headers.get("Content-Length", 0)) or 0)
        try:
            if parts[:2] == ["api", "video"]:
                with lock:
                    d = load_prompts()
                    d[parts[2]] = json.loads(body)
                    PROMPTS.write_text(json.dumps(d, indent=1) + "\n")
                self.send({"ok": True})
            elif parts[:2] == ["api", "run"]:
                if job["proc"] is not None and job["proc"].poll() is None:
                    self.send({"error": f"SAM is already running on {job['vid']}"}, code=409)
                    return
                job["vid"] = parts[2]
                job["proc"] = subprocess.Popen([sys.executable, str(HERE / "sam2_label.py"), parts[2]],
                                               stdout=open(job["log"], "w"), stderr=subprocess.STDOUT, cwd=ROOT)
                self.send({"ok": True})
            else:
                self.send({"error": "not found"}, code=404)
        except Exception as e:
            self.send({"error": repr(e)}, code=500)


PAGE = r"""<!doctype html><meta charset="utf-8"><title>Vessel label tool</title>
<style>
 body{font:14px system-ui,sans-serif;margin:0;background:#16181c;color:#e6e6e6;display:flex;height:100vh}
 #left{flex:1;min-width:0;display:flex;flex-direction:column;padding:10px;gap:8px}
 #right{width:340px;padding:10px;overflow:auto;border-left:1px solid #333;background:#1d2026}
 #stage{flex:1;min-height:0;display:flex;align-items:center;justify-content:center;background:#000}
 canvas#view{max-width:100%;max-height:100%;cursor:crosshair}
 canvas#tl{width:100%;height:90px;background:#0d0e10;cursor:pointer}
 button,select,input{font:inherit;background:#2b2f37;color:#e6e6e6;border:1px solid #454a55;border-radius:4px;padding:4px 8px}
 button:hover{background:#3a3f49} button.on{outline:2px solid #fff}
 .art{border-color:#ff5a5a}.vein{border-color:#4f9dff}
 h3{margin:14px 0 6px;font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:#9aa3b2}
 .row{display:flex;gap:6px;align-items:center;flex-wrap:wrap;margin:4px 0}
 .item{display:flex;justify-content:space-between;gap:6px;padding:2px 0;border-bottom:1px solid #2a2d33}
 .item a{color:#8ab4ff;cursor:pointer} small{color:#9aa3b2} #status{color:#ffd166;min-height:1.2em}
 kbd{background:#333;border-radius:3px;padding:0 4px}
</style>
<div id="left">
 <div class="row"><select id="vid"></select>
  <button id="prev">◀</button><button id="play">▶ play</button><button id="next">▶</button>
  <span>frame <input id="fnum" type="number" style="width:70px"> / <span id="n"></span></span>
  <label><input type="checkbox" id="ov" checked> show masks</label>
  <span id="fstate"></span></div>
 <div id="stage"><canvas id="view"></canvas></div>
 <canvas id="tl"></canvas>
 <small>Timeline: red = artery area, blue = vein area (from the last SAM run). White ticks = frames with clicks.
  Grey = skipped, yellow = excluded from training, hatched red/blue = vessel forced absent. Click to jump.</small>
</div>
<div id="right">
 <h3>Clicks on this frame</h3>
 <div class="row"><button id="mA" class="art on">artery <kbd>a</kbd></button><button id="mV" class="vein">vein <kbd>v</kbd></button></div>
 <small>Left click = "this is the vessel". Right click (or shift+click) = "this is NOT it". When artery and vein touch,
  give each a positive click and a negative click on the other one.</small>
 <div id="pts"></div>
 <div class="row"><button id="undo">undo last click <kbd>z</kbd></button><button id="clr">clear clicks on this frame</button></div>
 <h3>Frame ranges</h3>
 <div class="row">from <input id="r0" type="number" style="width:64px"> to <input id="r1" type="number" style="width:64px">
  <button id="s0">start = here</button><button id="s1">end = here</button></div>
 <div class="row"><button data-r="absent:artery" class="art">artery absent</button><button data-r="absent:vein" class="vein">vein absent</button>
  <button data-r="exclude">exclude from training</button><button data-r="skip">skip (bad frames)</button></div>
 <small><b>absent</b>: vessel is collapsed, force an empty mask. <b>exclude</b>: label not trustworthy, keep out of training
  (no SAM re-run needed). <b>skip</b>: frames SAM should not even see (frozen screen, probe off skin).</small>
 <div id="ranges"></div>
 <h3>All click frames</h3><div id="pf"></div>
 <h3>Save / run</h3>
 <div class="row">label every <input id="stride" type="number" min="1" style="width:50px"> frames</div>
 <div class="row"><button id="save">save</button><button id="run">save + re-run SAM on this video</button></div>
 <div id="status"></div>
 <h3>Notes</h3><textarea id="notes" style="width:100%;height:70px;background:#2b2f37;color:#e6e6e6"></textarea>
 <small>Keys: <kbd>←</kbd>/<kbd>→</kbd> step, with <kbd>shift</kbd> ×10, <kbd>space</kbd> play, <kbd>a</kbd>/<kbd>v</kbd> vessel, <kbd>z</kbd> undo click.</small>
</div>
<script>
const $=id=>document.getElementById(id), view=$('view'), ctx=view.getContext('2d'), tl=$('tl');
let V=null, f=0, obj='artery', dirty=false, playing=null, img=new Image(), scale=1;
const api=(u,o)=>fetch(u,o).then(r=>r.json());
function setStatus(s){$('status').textContent=s}
async function init(){
  const vs=await api('/api/videos');
  $('vid').innerHTML=vs.map(v=>`<option value="${v.vid}">${v.vid}${v.labelled?'':'  (no labels yet)'}</option>`).join('');
  $('vid').onchange=()=>{ if(dirty&&!confirmLeave()) return; loadVideo($('vid').value) };
  loadVideo(vs[0].vid); setInterval(pollJob,2000);
}
function confirmLeave(){ setStatus('Unsaved changes were discarded for the previous video.'); return true }
async function loadVideo(v){
  V=await api('/api/video/'+encodeURIComponent(v)); const c=V.cfg; c.prompts=c.prompts||[];
  for(const k of ['absent','exclude','skip']) c[k]=c[k]||[];
  hist=[];
  c.prompts.forEach(p=>{ if(!p.labels) p.labels=p.points.map(_=>1) });
  scale=V.w<450?2:1; view.width=V.w*scale; view.height=V.h*scale;
  $('n').textContent=V.n-1; $('stride').value=c.stride||1; $('notes').value=c.notes||''; dirty=false; go(0); side();
}
function go(i){ f=Math.max(0,Math.min(V.n-1,i)); $('fnum').value=f;
  img.onload=draw; img.src=`/frame/${encodeURIComponent(V.vid)}/${f}?overlay=${$('ov').checked?1:0}&scale=${scale}&t=${Date.now()}`;
  drawTL(); side(); }
const inR=(i,r)=>r.some(x=>(x.from??x[0])<=i&&i<=(x.to??x[1]));
function draw(){ ctx.drawImage(img,0,0);
  for(const p of V.cfg.prompts) if(p.frame===f) p.points.forEach((pt,k)=>{
    const x=pt[0]*scale,y=pt[1]*scale,pos=p.labels[k]===1; ctx.lineWidth=3; ctx.strokeStyle=p.obj==='artery'?'#ff5a5a':'#4f9dff';
    ctx.beginPath(); if(pos){ctx.arc(x,y,7,0,7)} else {ctx.moveTo(x-7,y-7);ctx.lineTo(x+7,y+7);ctx.moveTo(x+7,y-7);ctx.lineTo(x-7,y+7)} ctx.stroke();
    ctx.lineWidth=1; ctx.strokeStyle='#fff'; ctx.beginPath(); ctx.arc(x,y,pos?9:10,0,7); ctx.stroke(); });
  const c=V.cfg, st=[]; const lab=V.areas.find(a=>a[0]===f);
  if(inR(f,c.skip)) st.push('SKIPPED'); else if(!lab) st.push('not labelled (stride / blank frame / not run yet)');
  else st.push(`artery ${lab[1]} px, vein ${lab[2]} px`);
  if(inR(f,c.exclude)) st.push('EXCLUDED from training');
  c.absent.forEach(a=>{ if(a.from<=f&&f<=a.to) st.push(a.obj+' forced absent') });
  $('fstate').textContent=st.join(' · ');
}
function drawTL(){ const w=tl.width=tl.clientWidth, h=tl.height=90, g=tl.getContext('2d'), X=i=>i/(V.n-1)*w, c=V.cfg;
  const band=(a,b,col,y0,y1)=>{g.fillStyle=col; g.fillRect(X(a),y0,Math.max(2,X(b)-X(a)),y1-y0)};
  c.skip.forEach(r=>band(r[0],r[1],'#555',0,h)); c.exclude.forEach(r=>band(r[0],r[1],'rgba(255,209,102,.35)',0,h));
  c.absent.forEach(a=>band(a.from,a.to,a.obj==='artery'?'rgba(255,90,90,.45)':'rgba(79,157,255,.45)',a.obj==='artery'?h-12:h-6,a.obj==='artery'?h-6:h));
  const mx=Math.max(1,...V.areas.map(a=>Math.max(a[1],a[2])));
  [[1,'#ff5a5a'],[2,'#4f9dff']].forEach(([k,col])=>{ g.strokeStyle=col; g.beginPath();
    V.areas.forEach((a,j)=>{const x=X(a[0]),y=h-14-(a[k]/mx)*(h-22); j?g.lineTo(x,y):g.moveTo(x,y)}); g.stroke() });
  g.fillStyle='#fff'; new Set(c.prompts.map(p=>p.frame)).forEach(i=>g.fillRect(X(i)-1,0,2,8));
  g.fillStyle='#ffd166'; g.fillRect(X(f)-1,0,2,h);
}
function side(){ const c=V.cfg;
  const here=c.prompts.filter(p=>p.frame===f);
  $('pts').innerHTML=here.length?here.map(p=>p.points.map((pt,k)=>`<div class="item"><span>${p.obj} ${p.labels[k]?'+ is vessel':'− not vessel'} (${pt.map(Math.round)})</span><a onclick="delPt('${p.obj}',${k})">remove</a></div>`).join('')).join(''):'<small>none</small>';
  const rs=[]; c.absent.forEach((a,i)=>rs.push([`${a.obj} absent ${a.from}–${a.to}`,'absent',i,a.from]));
  c.exclude.forEach((r,i)=>rs.push([`exclude ${r[0]}–${r[1]}`,'exclude',i,r[0]])); c.skip.forEach((r,i)=>rs.push([`skip ${r[0]}–${r[1]}`,'skip',i,r[0]]));
  $('ranges').innerHTML=rs.map(r=>`<div class="item"><a onclick="go(${r[3]})">${r[0]}</a><a onclick="delR('${r[1]}',${r[2]})">remove</a></div>`).join('')||'<small>none</small>';
  const fr=[...new Set(c.prompts.map(p=>p.frame))].sort((a,b)=>a-b);
  $('pf').innerHTML=fr.map(i=>`<a style="color:#8ab4ff;cursor:pointer;margin-right:8px" onclick="go(${i})">${i}</a>`).join('')||'<small>none</small>';
}
function changed(){ V.cfg.prompts=V.cfg.prompts.filter(p=>p.points.length); dirty=true; draw(); drawTL(); side(); setStatus('unsaved changes') }
function delPt(o,k){ const p=V.cfg.prompts.find(p=>p.frame===f&&p.obj===o); if(!p) return; p.points.splice(k,1); p.labels.splice(k,1); changed() }
let hist=[];
function undo(){ const h=hist.pop(); if(!h){setStatus('nothing to undo');return} const p=V.cfg.prompts.find(p=>p.frame===h.frame&&p.obj===h.obj);
  if(p){ p.points.pop(); p.labels.pop() } if(h.frame!==f) go(h.frame); changed() }
function delR(k,i){ V.cfg[k].splice(i,1); dirty=true; drawTL(); side(); draw() }
function click(e,neg){ const r=view.getBoundingClientRect(); const x=(e.clientX-r.left)/r.width*V.w, y=(e.clientY-r.top)/r.height*V.h;
  let p=V.cfg.prompts.find(p=>p.frame===f&&p.obj===obj); if(!p){p={frame:f,obj,points:[],labels:[]}; V.cfg.prompts.push(p)}
  p.points.push([Math.round(x),Math.round(y)]); p.labels.push(neg?0:1); hist.push({frame:f,obj}); dirty=true; draw(); drawTL(); side(); setStatus('unsaved changes') }
view.onclick=e=>click(e,e.shiftKey); view.oncontextmenu=e=>{e.preventDefault(); click(e,true)};
$('clr').onclick=()=>{ V.cfg.prompts=V.cfg.prompts.filter(p=>p.frame!==f); dirty=true; draw(); drawTL(); side(); setStatus('unsaved changes') };
function mode(o){ obj=o; $('mA').classList.toggle('on',o==='artery'); $('mV').classList.toggle('on',o==='vein') }
$('undo').onclick=undo; $('mA').onclick=()=>mode('artery'); $('mV').onclick=()=>mode('vein');
$('s0').onclick=()=>$('r0').value=f; $('s1').onclick=()=>$('r1').value=f;
document.querySelectorAll('[data-r]').forEach(b=>b.onclick=()=>{ const a=+$('r0').value,z=+$('r1').value;
  if($('r0').value===''||$('r1').value===''||z<a){setStatus('set a valid from / to range first');return}
  const [k,o]=b.dataset.r.split(':'); if(k==='absent') V.cfg.absent.push({obj:o,from:a,to:z}); else V.cfg[k].push([a,z]);
  dirty=true; drawTL(); side(); draw(); setStatus('unsaved changes') });
async function save(){ const c=JSON.parse(JSON.stringify(V.cfg)); c.stride=Math.max(1,+$('stride').value||1); c.notes=$('notes').value;
  for(const k of ['absent','exclude','skip']) if(!c[k].length) delete c[k];
  c.prompts=c.prompts.filter(p=>p.points.length).sort((a,b)=>a.frame-b.frame||a.obj.localeCompare(b.obj)); c.human_checked=true;
  await api('/api/video/'+encodeURIComponent(V.vid),{method:'POST',body:JSON.stringify(c)}); dirty=false; setStatus('saved to prompts.json') }
$('save').onclick=save;
$('run').onclick=async()=>{ if(!V.cfg.prompts.length){setStatus('add at least one click first');return}
  await save(); const r=await api('/api/run/'+encodeURIComponent(V.vid),{method:'POST'}); setStatus(r.error||'SAM started…') };
let wasRunning=false;
async function pollJob(){ const j=await api('/api/job'); if(j.running){ wasRunning=true; setStatus(`SAM running on ${j.vid}: ${j.tail}`) }
  else if(wasRunning){ wasRunning=false; setStatus(j.code===0?'SAM finished, masks reloaded':'SAM failed: '+j.tail);
    if(j.vid===V.vid){ const keep=f; V=Object.assign(await api('/api/video/'+encodeURIComponent(V.vid)),{}); for(const k of ['absent','exclude','skip']) V.cfg[k]=V.cfg[k]||[]; V.cfg.prompts.forEach(p=>{if(!p.labels)p.labels=p.points.map(_=>1)}); go(keep) } } }
$('prev').onclick=()=>go(f-1); $('next').onclick=()=>go(f+1); $('fnum').onchange=()=>go(+$('fnum').value); $('ov').onchange=()=>go(f);
function step(){ const labelled=V.areas.map(a=>a[0]); const nx=labelled.find(i=>i>f); return nx===undefined?f+1:nx }
$('play').onclick=()=>{ if(playing){clearInterval(playing);playing=null;$('play').textContent='▶ play'} else {playing=setInterval(()=>{ if(f>=V.n-1){$('play').click()} else go(step()) },120);$('play').textContent='⏸ pause'} };
tl.onclick=e=>{ const r=tl.getBoundingClientRect(); go(Math.round((e.clientX-r.left)/r.width*(V.n-1))) };
document.onkeydown=e=>{ if(['INPUT','TEXTAREA','SELECT'].includes(e.target.tagName)) return;
  if(e.key==='ArrowRight') go(f+(e.shiftKey?10:1)); else if(e.key==='ArrowLeft') go(f-(e.shiftKey?10:1));
  else if(e.key==='z'||e.key==='Backspace') undo(); else if(e.key==='a') mode('artery'); else if(e.key==='v') mode('vein'); else if(e.key===' '){e.preventDefault();$('play').click()} };
window.onresize=()=>V&&drawTL(); window.onbeforeunload=()=>dirty?'unsaved':undefined; init();
</script>"""


if __name__ == "__main__":
    print(f"Label tool on http://localhost:{PORT}  (Ctrl+C to stop)")
    ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
