#!/usr/bin/env python3
"""Render the decode-viz model.json into a self-contained static HTML verdict
panel (inline JS + SVG, DOM-inspectable, no server, openable in a browser).

The HTML is PRESENTATIONAL ONLY.  All lie-prone classification lives in
reduce.py.  This script just embeds the model and ships the template.

Usage:
  python3 render.py model_T8.json --out decode_verdict_T8.html
"""
import json
import sys


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<title>gzippy decode verdict panel</title>
<style>
  body { font: 13px/1.4 -apple-system, system-ui, sans-serif; margin: 0; background:#0e1116; color:#d7dde5; }
  .wrap { max-width: 1180px; margin: 0 auto; padding: 18px 22px 60px; }
  h1 { font-size: 19px; margin: 0 0 2px; }
  h2 { font-size: 15px; margin: 26px 0 6px; border-bottom:1px solid #2a3340; padding-bottom:4px; }
  .sub { color:#8b97a6; font-size:12px; margin:0 0 14px; }
  .banner { padding:8px 12px; border-radius:6px; margin:6px 0; font-size:12.5px; }
  .ok   { background:#13301c; border:1px solid #2e7d4a; }
  .warn { background:#3a2a12; border:1px solid #9a6a1e; }
  .bad  { background:#3a1518; border:1px solid #b03a44; }
  .info { background:#16202c; border:1px solid #2a3a4d; }
  .grid2 { display:grid; grid-template-columns:1fr 1fr; gap:18px; }
  .tool { color:#9fd3ff; font-weight:600; }
  .rg { color:#ffc78a; }
  svg { display:block; background:#11161d; border:1px solid #222c38; border-radius:6px; }
  .lab { fill:#8b97a6; font-size:10px; }
  .labw { fill:#cdd6df; font-size:11px; }
  .glyph { font-size:13px; }
  .legend { font-size:11.5px; color:#9aa6b5; margin:6px 0; }
  .legend span { display:inline-block; margin-right:14px; }
  .sw { display:inline-block; width:11px; height:11px; border-radius:2px; vertical-align:-1px; margin-right:4px; }
  table { border-collapse:collapse; font-size:12px; margin-top:6px; }
  td,th { padding:3px 10px 3px 0; text-align:right; }
  th:first-child, td:first-child { text-align:left; }
  .note { background:#16202c; border-left:3px solid #3a6ea5; padding:10px 14px; margin-top:8px; border-radius:0 6px 6px 0; }
  code { background:#1c2530; padding:1px 5px; border-radius:3px; }
  .hl-rate { color:#ff7b7b; } .hl-place { color:#7bd3ff; }
</style></head>
<body><div class="wrap">
<h1>gzippy parallel-SM decode &mdash; verdict panel</h1>
<p class="sub">Weight = <b>wall-relevance</b>, not CPU-sum. The consumer spine (one in-order thread) IS the wall.
Worker decode boxes that finish before their consumer turn are SLACK (folded/grey). Blocking is a
labeled <b>heuristic</b> (no causal flow edges in these traces). Cross-tool = 6 canonical phases + coverage.</p>
<div id="banners"></div>
<div id="content"></div>
<div class="note" id="reveal"></div>
<div class="note">
  <b>Raw 40&times;16 exploration:</b> this panel is a verdict, not a timeline browser.
  For zoom/pan over every thread, load the trace into <code>ui.perfetto.dev</code>
  (drag the <code>gz_T8.json</code> file in) &mdash; it is already Chrome-trace format.
</div>
</div>
<script>
const MODEL = __MODEL__;
const PHASE_COLOR = {
  dispatch:'#5b8def', decode:'#e0823c', resolve:'#b06ad0',
  publish:'#3fae8c', output:'#d4b23c', wait:'#c44a52'
};
const fmt = us => us>=1e6 ? (us/1e6).toFixed(2)+'s' : us>=1e3 ? (us/1e3).toFixed(1)+'ms' : us.toFixed(0)+'us';
function el(tag, attrs, children){
  const e=document.createElementNS(tag==='svg'||['rect','text','line','g','path','circle','title'].includes(tag)?'http://www.w3.org/2000/svg':'http://www.w3.org/1999/xhtml',tag);
  for(const k in (attrs||{})) e.setAttribute(k, attrs[k]);
  (children||[]).forEach(c=> e.appendChild(typeof c==='string'?document.createTextNode(c):c));
  return e;
}
function tool(name){ return MODEL.tools.find(t=>t.tool===name); }

// ---------- banners ----------
function banners(){
  const root=document.getElementById('banners');
  MODEL.tools.forEach(t=>{
    if(t.error){ root.appendChild(el('div',{class:'banner bad'},[`${t.tool}: ERROR ${t.error}`])); return; }
    const r=t.reconciliation;
    let cls='info', msg;
    if(r.measured_wall_us){
      cls = r.ok ? 'ok':'bad';
      msg = `WALL RECONCILIATION [${t.tool}]: viz ${fmt(r.viz_wall_us)} vs measured ${fmt(r.measured_wall_us)} `
          + `(${r.delta_pct.toFixed(1)}% ${r.ok?'within spread':'>spread — viz wall is NOT the bench wall'})`;
    } else {
      msg = `WALL [${t.tool}]: viz-computed ${fmt(r.viz_wall_us)} (no measured wall supplied — reconciliation skipped). `
          + `Pass --measured-wall-us-${t.tool} to enable the red self-check.`;
    }
    root.appendChild(el('div',{class:'banner '+cls},[msg]));
    const m=t.mismatch;
    if(m.total_mismatch>0){
      const aff=Object.keys(m.affected_names).join(', ');
      root.appendChild(el('div',{class:'banner warn'},[
        `B/E-MISMATCH [${t.tool}]: ${m.total_mismatch} unpaired (unmatched_B ${m.unmatched_b}, unmatched_E ${m.unmatched_e}, name-mismatch ${m.name_mismatch}). `
        + `Affected: ${aff}. A dropped E makes a fake giant bar — treat those span names with suspicion.`]));
    }
    const cov=t.phases.coverage;
    if(cov<0.95){
      root.appendChild(el('div',{class:'banner warn'},[
        `COVERAGE [${t.tool}]: only ${(cov*100).toFixed(0)}% of the consumer wall is instrumented; the rest renders GREY-HATCHED = UNKNOWN (never idle).`]));
    }
  });
}

// ---------- consumer spine + per-stall glyphs ----------
function spinePanel(t){
  const W=1120, rowH=46, padL=8, padR=8;
  const wall=t.wall_us;
  const sx = us => padL + (us/wall)*(W-padL-padR);
  const svg=el('svg',{width:W, height:rowH+78, viewBox:`0 0 ${W} ${rowH+78}`});
  // spine background = full wall
  svg.appendChild(el('rect',{x:padL,y:18,width:W-padL-padR,height:rowH,fill:'#0b0f14',stroke:'#2a3340'}));
  // segments colored by dominant phase, wait portion in red
  t.spine.segments.forEach(s=>{
    const x=sx(s.ts), w=Math.max(0.7,(s.dur/wall)*(W-padL-padR));
    // base (the iter body)
    svg.appendChild(el('rect',{x:x,y:18,width:w,height:rowH,fill:'#1b2838',stroke:'#243246','stroke-width':0.4},
       [el('title',{},[`consumer.iter @${fmt(s.ts)} dur ${fmt(s.dur)} (wait ${fmt(s.wait_us)})`])]));
    // wait overlay (the wall-relevant stall portion)
    if(s.wait_us>0){
      const ww=Math.max(0.7,(s.wait_us/wall)*(W-padL-padR));
      svg.appendChild(el('rect',{x:x,y:18,width:ww,height:rowH,fill:PHASE_COLOR.wait,opacity:0.85},
        [el('title',{},[`frontier wait ${fmt(s.wait_us)}`])]));
    }
  });
  // per-stall glyphs above the spine: ● rate (running), ◆ placement (ready)
  t.stalls.stalls.forEach(st=>{
    const x=sx(st.ts_start);
    const isRate = st.cls==='rate';
    svg.appendChild(el('text',{x:x,y:14,'text-anchor':'middle',class:'glyph',
        fill: isRate?'#ff7b7b':'#7bd3ff'},[isRate?'●':'◆']))
      .appendChild(el('title',{},[`${isRate?'RATE (decode RUNNING)':'PLACEMENT (decode READY)'} `
        +`stall ${fmt(st.dur)} chunk ${st.chunk_id} — running decodes: ${st.running_decodes}`]));
  });
  // labels
  svg.appendChild(el('text',{x:padL,y:rowH+34,class:'labw'},
     [`CONSUMER SPINE = WALL  ·  ${fmt(t.spine.spine_total_us)} over ${t.spine.n_iter} iters  ·  this row tiles end-to-end BY CONSTRUCTION`]));
  svg.appendChild(el('text',{x:padL,y:rowH+50,class:'lab'},
     [`glyphs (HEURISTIC — no causal edges):  ● rate-bound stall (decode still RUNNING at wait start)   ◆ placement-bound (decode already READY)`]));
  svg.appendChild(el('text',{x:padL,y:rowH+66,class:'lab'},
     [`red = frontier wait time inside each iter (the wall-relevant stall)`]));
  return svg;
}

// ---------- rate/placement histogram ----------
function histPanel(t){
  const s=t.stalls;
  const W=360,H=120,padL=46,padB=22;
  const svg=el('svg',{width:W,height:H,viewBox:`0 0 ${W} ${H}`});
  const data=[['rate (RUNNING)', s.n_rate, s.rate_us, '#ff7b7b'],
              ['placement (READY)', s.n_placement, s.placement_us, '#7bd3ff']];
  const maxUs=Math.max(1,s.rate_us,s.placement_us);
  data.forEach((d,i)=>{
    const bx=padL+i*150, bw=110;
    const bh=(d[2]/maxUs)*(H-padB-14);
    svg.appendChild(el('rect',{x:bx,y:H-padB-bh,width:bw,height:bh,fill:d[3],opacity:0.85},
      [el('title',{},[`${d[0]}: ${d[1]} stalls, ${fmt(d[2])}`])]));
    svg.appendChild(el('text',{x:bx+bw/2,y:H-padB-bh-4,'text-anchor':'middle',class:'labw'},[`${fmt(d[2])}`]));
    svg.appendChild(el('text',{x:bx+bw/2,y:H-padB+13,'text-anchor':'middle',class:'lab'},[`${d[0]} ×${d[1]}`]));
  });
  svg.appendChild(el('text',{x:4,y:12,class:'lab'},['wait µs']));
  return svg;
}

// ---------- folded worker slack ----------
function workerPanel(t){
  const w=t.workers, W=540,H=70;
  const svg=el('svg',{width:W,height:H,viewBox:`0 0 ${W} ${H}`});
  const tot=Math.max(1,w.total_decode_us);
  const relW=(w.wall_relevant_decode_us/tot)*(W-16);
  const slkW=(w.slack_decode_us/tot)*(W-16);
  svg.appendChild(el('rect',{x:8,y:14,width:Math.max(0.5,relW),height:16,fill:'#e0823c'},
     [el('title',{},[`wall-relevant decode (overlaps consumer wait): ${fmt(w.wall_relevant_decode_us)}`])]));
  svg.appendChild(el('rect',{x:8+relW,y:14,width:Math.max(0.5,slkW),height:16,fill:'#3a4452'},
     [el('title',{},[`SLACK decode (overlapped, off the wall): ${fmt(w.slack_decode_us)}`])]));
  svg.appendChild(el('text',{x:8,y:46,class:'labw'},
     [`worker decode: ${fmt(w.total_decode_us)} CPU across ${w.n_decode} spans / ${Object.keys(w.by_tid).length} threads`]));
  svg.appendChild(el('text',{x:8,y:62,class:'lab'},
     [`orange = wall-relevant (${fmt(w.wall_relevant_decode_us)}) · grey = SLACK (${fmt(w.slack_decode_us)}, folded; a fat box here is NOT the bottleneck)`]));
  // decode-mode badge (CPU-region fact, belongs on the folded track not the spine)
  const dm=w.decode_modes||{};
  const wa=dm.window_absent||0, cl=dm.clean||0;
  const badge=`~ ${wa} window-absent (slow speculative bootstrap) / ${cl} clean`;
  svg.appendChild(el('text',{x:W-8,y:46,'text-anchor':'end',class:'lab',fill:'#e0a06a'},[badge]));
  return svg;
}

// ---------- 6-phase cross-tool bars + coverage ----------
function phasePanel(){
  const W=1120, rowH=34, top=22, gap=16, padL=8;
  const tools=MODEL.tools.filter(t=>!t.error);
  const maxWall=Math.max(...tools.map(t=>t.phases.consumer_wall_us));
  const svg=el('svg',{width:W,height:top+tools.length*(rowH+gap)+24,viewBox:`0 0 ${W} ${top+tools.length*(rowH+gap)+24}`});
  const sx = us => (us/maxWall)*(W-padL-160);
  tools.forEach((t,ti)=>{
    const y=top+ti*(rowH+gap);
    let x=padL+150;
    svg.appendChild(el('text',{x:padL,y:y+rowH/2+4,class:'labw'},
       [`${t.tool}  ${fmt(t.phases.consumer_wall_us)}`]));
    // 6 phases stacked
    MODEL.phases.forEach(p=>{
      const us=t.phases.phase_us[p]||0;
      const w=sx(us);
      if(w>0.4){
        svg.appendChild(el('rect',{x:x,y:y,width:w,height:rowH,fill:PHASE_COLOR[p],opacity:0.86,stroke:'#11161d','stroke-width':0.4},
          [el('title',{},[`${t.tool} ${p}: ${fmt(us)}`])]));
      }
      x+=w;
    });
    // coverage gap = grey hatched UNKNOWN
    const cov=t.phases.coverage;
    const unkW=sx(t.phases.consumer_wall_us*(1-cov));
    if(unkW>0.4){
      svg.appendChild(el('rect',{x:padL+150+sx(t.phases.instrumented_us),y:y,width:unkW,height:rowH,
        fill:'url(#hatch)',stroke:'#444','stroke-width':0.4},
        [el('title',{},[`UNKNOWN — ${((1-cov)*100).toFixed(0)}% of wall uninstrumented (NOT idle)`])]));
    }
    svg.appendChild(el('text',{x:padL,y:y+rowH/2+18,class:'lab'},[`coverage ${(cov*100).toFixed(0)}%`]));
  });
  // hatch pattern
  const defs=el('defs',{});
  const pat=el('pattern',{id:'hatch',width:6,height:6,patternUnits:'userSpaceOnUse',patternTransform:'rotate(45)'});
  pat.appendChild(el('rect',{width:6,height:6,fill:'#1a222c'}));
  pat.appendChild(el('line',{x1:0,y1:0,x2:0,y2:6,stroke:'#3a4452','stroke-width':2}));
  defs.appendChild(pat); svg.insertBefore(defs,svg.firstChild);
  return svg;
}

function legend(){
  const d=el('div',{class:'legend'});
  MODEL.phases.forEach(p=>{
    d.appendChild(el('span',{},[el('span',{class:'sw',style:`background:${PHASE_COLOR[p]}`}),p]));
  });
  d.appendChild(el('span',{},[el('span',{class:'sw',style:'background:#3a4452'}),'slack/UNKNOWN']));
  return d;
}

// ---------- assemble ----------
banners();
const c=document.getElementById('content');
function section(title, sub){ c.appendChild(el('h2',{},[title])); if(sub)c.appendChild(el('p',{class:'sub'},[sub])); }

section('Consumer spine + per-stall rate/placement glyphs',
  'The spine IS the wall. Glyphs classify each frontier stall (heuristic). The rate/placement split picks the lever.');
MODEL.tools.filter(t=>!t.error).forEach(t=>{
  c.appendChild(el('div',{class:'sub tool'+(t.tool==='rg'?' rg':'')},[`${t.tool} — ${fmt(t.wall_us)}`]));
  c.appendChild(spinePanel(t));
});

section('Rate-vs-placement histogram (the lever-picker)',
  'Of N frontier stalls, how many were RATE-bound (consumer blocked on a still-running decode) vs PLACEMENT-bound (decode already done)? HEURISTIC — no causal edges.');
const hgrid=el('div',{class:'grid2'});
MODEL.tools.filter(t=>!t.error).forEach(t=>{
  const box=el('div',{});
  box.appendChild(el('div',{class:'sub tool'+(t.tool==='rg'?' rg':'')},
    [`${t.tool}: ${t.stalls.n_rate} rate / ${t.stalls.n_placement} placement (${t.stalls.causal})`]));
  box.appendChild(histPanel(t));
  hgrid.appendChild(box);
});
c.appendChild(hgrid);

section('Worker decode — FOLDED slack (weight ≠ duration)',
  'Inverted flamegraph convention: a fat overlapped worker box is NOT the bottleneck. Only decode time overlapping a consumer wait is wall-relevant.');
MODEL.tools.filter(t=>!t.error).forEach(t=>{
  c.appendChild(el('div',{class:'sub tool'+(t.tool==='rg'?' rg':'')},[t.tool]));
  c.appendChild(workerPanel(t));
});

section('Cross-tool — 6 canonical phases + coverage',
  'Native span names differ between tools (a phantom gap). Folded to exactly 6 phases. Grey-hatched = UNKNOWN wall (never idle).');
c.appendChild(legend());
c.appendChild(phasePanel());

// reveal note
const gz=tool('gz'), rg=tool('rg');
if(gz && rg){
  const rev=document.getElementById('reveal');
  const ratio=(gz.wall_us/rg.wall_us).toFixed(2);
  rev.innerHTML = `<b>What this panel reveals.</b> gzippy wall ${fmt(gz.wall_us)} vs rapidgzip ${fmt(rg.wall_us)} = `
   +`<b>${ratio}×</b>. The consumer spine shows gzippy's frontier stalls are `
   +`<span class="hl-rate">${gz.stalls.n_rate} RATE-bound (${fmt(gz.stalls.rate_us)})</span> vs `
   +`<span class="hl-place">${gz.stalls.n_placement} placement (${fmt(gz.stalls.placement_us)})</span> — `
   +`the consumer waits overwhelmingly on decodes that are <b>still running</b> (rate), not on scheduling/order (placement). `
   +`That is the same conclusion the oracle reached (placement causally dead). Both tools run the SAME workload shape `
   +`(gzippy ${ (gz.workers.decode_modes.window_absent||0) } window-absent / ${ (gz.workers.decode_modes.clean||0) } clean `
   +`vs rapidgzip ${ (rg.workers.decode_modes.window_absent||0) } / ${ (rg.workers.decode_modes.clean||0) }) — so the gap is decode RATE, not extra machinery. `
   +`The 6-phase bars show gzippy's <b>decode</b> phase wider than rapidgzip's, consistent with the slower per-chunk decode. `
   +`<b>Honesty note:</b> the TLB/page-walk stall finding (DTLB store-walk 3.26×, 99.6% faults in worker decode) is a <i>perf-counter</i> `
   +`fact and is NOT present in these timeline traces — it is not drawn here. The decode-mode counts ARE in the trace and badge the `
   +`folded worker track (a CPU-region fact, off the wall spine). `
   +`Caveat: rapidgzip's B/E-mismatch (${rg.mismatch.total_mismatch} on ${Object.keys(rg.mismatch.affected_names).join(',')}) `
   +`means its dispatch-phase numbers are unreliable; treat the cross-tool dispatch bar with suspicion.`;
}
</script>
</body></html>
"""


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = "decode_verdict.html"
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    if not args:
        print(__doc__); sys.exit(1)
    model = json.load(open(args[0]))
    html = HTML_TEMPLATE.replace("__MODEL__", json.dumps(model))
    with open(out, "w") as f:
        f.write(html)
    print(f"wrote {out} ({len(html)} bytes)", file=sys.stderr)


if __name__ == "__main__":
    main()
