// Minimal DOM shim to execute the verdict-panel JS headless and assert it
// builds SVG without throwing. NOT a browser — just enough surface for render.py's JS.
const fs = require('fs');
const path = require('path');

class Node {
  constructor(tag){ this.tag=tag; this.children=[]; this.attrs={}; this.text=null; this.innerHTML_=''; }
  setAttribute(k,v){ this.attrs[k]=v; }
  appendChild(c){ this.children.push(c); return c; }
  insertBefore(c, ref){ this.children.unshift(c); return c; }
  set innerHTML(v){ this.innerHTML_=v; }
  get firstChild(){ return this.children[0]; }
  // count rects/text recursively
  count(tag){ let n=this.tag===tag?1:0; for(const c of this.children) if(c.count) n+=c.count(tag); return n; }
}
const byId = {};
const document = {
  createElementNS(ns, tag){ return new Node(tag); },
  createElement(tag){ return new Node(tag); },
  createTextNode(t){ const n=new Node('#text'); n.text=t; return n; },
  getElementById(id){ if(!byId[id]) byId[id]=new Node('#'+id); return byId[id]; },
};

const html = fs.readFileSync(path.join(__dirname, process.argv[2]||'decode_verdict_T8.html'),'utf8');
// extract the <script> body (the last script block)
const m = html.match(/<script>([\s\S]*?)<\/script>/);
if(!m){ console.error('no script'); process.exit(1); }
let js = m[1];

// run it
try {
  const fn = new Function('document', js + '\nreturn {banners:document.getElementById("banners"),content:document.getElementById("content"),reveal:document.getElementById("reveal")};');
  const r = fn(document);
  const rects = r.content.count('rect') + r.banners.count('rect');
  const texts = r.content.count('text');
  const banners = r.banners.children.length;
  console.log(`OK: produced ${rects} <rect>, ${texts} <text>, ${banners} banners`);
  if(rects < 10){ console.error('too few rects — render likely broke'); process.exit(1); }
  // reveal must contain the ratio
  if(!r.reveal.innerHTML_.includes('×')){ console.error('reveal missing ratio'); process.exit(1); }
  console.log('reveal text length:', r.reveal.innerHTML_.length);
} catch(e){
  console.error('JS THREW:', e.message, '\n', e.stack);
  process.exit(1);
}
