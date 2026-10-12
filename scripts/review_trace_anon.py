#!/usr/bin/env python3
"""Private side-by-side review of `nym traces anon` output.

Shows only changed string fields (changed lines plus context), highlighting
originals and their replacements using the per-session key files. Served as a
read-only capability-gated snapshot (loopback by default, `--tailnet` opt-in);
never serves a directory. Run through uv: no third-party dependencies.
"""
import argparse
import json
import secrets
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from serve_terms_review import Server, allowed_bind, tailnet_address  # noqa: E402

MAX_FIELD_CHARS = 6000
CONTEXT_LINES = 1


def leaves(value, path=''):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from leaves(item, f'{path}.{key}' if path else key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from leaves(item, f'{path}[{index}]')
    elif isinstance(value, str):
        yield path, value


def excerpt(original, anonymized):
    """Changed lines with context; long unchanged stretches collapse to a marker."""
    a, b = original.split('\n'), anonymized.split('\n')
    if len(a) != len(b):
        return [[original[:MAX_FIELD_CHARS], anonymized[:MAX_FIELD_CHARS]]]
    changed = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    keep = sorted({j for i in changed for j in range(i - CONTEXT_LINES, i + CONTEXT_LINES + 1) if 0 <= j < len(a)})
    rows, last, size = [], None, 0
    for i in keep:
        if last is not None and i != last + 1:
            rows.append(None)
        rows.append([a[i][:1500], b[i][:1500]])
        size += len(a[i])
        last = i
        if size > MAX_FIELD_CHARS:
            rows.append(None)
            break
    return rows


def load_keys(path):
    mapping = []
    if path and path.exists():
        for line in path.read_text(encoding='utf-8').splitlines():
            row = json.loads(line)
            if 'original' in row:
                mapping.append([row['original'], row['replacement'], row.get('pattern_name', '')])
    return mapping


def build(original_dir, anonymized_dir, keys_dir):
    sessions = []
    for anon_path in sorted(anonymized_dir.rglob('*.jsonl')):
        rel = anon_path.relative_to(anonymized_dir)
        source = original_dir / rel
        if not source.is_file():
            continue
        keys = load_keys(keys_dir / rel.with_suffix('.keys.jsonl') if keys_dir else None)
        changes, images = [], 0
        a_lines = source.read_text(encoding='utf-8').splitlines()
        b_lines = anon_path.read_text(encoding='utf-8').splitlines()
        for number, (a, b) in enumerate(zip(a_lines, b_lines), 1):
            if a == b or not a.strip():
                continue
            a_leaves = dict(leaves(json.loads(a)))
            for path, value in leaves(json.loads(b)):
                before = a_leaves.get(path)
                if before is None or before == value:
                    continue
                if path.endswith('.data') and value == '':
                    images += 1
                    continue
                changes.append({'line': number, 'path': path, 'rows': excerpt(before, value)})
        counts = {}
        for _, _, kind in keys:
            counts[kind] = counts.get(kind, 0) + 1
        sessions.append({'name': str(rel), 'changes': changes, 'keys': keys, 'images_removed': images,
                         'classes': counts})
    return sessions


PAGE = '''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>NYM trace anonymization review</title>
<style>
*{box-sizing:border-box}body{margin:0;font:14px/1.45 -apple-system,system-ui,sans-serif;background:#111;color:#e6e6e6}
header{position:sticky;top:0;background:#181818;border-bottom:1px solid #333;padding:8px 12px;display:flex;gap:8px;flex-wrap:wrap;align-items:center;z-index:2}
header h1{font-size:15px;margin:0 8px 0 0}button{font:inherit;background:#2b3a2f;color:inherit;border:1px solid #5a7a62;border-radius:6px;padding:6px 10px;min-height:36px;cursor:pointer}button:disabled{opacity:.5;cursor:default}select,input{font:inherit;background:#222;color:inherit;border:1px solid #555;border-radius:6px;padding:6px 8px;min-height:36px}
#summary{color:#aaa;font-size:13px}main{padding:10px 12px 60px;max-width:1500px;margin:auto}
.change{border:1px solid #333;border-radius:8px;margin:10px 0;overflow:hidden}.meta{background:#1d1d1d;padding:5px 10px;color:#aaa;font-size:12px;overflow-wrap:anywhere}
.grid{display:grid;grid-template-columns:minmax(0,1fr) minmax(0,1fr)}.change{min-width:0}.grid>div{padding:4px 10px;white-space:pre-wrap;overflow-wrap:anywhere;font:12.5px/1.45 ui-monospace,Menlo,monospace;border-top:1px solid #262626}
.grid>.a{border-right:1px solid #333;background:#151515}.gap{grid-column:1/-1;color:#666;text-align:center;font-size:11px;padding:2px}
.hdr{font:600 11px system-ui;color:#888;text-transform:uppercase;letter-spacing:.04em}
mark{border-radius:3px;padding:0 1px;cursor:pointer}mark.o{background:#5a2a2a;color:#ffd7d7}mark.r{background:#244d35;color:#d6ffe4}
#tip{position:fixed;left:8px;right:8px;bottom:8px;background:#222;border:1px solid #666;border-radius:8px;padding:10px;display:none;z-index:3;overflow-wrap:anywhere}
.classes{display:flex;flex-wrap:wrap;gap:6px;margin:6px 0 2px}.chip{background:#262626;border-radius:12px;padding:2px 8px;font-size:12px;color:#bbb}
@media(max-width:760px){.grid{grid-template-columns:minmax(0,1fr)}.grid>.a{border-right:0;border-left:3px solid #8a3d3d}.grid>.b{border-left:3px solid #2f7a4d}.hdr{display:none}.gap{border-top:1px dashed #444}}
</style></head><body>
<header><h1>NYM trace review</h1><select id="session" aria-label="Session"></select>
<select id="cls" aria-label="Class"><option value="">All classes</option></select>
<input id="q" placeholder="Filter text" aria-label="Filter"><button id="allowdl" type="button" disabled>Allowlist (0) ⤓</button><span id="summary"></span></header>
<main><div class="classes" id="classes"></div><div id="list"></div></main><div id="tip" role="status"><span id="tiptext"></span> <button id="allow" type="button">Never replace</button></div>
<script type="application/json" id="data">__DATA__</script>
<script>
const sessions=JSON.parse(document.getElementById('data').textContent);const $=id=>document.getElementById(id);
sessions.forEach((s,i)=>{const o=document.createElement('option');o.value=i;o.textContent=`${s.name} · ${s.changes.length} fields`;$('session').append(o)});
function esc(s){return s.replace(/[.*+?^${}()|[\\]\\\\]/g,'\\\\$&')}
function matcher(list){if(!list.length)return null;const sorted=[...list].sort((a,b)=>b.length-a.length);return new RegExp(sorted.map(esc).join('|'),'g')}
function paint(text,re,map,cls,wanted){const frag=document.createDocumentFragment();if(!re){frag.append(text);return frag}let last=0;re.lastIndex=0;let m;
 while((m=re.exec(text))){if(m[0]==='')break;const info=map.get(m[0]);if(wanted&&info&&info[2]!==wanted){continue}
  frag.append(text.slice(last,m.index));const mk=document.createElement('mark');mk.className=cls;mk.textContent=m[0];mk.dataset.k=m[0];mk.dataset.side=cls;frag.append(mk);last=m.index+m[0].length}
 frag.append(text.slice(last));return frag}
function render(){const s=sessions[+$('session').value||0];const wanted=$('cls').value;const q=$('q').value.toLowerCase();
 const byOrig=new Map(s.keys.map(k=>[k[0],k])),byRep=new Map(s.keys.map(k=>[k[1],k]));
 const reO=matcher(s.keys.filter(k=>k[0].length>1).map(k=>k[0])),reR=matcher(s.keys.filter(k=>k[1].length>1).map(k=>k[1]));
 $('classes').replaceChildren(...Object.entries(s.classes).sort((a,b)=>b[1]-a[1]).map(([k,v])=>{const c=document.createElement('span');c.className='chip';c.textContent=`${k} ${v}`;return c}));
 const cur=$('cls').value;$('cls').replaceChildren(Object.assign(document.createElement('option'),{value:'',textContent:'All classes'}),...Object.keys(s.classes).sort().map(k=>Object.assign(document.createElement('option'),{value:k,textContent:k})));$('cls').value=s.classes[cur]?cur:'';
 const list=$('list');list.replaceChildren();let shown=0;
 for(const c of s.changes){const text=c.rows.filter(Boolean).map(r=>r[0]+'\\n'+r[1]).join('\\n').toLowerCase();if(q&&!text.includes(q))continue;
  if(wanted&&!s.keys.some(k=>k[2]===wanted&&c.rows.some(r=>r&&(r[0].includes(k[0])||r[1].includes(k[1])))))continue;shown++;
  const box=document.createElement('div');box.className='change';const meta=document.createElement('div');meta.className='meta';meta.textContent=`record ${c.line} · ${c.path}`;box.append(meta);
  const grid=document.createElement('div');grid.className='grid';const h1=document.createElement('div');h1.className='hdr a';h1.textContent='Original';const h2=document.createElement('div');h2.className='hdr b';h2.textContent='Anonymized';grid.append(h1,h2);
  for(const r of c.rows){if(!r){const g=document.createElement('div');g.className='gap';g.textContent='⋯';grid.append(g);continue}
   const a=document.createElement('div');a.className='a';a.append(paint(r[0],reO,byOrig,'o',wanted));const b=document.createElement('div');b.className='b';b.append(paint(r[1],reR,byRep,'r',wanted));grid.append(a,b)}
  box.append(grid);list.append(box)}
 $('summary').textContent=`${shown}/${s.changes.length} changed fields · ${s.keys.length} mappings · ${s.images_removed} images removed`;
 list.onclick=e=>{const m=e.target.closest('mark');if(!m){$('tip').style.display='none';return}const k=(m.dataset.side==='o'?byOrig:byRep).get(m.dataset.k);if(!k)return;
  $('tiptext').textContent=`${k[2]}:  ${k[0]}  →  ${k[1]}`;$('allow').dataset.term=k[0];$('allow').textContent=allowed.has(k[0])?'In allowlist':'Never replace';$('tip').style.display='block'}}
let saved=[];try{saved=JSON.parse(sessionStorage.getItem('nym-trace-allow')||'[]')}catch{}const allowed=new Set(saved);
function syncAllow(){$('allowdl').textContent=`Allowlist (${allowed.size}) ⤓`;$('allowdl').disabled=!allowed.size;try{sessionStorage.setItem('nym-trace-allow',JSON.stringify([...allowed]))}catch{}}
$('allow').onclick=e=>{e.stopPropagation();allowed.add($('allow').dataset.term);$('allow').textContent='In allowlist';syncAllow()};
$('allowdl').onclick=()=>{const url=URL.createObjectURL(new Blob([[...allowed].sort().join('\\n')+'\\n'],{type:'text/plain'}));const a=document.createElement('a');a.href=url;a.download='trace-allow.txt';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000)};
syncAllow();
$('session').onchange=render;$('cls').onchange=render;$('q').oninput=render;$('tiptext').onclick=()=>$('tip').style.display='none';render();
</script></body></html>'''


def page(sessions):
    data = json.dumps(sessions, ensure_ascii=False).replace('</', '<\\/')
    return PAGE.replace('__DATA__', data).encode('utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description='Privately review nym traces anon output side by side.')
    parser.add_argument('--original', type=Path, required=True)
    parser.add_argument('--anonymized', type=Path, required=True)
    parser.add_argument('--keys', type=Path)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--tailnet', action='store_true')
    group.add_argument('--bind', default=None)
    parser.add_argument('--port', type=int, default=8522)
    parser.add_argument('--write-html', type=Path, help='Write the page instead of serving it')
    args = parser.parse_args(argv)
    body = page(build(args.original, args.anonymized, args.keys))
    if args.write_html:
        args.write_html.write_bytes(body)
        args.write_html.chmod(0o600)
        return 0
    try:
        address = tailnet_address() if args.tailnet else allowed_bind(args.bind or '127.0.0.1')
        token = secrets.token_urlsafe(32)
        with Server((address, args.port), body, token) as server:
            print(f'http://{address}:{server.server_address[1]}/#token={token}', flush=True)
            try:
                server.serve_forever(poll_interval=0.1)
            except KeyboardInterrupt:
                pass
        return 0
    except Exception:
        print('error: private review server could not start; no directory or public listener exposed', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
