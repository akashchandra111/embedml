// The browser front end for the embedml studio wasm module.
//
// CUI draws the studio into a software surface; this file copies that surface
// to a canvas and forwards mouse, wheel and keyboard input. Training runs one
// epoch per animation frame, so the page stays responsive without threads.

import { loadStudio } from './loader.js';

const VIEW_W = 1320;
const VIEW_H = 880;
const ACCENT_VIOLET = 1;

const $ = (id) => document.getElementById(id);

// cui_key values (third_party/CUI/include/cui/cui.h).
const MODKEYS = new Set(['Shift', 'Control', 'Alt', 'Meta']);
const KEYS = {
  Tab: 1, Enter: 2, NumpadEnter: 2, Escape: 3, Backspace: 4, Delete: 5, Space: 6,
  ArrowLeft: 7, ArrowRight: 8, ArrowUp: 9, ArrowDown: 10, Home: 11, End: 12,
  PageUp: 13, PageDown: 14, Minus: 51, Equal: 52, Comma: 53, Period: 54, Slash: 55,
};
for (let i = 0; i < 26; i++) KEYS[`Key${String.fromCharCode(65 + i)}`] = 15 + i;
for (let i = 0; i < 10; i++) {
  KEYS[`Digit${i}`] = 41 + i;
  KEYS[`Numpad${i}`] = 41 + i;
}
for (let i = 1; i <= 12; i++) KEYS[`F${i}`] = 55 + i;
const CURSORS = ['default', 'pointer', 'text', 'ew-resize', 'ns-resize', 'move', 'not-allowed'];
const PASS = new Set(['F5', 'F11', 'F12']);
const PASS_CTRL = new Set(['KeyR', 'KeyT', 'KeyW', 'KeyL', 'Tab', 'PageUp', 'PageDown']);

const state = {
  mod: null,
  inst: 0,
  canvas: null,
  ctx: null,
  pixelData: null,
  dirty: true,
  last: 0,
  wakeAt: 0,
  buttons: 0,
  pasteMods: 2,
  prevFiles: '',
};

const FILE_KINDS = [
  { kind: 0, id: 'file-config', accept: '.ini,text/plain', label: 'Configuration' },
  { kind: 1, id: 'file-train', accept: '.csv,text/plain', label: 'Training CSV' },
  { kind: 2, id: 'file-val', accept: '.csv,text/plain', label: 'Validation CSV' },
  { kind: 3, id: 'file-model', accept: '.elml,.bin,application/octet-stream', label: 'Model' },
  { kind: 4, id: 'file-eval', accept: '.csv,text/plain', label: 'Evaluation CSV' },
];

function readBytes(file) {
  return file.arrayBuffer().then((buffer) => new Uint8Array(buffer));
}

function setStatus(text, kind) {
  const el = $('status');
  if (el) {
    el.textContent = text || '';
    el.dataset.kind = kind || '';
  }
}

function pick(kind) {
  const spec = FILE_KINDS.find((f) => f.kind === kind);
  const input = spec && $(spec.id);
  if (input) input.click();
}

async function loadFile(kind, file) {
  if (!state.mod || !file) return;
  const bytes = await readBytes(file);
  const rc = state.mod.setFile(kind, file.name, bytes);
  if (rc !== 0) {
    setStatus(`Could not load ${file.name} (code ${rc}).`, 'error');
    return;
  }
  const spec = FILE_KINDS.find((f) => f.kind === kind);
  setStatus(`Loaded ${spec ? spec.label.toLowerCase() : 'file'}: ${file.name}`);
  state.dirty = true;
  refreshFiles();
}

function refreshFiles() {
  const list = $('file-list');
  if (!list || !state.mod) return;
  const count = state.mod.fileCount();
  const entries = [];
  for (let i = 0; i < count; i++) {
    const name = state.mod.fileName(i);
    const size = state.mod.fileSize(i);
    if (name) entries.push({ name, size });
  }
  const key = JSON.stringify(entries);
  if (key === state.prevFiles) return;
  state.prevFiles = key;
  list.innerHTML = '';
  for (const entry of entries) {
    const row = document.createElement('li');
    const label = document.createElement('code');
    label.textContent = entry.name;
    const meta = document.createElement('span');
    meta.textContent = `${entry.size} B`;
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = 'Download';
    button.addEventListener('click', () => download(entry.name));
    row.append(label, meta, button);
    list.append(row);
  }
  if (!entries.length) {
    const row = document.createElement('li');
    row.className = 'empty';
    row.textContent = 'No files loaded yet.';
    list.append(row);
  }
}

function download(name) {
  if (!state.mod) return;
  const bytes = state.mod.readFile(name);
  if (!bytes) {
    setStatus(`Could not read ${name}.`, 'error');
    return;
  }
  const blob = new Blob([bytes], { type: name.endsWith('.csv') ? 'text/csv' : 'application/octet-stream' });
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = name;
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function resize() {
  const wrap = $('canvas-wrap');
  const canvas = state.canvas;
  if (!wrap || !canvas || !state.mod) return;
  const cssW = Math.max(320, Math.floor(wrap.clientWidth));
  const dpr = window.devicePixelRatio || 1;
  const scale = (cssW * dpr) / VIEW_W;
  const cssH = (VIEW_H * scale) / dpr;
  canvas.style.width = `${cssW}px`;
  canvas.style.height = `${cssH}px`;
  state.mod.exports.inst_resize(state.inst, VIEW_W, VIEW_H, scale);
  state.dirty = true;
}

function blit() {
  const px = state.mod.pixels(state.inst);
  const canvas = state.canvas;
  if (canvas.width !== px.w || canvas.height !== px.h) {
    canvas.width = px.w;
    canvas.height = px.h;
    state.pixelData = null;
  }
  if (!state.pixelData) state.pixelData = new ImageData(px.w, px.h);
  state.pixelData.data.set(px.data);
  state.ctx.putImageData(state.pixelData, 0, 0);
}

function hud() {
  const x = state.mod.exports;
  const running = x.elml_studio_running() === 1;
  const evaluating = x.elml_studio_eval_running() === 1;
  const finished = x.elml_studio_finished() === 1;
  const epoch = x.elml_studio_epoch();
  const epochs = x.elml_studio_epochs();
  const hasAcc = x.elml_studio_has_accuracy() === 1;
  const loss = x.elml_studio_loss();
  const acc = x.elml_studio_accuracy();

  $('kpi-epoch').textContent = epochs ? `${epoch} / ${epochs}` : '—';
  $('kpi-loss').textContent = (epoch > 0) ? loss.toFixed(4) : '—';
  $('kpi-accuracy').textContent = (epoch > 0 && hasAcc) ? `${(acc * 100).toFixed(1)}%` : (epoch > 0 ? 'n/a' : '—');
  $('kpi-best').textContent = epoch > 0 ? String(x.elml_studio_best_epoch() || '—') : '—';
  $('kpi-model').textContent = state.mod.text(x.elml_studio_model()) || '—';
  $('kpi-rows').textContent = x.elml_studio_train_rows() ? `${x.elml_studio_train_rows()} / ${x.elml_studio_val_rows()}` : '—';

  $('btn-train').disabled = running;
  $('btn-stop').disabled = !running && !evaluating;
  $('btn-eval').disabled = running;

  const error = state.mod.text(x.elml_studio_error());
  const status = state.mod.text(x.elml_studio_status());
  if (error) setStatus(error, 'error');
  else if (status) setStatus(status);
  if (finished && x.elml_studio_exported()) $('btn-download-model').disabled = false;

  const evalStatus = state.mod.text(x.elml_studio_eval_status());
  const evalError = state.mod.text(x.elml_studio_eval_error());
  $('eval-status').textContent = evalError || evalStatus || 'No evaluation yet.';
  $('eval-status').dataset.kind = evalError ? 'error' : '';
  refreshFiles();
}

function step(now) {
  if (!state.inst) return;
  const x = state.mod.exports;
  const running = x.elml_studio_running() === 1;
  const evaluating = x.elml_studio_eval_running() === 1;
  const wake = state.wakeAt && now >= state.wakeAt;
  const needs = state.dirty || running || evaluating || wake || x.inst_needs_redraw(state.inst) === 1;
  if (!needs) return;

  const dt = state.last ? Math.min((now - state.last) / 1000, 0.25) : 1 / 60;
  state.last = now;
  state.dirty = false;

  x.inst_frame(state.inst, dt, 1);
  blit();
  hud();

  const nextWake = x.inst_next_wakeup(state.inst);
  state.wakeAt = nextWake >= 0 ? now + nextWake * 1000 : 0;
  const cursor = CURSORS[x.inst_cursor(state.inst)] || 'default';
  if (state.canvas.style.cursor !== cursor) state.canvas.style.cursor = cursor;
  if (!running && !evaluating) state.last = 0;
}

function loop(now) {
  try {
    step(now);
  } catch (error) {
    console.error(error);
    setStatus(`Studio error: ${error.message}`, 'error');
    return;
  }
  requestAnimationFrame(loop);
}

function pos(event) {
  const rect = state.canvas.getBoundingClientRect();
  return [
    ((event.clientX - rect.left) * VIEW_W) / rect.width,
    ((event.clientY - rect.top) * VIEW_H) / rect.height,
  ];
}

function buttonOf(event) {
  return event.button === 2 ? 1 : event.button === 1 ? 2 : 0;
}

function bindInput() {
  const canvas = state.canvas;
  const x = state.mod.exports;
  const inst = state.inst;
  const poke = () => { state.dirty = true; };

  canvas.addEventListener('pointermove', (event) => {
    const events = event.getCoalescedEvents ? event.getCoalescedEvents() : [event];
    const last = events[events.length - 1] || event;
    const [mx, my] = pos(last);
    x.inst_mouse_move(inst, mx, my);
    poke();
  });
  canvas.addEventListener('pointerdown', (event) => {
    if (event.pointerType === 'mouse' && event.button > 2) return;
    canvas.focus({ preventScroll: true });
    try { canvas.setPointerCapture(event.pointerId); } catch (error) { /* synthetic pointer */ }
    const [mx, my] = pos(event);
    x.inst_mouse_move(inst, mx, my);
    const b = buttonOf(event);
    state.buttons |= 1 << b;
    x.inst_mouse_button(inst, b, 1);
    poke();
    event.preventDefault();
  });
  const up = (event) => {
    const b = buttonOf(event);
    if (!(state.buttons & (1 << b))) return;
    state.buttons &= ~(1 << b);
    const [mx, my] = pos(event);
    x.inst_mouse_move(inst, mx, my);
    x.inst_mouse_button(inst, b, 0);
    poke();
  };
  canvas.addEventListener('pointerup', up);
  canvas.addEventListener('pointercancel', up);
  canvas.addEventListener('pointerleave', () => {
    if (!state.buttons) {
      x.inst_mouse_leave(inst);
      poke();
    }
  });
  canvas.addEventListener('contextmenu', (event) => event.preventDefault());
  canvas.addEventListener('wheel', (event) => {
    const k = event.deltaMode === 1 ? 1 : event.deltaMode === 2 ? 10 : 1 / 100;
    let dx = -event.deltaX * k;
    let dy = -event.deltaY * k;
    if (event.shiftKey && !event.deltaX) {
      dx = dy;
      dy = 0;
    }
    const [mx, my] = pos(event);
    x.inst_mouse_move(inst, mx, my);
    x.inst_scroll(inst, dx, dy);
    x.inst_frame(inst, 0, 0);
    if (x.inst_scroll_consumed(inst) === 1) event.preventDefault();
    poke();
  }, { passive: false });

  canvas.addEventListener('keydown', (event) => {
    if (PASS.has(event.code) || ((event.ctrlKey || event.metaKey) && PASS_CTRL.has(event.code))) return;
    const mods = (event.shiftKey ? 1 : 0) | (event.ctrlKey ? 2 : 0) | (event.altKey ? 4 : 0) | (event.metaKey ? 8 : 0);
    if ((event.ctrlKey || event.metaKey) && event.code === 'KeyV') {
      state.pasteMods = mods;
      return;
    }
    const key = KEYS[event.code];
    if (key) x.inst_key(inst, key, 1, mods);
    else if (MODKEYS.has(event.key)) x.inst_key(inst, 0, 1, mods);
    if (event.key.length >= 1 && [...event.key].length === 1 && !event.ctrlKey && !event.metaKey) {
      const p = state.mod.cstr(event.key);
      x.inst_text(inst, p);
      x.web_free(p);
    }
    if (key || event.key.length === 1) event.preventDefault();
    poke();
  });
  canvas.addEventListener('keyup', (event) => {
    const key = KEYS[event.code];
    const mods = (event.shiftKey ? 1 : 0) | (event.ctrlKey ? 2 : 0) | (event.altKey ? 4 : 0) | (event.metaKey ? 8 : 0);
    if (!key) {
      if (MODKEYS.has(event.key)) {
        x.inst_key(inst, 0, 0, mods);
        poke();
      }
      return;
    }
    x.inst_key(inst, key, 0, mods);
    poke();
  });
  canvas.addEventListener('paste', (event) => {
    const text = event.clipboardData?.getData('text/plain') ?? '';
    const b = new TextEncoder().encode(text);
    const p = x.web_clipboard_buffer(b.length);
    new Uint8Array(state.mod.memory().buffer).set(b, p);
    x.inst_key(inst, KEYS.KeyV, 1, state.pasteMods);
    x.inst_key(inst, KEYS.KeyV, 0, state.pasteMods);
    event.preventDefault();
    poke();
  });
  canvas.addEventListener('focus', () => { x.inst_focus(inst, 1); poke(); });
  canvas.addEventListener('blur', () => {
    for (const key of Object.values(KEYS)) x.inst_key(inst, key, 0, 0);
    x.inst_focus(inst, 0);
    poke();
  });
}

async function loadSamples() {
  const [ini, csv] = await Promise.all([
    fetch('samples/iris.ini'),
    fetch('samples/iris.csv'),
  ]);
  if (!ini.ok || !csv.ok) throw new Error('sample files are missing (run wasm/build.sh)');
  const iniBytes = new Uint8Array(await ini.arrayBuffer());
  const csvBytes = new Uint8Array(await csv.arrayBuffer());
  state.mod.setFile(0, 'iris.ini', iniBytes);
  state.mod.setFile(1, 'iris.csv', csvBytes);
  state.mod.setFile(4, 'iris.csv', csvBytes);
  refreshFiles();
  setStatus('Loaded the iris sample. Train it, then switch to Evaluate.');
}

function bindChrome() {
  for (const spec of FILE_KINDS) {
    const input = $(spec.id);
    if (!input) continue;
    input.addEventListener('change', () => {
      const file = input.files && input.files[0];
      input.value = '';
      if (file) loadFile(spec.kind, file);
    });
  }
  $('btn-train').addEventListener('click', () => {
    if (!state.mod) return;
    state.mod.exports.elml_studio_set_tab(0);
    state.mod.exports.elml_studio_start();
    setStatus('Training started.');
  });
  $('btn-stop').addEventListener('click', () => {
    if (!state.mod) return;
    state.mod.exports.elml_studio_stop();
    setStatus('Stopping at the next batch boundary...');
  });
  $('btn-eval').addEventListener('click', () => {
    if (!state.mod) return;
    state.mod.exports.elml_studio_set_tab(1);
    state.mod.exports.elml_studio_evaluate();
    setStatus('Evaluating the saved model...');
  });
  $('btn-sample').addEventListener('click', () => {
    loadSamples().catch((error) => setStatus(error.message, 'error'));
  });
  $('btn-download-model').addEventListener('click', () => {
    const model = state.mod && state.mod.readFile('iris.elml');
    if (model) {
      const blob = new Blob([model], { type: 'application/octet-stream' });
      const url = URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = 'iris.elml';
      document.body.append(link);
      link.click();
      link.remove();
      setTimeout(() => URL.revokeObjectURL(url), 1000);
    } else {
      setStatus('No exported model yet; train first.', 'error');
    }
  });

  window.addEventListener('pagehide', () => {
    try { state.mod?.exports.elml_studio_shutdown(); } catch (error) { /* unloading */ }
  });
}

async function main() {
  state.canvas = $('canvas');
  state.ctx = state.canvas.getContext('2d', { alpha: false });
  if (!state.ctx) {
    setStatus('This browser does not support a 2D canvas.', 'error');
    return;
  }

  setStatus('Loading the studio...');
  state.mod = await loadStudio('elml_studio.wasm', { pickFile: pick });
  const x = state.mod.exports;

  state.inst = x.inst_new(0, VIEW_W, VIEW_H, 1, 1, ACCENT_VIOLET, 0);
  if (!state.inst) throw new Error('could not create the studio surface');

  resize();
  bindInput();
  bindChrome();
  new ResizeObserver(() => resize()).observe($('canvas-wrap'));

  await loadSamples();
  state.dirty = true;
  requestAnimationFrame(loop);
}

main().catch((error) => {
  console.error(error);
  setStatus(`Could not start the studio: ${error.message}`, 'error');
});
