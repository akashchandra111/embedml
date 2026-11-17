// Loads the embedml studio wasm module and provides the small host shim it
// expects (libc + CUI's browser host). This is a trimmed copy of CUI's
// site/js/runtime.js with one extra import: js_studio_pick_file, which the
// studio uses to ask the page for a file. No dependencies, works in a browser
// and in Node (for the smoke test).

const enc = new TextEncoder();
const dec = new TextDecoder();

function fmtFloat(v, conv, prec, alt) {
  const upper = conv === 70 || conv === 69 || conv === 71 || conv === 65; // F E G A
  let s;
  if (!Number.isFinite(v)) s = Number.isNaN(v) ? 'nan' : 'inf';
  else {
    const fixed = (x, p) => {
      if (x >= 1e21) {
        const int = BigInt(Math.trunc(x)).toString();
        return p > 0 ? `${int}.${'0'.repeat(p)}` : int;
      }
      return x.toFixed(Math.min(p, 100));
    };
    const expo = (x, p) => x.toExponential(Math.min(p, 100)).replace(/e([+-])(\d)$/, 'e$10$2');
    const c = String.fromCharCode(conv).toLowerCase();
    if (c === 'f') s = fixed(v, prec);
    else if (c === 'e') s = expo(v, prec);
    else if (c === 'a') s = v.toString(16);
    else {
      const P = prec === 0 ? 1 : prec;
      const X = v === 0 ? 0 : Number(v.toExponential(P - 1).split('e')[1]);
      if (P > X && X >= -4) s = fixed(v, P - 1 - X);
      else s = expo(v, P - 1);
      if (!alt && s.includes('.')) s = s.replace(/\.?0+(e|$)/, '$1');
    }
    if (alt && prec === 0 && !s.includes('.') && c !== 'g') s = s.replace(/^(\d+)/, '$1.');
  }
  return upper ? s.toUpperCase() : s;
}

export async function loadStudio(source, hooks = {}) {
  let mem;
  const u8 = () => new Uint8Array(mem.buffer);
  const str = (p, n) => dec.decode(u8().subarray(p, n === undefined ? u8().indexOf(0, p) : p + n));
  const lines = { 1: '', 2: '' };

  const env = {
    js_write(fd, p, n) {
      const sink = (lines[fd] = (lines[fd] || '') + str(p, n));
      let i;
      let rest = sink;
      while ((i = rest.indexOf('\n')) >= 0) {
        const line = rest.slice(0, i);
        rest = rest.slice(i + 1);
        (fd === 2 ? console.warn : console.log)('[elml]', line);
      }
      lines[fd] = rest;
    },
    js_fmt_float(v, conv, prec, alt, buf, cap) {
      const b = enc.encode(fmtFloat(v, conv, prec, alt));
      const n = Math.min(b.length, cap);
      u8().set(b.subarray(0, n), buf);
      return n;
    },
    js_parse_float: (p, n) => parseFloat(str(p, n)),
    js_date_now: () => Date.now(),
    js_perf_now: () => performance.now(),
    js_tz_offset: (ms) => -new Date(ms).getTimezoneOffset(),
    js_abort(p) {
      throw new Error(`embedml studio aborted: ${str(p)}`);
    },
    // The studio asks the page for a file. kinds: 0 config, 1 train CSV,
    // 2 validation CSV, 3 model, 4 evaluation CSV.
    js_studio_pick_file(kind) {
      if (hooks.pickFile) hooks.pickFile(kind);
    },
    js_clipboard_set(p, n) {
      const text = str(p, n);
      navigator.clipboard?.writeText(text).catch(() => {});
    },
    // The software renderer is used, so no GPU textures are requested. The
    // exports are still imported by web_host.c, so provide inert versions.
    js_tex_create: () => 0,
    js_tex_update: () => {},
    js_tex_destroy: () => {},
    // The studio does not draw emoji today; returning 0 means "no glyph".
    js_draw_emoji: () => 0,
  };

  for (const f of ['sin', 'cos', 'tan', 'asin', 'acos', 'atan', 'sinh', 'cosh', 'tanh', 'exp', 'expm1', 'log', 'log10', 'log2', 'log1p', 'cbrt'])
    env[`math_${f}`] = Math[f];
  env.math_atan2 = Math.atan2;
  env.math_pow = Math.pow;
  env.math_hypot = Math.hypot;
  env.math_fmod = (x, y) => x % y;

  let instance;
  if (source instanceof WebAssembly.Module) instance = await WebAssembly.instantiate(source, { env });
  else if (source instanceof ArrayBuffer || ArrayBuffer.isView(source)) instance = (await WebAssembly.instantiate(source, { env })).instance;
  else {
    const response = await fetch(source);
    if (!response.ok) throw new Error(`${source}: ${response.status}`);
    instance = (await WebAssembly.instantiate(await response.arrayBuffer(), { env })).instance;
  }

  const x = instance.exports;
  mem = x.memory;
  x.web_boot();

  const cstr = (s) => {
    const b = enc.encode(s);
    const p = x.web_alloc(b.length + 1);
    u8().set(b, p);
    u8()[p + b.length] = 0;
    return p;
  };

  return {
    exports: x,
    memory: () => mem,
    text(s) {
      return str(s);
    },
    cstr,
    // Registers `bytes` in the studio's in-memory file table and tells the
    // studio which slot it belongs to.
    setFile(kind, name, bytes) {
      const data = bytes instanceof Uint8Array ? bytes : new Uint8Array(bytes);
      const p = x.web_alloc(data.length);
      u8().set(data, p);
      const namep = cstr(name);
      const rc = x.elml_studio_set_file(kind, namep, p, data.length);
      x.web_free(namep);
      x.web_free(p);
      return rc;
    },
    readFile(name) {
      const namep = cstr(name);
      const lenp = x.web_alloc(4);
      const p = x.elml_studio_read_file(namep, lenp);
      x.web_free(namep);
      if (!p) {
        x.web_free(lenp);
        return null;
      }
      const n = new Int32Array(mem.buffer, lenp, 1)[0];
      const copy = new Uint8Array(n);
      copy.set(u8().subarray(p, p + n));
      x.web_free(lenp);
      // The returned pointer aliases the memfs entry, so hand back a copy that
      // stays valid after the studio replaces or clears that entry.
      return copy;
    },
    pixels(inst) {
      const p = x.inst_pixels(inst);
      const w = x.inst_pixel_width(inst);
      const h = x.inst_pixel_height(inst);
      return { w, h, data: new Uint8ClampedArray(mem.buffer, p, w * h * 4) };
    },
    fileCount: () => x.elml_studio_file_count(),
    fileName: (i) => str(x.elml_studio_file_name(i)),
    fileSize: (i) => x.elml_studio_file_size(i),
  };
}
