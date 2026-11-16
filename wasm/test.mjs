// Node smoke test for the browser build.
//
//   node wasm/test.mjs [build-web]
//
// Loads elml_studio.wasm, registers the iris sample, drives one frame per
// epoch until training finishes, then evaluates the exported model. Exits
// non-zero on the first failed assertion. No browser or test framework
// needed; this is the same module the page loads.

import { readFileSync, existsSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import assert from 'node:assert/strict';

import { loadStudio } from './web/loader.js';

const here = dirname(fileURLToPath(import.meta.url));
const root = resolve(here, '..');
const out = resolve(process.argv[2] ? process.argv[2] : join(root, 'build-web'));

const wasmPath = join(out, 'elml_studio.wasm');
assert.ok(existsSync(wasmPath), `missing ${wasmPath}; run wasm/build.sh first`);

const wasm = readFileSync(wasmPath);
const sampleDir = existsSync(join(out, 'samples'))
  ? join(out, 'samples')
  : join(root, 'examples', 'iris');

const mod = await loadStudio(wasm.buffer.slice(wasm.byteOffset, wasm.byteOffset + wasm.byteLength));
const x = mod.exports;

console.log(`loaded ${(wasm.length / 1024).toFixed(0)} KB wasm`);
assert.ok(typeof x.inst_new === 'function', 'inst_new export is missing');
assert.ok(typeof x.elml_studio_start === 'function', 'studio exports are missing');

// The page creates the surface before the studio's frame callback runs.
const inst = x.inst_new(0, 1320, 880, 1, 1, 1, 0);
assert.ok(inst, 'inst_new failed');

x.inst_frame(inst, 1 / 60, 0); // one frame so the studio owns its context

const ini = readFileSync(join(sampleDir, 'iris.ini'));
const csv = readFileSync(join(sampleDir, 'iris.csv'));
assert.equal(mod.setFile(0, 'iris.ini', ini), 0, 'could not register iris.ini');
assert.equal(mod.setFile(1, 'iris.csv', csv), 0, 'could not register iris.csv');
assert.equal(mod.setFile(4, 'iris.csv', csv), 0, 'could not register the evaluation CSV');

assert.equal(x.elml_studio_train_rows(), 0, 'rows should be zero before training starts');
x.elml_studio_start();
assert.equal(x.elml_studio_running(), 1, 'training did not start');

let frames = 0;
const limit = 1200;
while (x.elml_studio_running() === 1 && frames < limit) {
  x.inst_frame(inst, 1 / 60, 0);
  frames += 1;
}

assert.ok(frames < limit, `training did not finish within ${limit} frames`);
assert.equal(x.elml_studio_running(), 0, 'training is still running');
assert.equal(x.elml_studio_finished(), 1, 'training did not publish a finished state');
assert.ok(x.elml_studio_epoch() > 0, 'no epoch completed');
assert.equal(x.elml_studio_train_rows() > 0, true, 'no training rows');

const error = mod.text(x.elml_studio_error());
assert.equal(error, '', `training error: ${error}`);
assert.equal(x.elml_studio_exported(), 1, `model was not exported: ${mod.text(x.elml_studio_status())}`);

const model = mod.readFile('iris.elml');
assert.ok(model && model.length > 0, 'iris.elml is missing from the in-memory table');

console.log(
  `trained ${x.elml_studio_epoch()} epochs in ${frames} frames ` +
  `(best epoch ${x.elml_studio_best_epoch()}, loss ${x.elml_studio_loss().toFixed(4)}, ` +
  `accuracy ${(x.elml_studio_accuracy() * 100).toFixed(1)}%)`
);
console.log(`exported iris.elml (${model.length} bytes)`);

// The export seeds the evaluation model path; the page registers the CSV.
x.inst_frame(inst, 1 / 60, 0);
x.elml_studio_set_tab(1);
x.elml_studio_evaluate();
x.inst_frame(inst, 1 / 60, 0);

const evalError = mod.text(x.elml_studio_eval_error());
assert.equal(evalError, '', `evaluation error: ${evalError}`);
assert.equal(x.elml_studio_eval_has_result(), 1, 'evaluation produced no result');
assert.ok(x.elml_studio_eval_correct() > 0, 'evaluation scored no correct rows');
console.log(`evaluated: ${mod.text(x.elml_studio_eval_status())}`);

// Predictions can be saved from the Evaluate tab; the file lands in memfs.
assert.ok(mod.fileCount() >= 2, 'expected the sample and the exported model in memfs');
console.log(`files in memfs: ${Array.from({ length: mod.fileCount() }, (_, i) => `${mod.fileName(i)} (${mod.fileSize(i)} B)`).join(', ')}`);

x.elml_studio_shutdown();
console.log('wasm smoke test passed');
