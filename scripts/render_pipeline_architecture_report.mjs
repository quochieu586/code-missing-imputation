/** Render the reviewed pipeline figures without running any research code.
 * Usage: node scripts/render_pipeline_architecture_report.mjs [node_modules]
 * Dependencies: @viz-js/viz and sharp (available in the bundled runtime).
 */
import { createRequire } from 'node:module';
import { readFile, writeFile } from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const dependencyRoot = process.argv[2];
const require = dependencyRoot
  ? createRequire(path.resolve(dependencyRoot, '_pipeline_renderer.cjs'))
  : createRequire(import.meta.url);
const { instance } = require('@viz-js/viz');
const sharp = require('sharp');
const viz = await instance();
for (const name of ['pipeline_experiment_overview', 'pipeline_model_detail']) {
  const base = path.join(root, 'reports', 'figures', name);
  const dot = await readFile(`${base}.dot`, 'utf8');
  let result;
  try {
    result = viz.render(dot, { format: 'svg', engine: 'dot' });
  } catch (error) {
    console.error(`${name}: ${error.message}`);
    process.exit(1);
  }
  if (result.status !== 'success') throw new Error(JSON.stringify(result.errors));
  const warnings = result.errors.filter(e => e.level === 'warning');
  if (warnings.length) throw new Error(JSON.stringify(warnings));
  await writeFile(`${base}.svg`, result.output);
  await sharp(Buffer.from(result.output), { density: 180 })
    .resize({ width: 2600 }).flatten({ background: '#ffffff' })
    .png().toFile(`${base}.png`);
  const meta = await sharp(`${base}.png`).metadata();
  console.log(`${name}: ${meta.width} x ${meta.height}; SVG and PNG rendered`);
}
