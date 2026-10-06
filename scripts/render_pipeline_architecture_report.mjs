#!/usr/bin/env node
/**
 * Render pipeline architecture figures from Mermaid/DOT sources.
 * Usage: node scripts/render_pipeline_architecture_report.mjs [node_modules_path]
 *
 * This script generates PNG and SVG figures for the pipeline architecture report
 * from the companion .mmd (Mermaid) and .dot (Graphviz) source files.
 *
 * Requires: @mermaid-js/cli and viz.js packages in node_modules_path
 *   npm install @mermaid-js/cli viz.js
 */
import { execFileSync } from 'child_process';
import { writeFileSync, readFileSync, mkdirSync } from 'fs';
import { dirname, join } from 'path';
import { fileURLToPath } from 'url';

const __dirname = dirname(fileURLToPath(import.meta.url));
const ROOT = join(__dirname, '..');
const FIG_DIR = join(ROOT, 'reports', 'figures');

const nodeModules = process.argv[2] || join(ROOT, 'node_modules');

const figures = [
  { name: 'pipeline_experiment_overview', format: 'png' },
  { name: 'pipeline_experiment_overview', format: 'svg' },
  { name: 'pipeline_model_detail', format: 'png' },
  { name: 'pipeline_model_detail', format: 'svg' },
];

console.log('Rendering pipeline figures...');
for (const fig of figures) {
  const mmdPath = join(FIG_DIR, fig.name + '.mmd');
  const outPath = join(FIG_DIR, fig.name + '.' + fig.format);
  try {
    execFileSync('npx', [
      '--prefer-offline',
      '--prefix', nodeModules,
      '@mermaid-js/cli',
      '-i', mmdPath,
      '-o', outPath,
      '-w', fig.format === 'png' ? '620' : '600',
      '-H', fig.format === 'png' ? '1050' : '770',
      '-b', 'transparent',
    ], { stdio: 'pipe' });
    console.log(`  rendered ${fig.name}.${fig.format}`);
  } catch (e) {
    console.error(`  FAILED ${fig.name}.${fig.format}: ${e.message}`);
    console.error('  Ensure @mermaid-js/cli is installed: npm install -g @mermaid-js/cli');
  }
}
console.log('Done.');
