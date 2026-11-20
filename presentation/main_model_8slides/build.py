"""Build the complete 15-slide talk using installed tools; run from any cwd.

    python /absolute/path/to/main_model_8slides/build.py [all|deck]

all: editable TikZ -> PDF/SVG/PNG, then Beamer -> 15-page PDF and PNG pages.
deck: reuse existing figure PDF exports, compile and render the final talk.
No package installation, shell escape, runtime/model imports or network calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parent
BUILD = ROOT / 'build'
FIGURES = ('02_pipeline', '03_two_views', '04_network', '07_r2_allocation', '08_refinement', 'eval_mae', 'eval_tier3')
EXPECTED_PAGES = 15
COMMANDS = []


def command(args, cwd, label):
    run = subprocess.run([str(x) for x in args], cwd=cwd, capture_output=True,
                         text=True, encoding='utf-8', errors='replace')
    log = BUILD / 'logs' / f'{label}.txt'
    log.write_text(run.stdout + run.stderr, encoding='utf-8')
    COMMANDS.append({'args': [str(x) for x in args], 'cwd': str(cwd),
                     'exit_code': run.returncode, 'log': str(log.relative_to(ROOT))})
    if run.returncode:
        raise RuntimeError(f'{label} failed: {log}')
    return run.stdout


def compile_tex(source, cwd, out, name, passes=1):
    for turn in range(passes):
        command(['pdflatex', '-disable-installer', '-disable-write18', '-halt-on-error',
                 '-interaction=nonstopmode', f'-output-directory={out}',
                 f'-jobname={name}', source], cwd, f'{name}-{turn + 1}')
    return out / f'{name}.pdf'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('target', nargs='?', choices=('all', 'deck'), default='all')
    args = parser.parse_args()
    for tool in ('pdflatex', 'pdftocairo', 'pdftoppm', 'pdfinfo', 'pdffonts', 'pdftotext'):
        if not shutil.which(tool):
            raise RuntimeError(f'Missing installed tool: {tool}; no automatic installer')
    for sub in ('logs', 'figures', 'pages'):
        (BUILD / sub).mkdir(parents=True, exist_ok=True)
    for ext in ('pdf', 'svg', 'png'):
        (ROOT / 'exports' / ext).mkdir(parents=True, exist_ok=True)
    if args.target == 'all':
        for name in FIGURES:
            pdf = compile_tex(f'{name}.tex', ROOT / 'figures', BUILD / 'figures', name)
            shutil.copyfile(pdf, ROOT / 'exports/pdf' / pdf.name)
            command(['pdftocairo', '-svg', pdf, ROOT / 'exports/svg' / f'{name}.svg'], ROOT, f'{name}-svg')
            command(['pdftoppm', '-r', '180', '-png', '-singlefile', pdf,
                     ROOT / 'exports/png' / name], ROOT, f'{name}-png')
    for name in FIGURES:
        if not (ROOT / 'exports/pdf' / f'{name}.pdf').is_file():
            raise RuntimeError(f'Missing figure PDF; run build.py all: {name}')
    pdf = compile_tex('main.tex', ROOT, BUILD, 'main_model_presentation', passes=2)
    # Publish the single current deck at the project root.
    shutil.copyfile(pdf, ROOT / 'main_model_presentation_8slides.pdf')
    info = command(['pdfinfo', pdf], ROOT, 'deck-info')
    pages = int(re.search(r'^Pages:\s+(\d+)', info, re.MULTILINE)[1])
    if pages != EXPECTED_PAGES:
        raise RuntimeError(f'Expected exactly {EXPECTED_PAGES} pages, got {pages}')
    fonts = command(['pdffonts', pdf], ROOT, 'deck-fonts')
    if 'Type 3' in fonts:
        raise RuntimeError('Type 3 fonts are not acceptable')
    text = command(['pdftotext', '-layout', pdf, '-'], ROOT, 'deck-text')
    if re.search(r'(?i)\bpending\b|setup preview|placement preview|asset placement', text):
        raise RuntimeError('Setup-only/pending text remains in the final deck')
    command(['pdftoppm', '-scale-to', '1440', '-png', pdf, BUILD / 'pages/slide'], ROOT, 'deck-render')
    issues = []
    for log in [BUILD / 'main_model_presentation.log', *sorted((BUILD / 'figures').glob('*.log'))]:
        for line in log.read_text(encoding='utf-8', errors='replace').splitlines():
            if any(term in line for term in ('Overfull ', 'Underfull ', 'LaTeX Font Warning', 'Missing character')):
                issues.append({'file': str(log.relative_to(ROOT)), 'message': line})
    outputs = [pdf, *sorted((BUILD / 'pages').glob('slide-*.png'))]
    report = {'status': 'built_visual_review_required' if not issues else 'diagnostics_require_repair',
              'pages': pages, 'final_prose': True, 'installer_enabled': False,
              'shell_escape_enabled': False, 'diagnostics': issues, 'commands': COMMANDS,
              'sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs}}
    (BUILD / 'build-report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(f'Built {pdf}: {pages} pages, {len(issues)} box/font diagnostics.')
    if issues:
        raise RuntimeError('Inspect build/build-report.json before accepting the deck')


if __name__ == '__main__':
    main()
