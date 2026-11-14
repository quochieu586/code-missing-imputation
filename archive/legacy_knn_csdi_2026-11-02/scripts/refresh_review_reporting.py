"""Explicitly reseal a report-only correction; preserve all experimental outputs."""
import json
import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
sys.dont_write_bytecode=True
from src.evaluation.research_pilot import sha256,write_json,write_hashes


def main():
    e3=ROOT/'artifacts/e3_csdi_revised';e4=ROOT/'artifacts/e4_pilot_revised'
    hashes=json.loads((e3/'file_hashes.json').read_text())
    changes={}
    for rel,want in hashes['artifact_hashes'].items():
        actual=sha256(e3/rel)
        if actual!=want:
            assert rel=='e3_report.md','experiment output changed: '+rel
            changes[rel]={'previous_hash':want,'corrected_hash':actual}
    assert len(changes)==1,'expected one report correction'
    for rel,want in hashes['code_hashes'].items():
        if sha256(rel)!=want:
            assert Path(rel).name=='finalize_e3_review.py','unexpected code change'
    write_json(e3/'report_correction.json',{'reason':'Init-only Aitchison scale guard is 1, as already recorded in JSON; table had stale zero',
        'changes':changes,'predictions_metrics_checkpoints_unchanged':True})
    write_hashes(e3,[Path(p) for p in hashes['code_hashes']])
    integrity=json.loads((e4/'upstream_integrity.json').read_text());updated={}
    allowed={'artifacts/e3_csdi_revised/e3_report.md','artifacts/e3_csdi_revised/file_hashes.json'}
    for rel,want in integrity['hashes'].items():
        actual=sha256(ROOT/rel)
        if actual!=want:
            assert rel.replace('\\','/') in allowed,'unexpected upstream modification: '+rel
            updated[rel]={'previous_hash':want,'corrected_hash':actual}
            integrity['hashes'][rel]=actual
    integrity['postrun_report_correction']=updated
    write_json(e4/'upstream_integrity.json',integrity)
    hashes=json.loads((e4/'file_hashes.json').read_text())
    write_hashes(e4,[Path(p) for p in hashes['code_hashes']]+[Path(__file__)])
    print('Report correction resealed; no predictions, metrics, checkpoints or frozen legacy files changed.')


if __name__=='__main__':main()
