#!/usr/bin/env python3
"""Metadata-only ARPEGE pairing. Run with Python 3.12 and pandas 3.

The isolated metadata reader is not the model training environment. Filenames
resolve an annotated release member; geometry equality is checked separately.
"""
import csv
import hashlib
import json
from pathlib import Path
import sys
import wave

# Optional binary extensions from the host base environment are unnecessary.
for name in ('pyarrow', 'numexpr', 'bottleneck'):
    sys.modules[name] = None
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'reports/arpege_external_evaluation'


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save_csv(name, rows, fields):
    with (OUT / name).open('w', newline='') as f:
        w = csv.DictWriter(f, fields, lineterminator='\n')
        w.writeheader(); w.writerows(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    annotation = ROOT / 'data/ARPEGE/code/annotation_ARPEGE.pkl'
    df = pd.read_pickle(annotation)
    pairs, unmatched = [], []
    clap_root = ROOT / 'data/ARPEGE/extracted/RIR_CEREMA_01.2026_claps'
    rir_root = ROOT / 'data/ARPEGE/extracted/ARPEGE_01.2026'
    for ci, c in df[df.type == 'CLAPS'].iterrows():
        room, pos, mic = int(c.room), int(c.pos_num), int(c.mic_id)
        pid = f'room{room}_pos{pos}_EM{mic}'
        refs = df[(df.type == 'RIR') & (df.room == room) & (df.pos_num == pos)
                  & (df.mic_id == mic) & (df.orientation == 'front')
                  & (df.orientation_value == mic)]
        relative = Path(str(c.path).removeprefix('./ARPEGE_01.2026/'))
        candidates = sorted((clap_root / relative.parent).glob(relative.stem + '_*.wav'))
        if len(refs) != 1 or len(candidates) != 1:
            unmatched.append(dict(pairing_id=pid, reason=f'nonunique annotation/reference release mapping: {len(refs)} RIR rows, {len(candidates)} clap files'))
            continue
        ri, r = next(refs.iterrows())
        receiver, angles = f'em{mic}_xyz', f'angle_zyz_em{mic}'
        if not all(np.array_equal(np.asarray(c[k]), np.asarray(r[k]))
                   for k in ('src_xyz', receiver, angles, 'size_room')):
            unmatched.append(dict(pairing_id=pid, reason='annotated geometry or receiver orientation mismatch'))
            continue
        cp = candidates[0]
        rp = rir_root / str(r.path).removeprefix('./ARPEGE_01.2026/')
        if not rp.is_file():
            unmatched.append(dict(pairing_id=pid, reason='annotated reference absent from extracted release'))
            continue
        headers = []
        for path in (cp, rp):
            with wave.open(str(path)) as w:
                headers.append(dict(fs=w.getframerate(), channels=w.getnchannels(), frames=w.getnframes(), sample_bytes=w.getsampwidth()))
        assert all(h['fs'] == 48000 and h['channels'] == mic and h['sample_bytes'] == 2 for h in headers)
        vec = lambda v: json.dumps(np.asarray(v).tolist())
        pairs.append(dict(pairing_id=pid, room=str(room), position=str(pos), array=f'EM{mic}',
            clap_file=str(cp.relative_to(ROOT)), rir_file=str(rp.relative_to(ROOT)), channel_index=0,
            geometry_verified=True, clap_annotation_index=int(ci), rir_annotation_index=int(ri),
            original_clap_annotation_path=str(c.path), original_rir_annotation_path=str(r.path),
            release_path_alias=f'{relative.name} -> {cp.name}; unique in annotated directory',
            speaker=str(c.speaker), source_xyz=vec(c.src_xyz), receiver_xyz=vec(c[receiver]),
            receiver_euler_zyz_degrees=vec(c[angles]), reference_orientation='front',
            reference_orientation_value=mic, clap_source_euler_xyz_degrees=vec(c.angle_xyz_src),
            reference_source_euler_xyz_degrees=vec(r.angle_xyz_src),
            source_angle_metadata_equal=bool(np.array_equal(c.angle_xyz_src, r.angle_xyz_src)),
            clap_orientation_known=False, source_directivity_matched=False,
            sample_rate=48000, channels=mic, clap_frames=headers[0]['frames'], rir_frames=headers[1]['frames'],
            clap_duration_s=headers[0]['frames']/48000, rir_duration_s=headers[1]['frames']/48000))
    pairs.sort(key=lambda r:r['pairing_id'])
    assert len({r['clap_file'] for r in pairs}) == len(pairs)
    assert len({r['rir_file'] for r in pairs}) == len(pairs)
    save_csv('pairing_manifest.csv', pairs, list(pairs[0]) if pairs else ['pairing_id'])
    save_csv('unmatched.csv', unmatched, ['pairing_id', 'reason'])
    result = dict(status='metadata-validated', annotation_sha256=digest(annotation),
        annotation_rows=len(df), annotation_types=df.type.value_counts().to_dict(),
        exact_geometry_pairs=len(pairs), rooms=len({r['room'] for r in pairs}),
        room_position_configurations=len({(r['room'],r['position']) for r in pairs}),
        unmatched=len(unmatched), physical_channel_index=0,
        orientation_rule='official notebook EM32 front32, EM64 front64; no result-based selection',
        limitation='Geometry and receiver configuration match. Hand-clap and loudspeaker directivity are not matched.',
        manifest_sha256=digest(OUT/'pairing_manifest.csv'), implementation_sha256=digest(Path(__file__)),
        official_sources=['https://zenodo.org/records/20622134','https://github.com/jdpascal/ARPEGE'],
        reader_versions=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__))
    (OUT/'pairing_audit.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
