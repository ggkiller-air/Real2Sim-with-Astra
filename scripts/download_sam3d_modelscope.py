#!/usr/bin/env python3
"""Download the public ModelScope snapshot with revision pinning and SHA-256 checks.

Uses the core environment's httpx. Never stores credentials or enables inference.
Interrupted files are retained as .part and resumed when the server supports Range.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import threading
import time
import httpx

API = 'https://www.modelscope.cn/api/v1/models/facebook/sam-3d-objects'


def sha256(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, default=Path('vendor/sam-3d-objects/checkpoints/modelscope'))
    parser.add_argument('--workers', type=int, default=3, choices=range(1, 5))
    args = parser.parse_args()
    dest = args.destination.resolve()
    dest.mkdir(parents=True, exist_ok=True)
    manifest_path = dest/'download_manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
    else:
        with httpx.Client(follow_redirects=True, timeout=60) as client:
            response = client.get(API+'/repo/files', params={'Revision':'master','Recursive':'true'})
            response.raise_for_status()
            files = response.json()['Data']['Files']
        selected = [f for f in files if f['Type']=='blob' and
                    (f['Path'].startswith('checkpoints/') or f['Path'] in ('LICENSE','README.md','configuration.json'))
                    and f['Size'] > 0]
        manifest = {'source':API, 'files':selected, 'skipped_empty':[f['Path'] for f in files if f['Type']=='blob' and f['Size']==0]}
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    files = manifest['files']
    for row in files:
        rel = Path(row['Path'])
        if rel.is_absolute() or '..' in rel.parts:
            raise ValueError('Unsafe remote path')
    # Preserve checkpoint-relative paths so pipeline.yaml needs no rewriting.
    def target(row):
        rel = Path(row['Path'])
        return dest/(Path(*rel.parts[1:]) if rel.parts[0]=='checkpoints' else rel)
    remaining = sum(max(0, row['Size']-max(target(row).stat().st_size if target(row).exists() else 0,
                                           target(row).with_suffix(target(row).suffix+'.part').stat().st_size if target(row).with_suffix(target(row).suffix+'.part').exists() else 0)) for row in files)
    if shutil.disk_usage(dest).free < remaining+2*1024**3:
        raise RuntimeError('Insufficient disk space (requires download plus 2 GiB reserve)')
    total = sum(row['Size'] for row in files)
    progress, lock = {}, threading.Lock()
    def download(row):
        final = target(row)
        final.parent.mkdir(parents=True, exist_ok=True)
        partial = final.with_suffix(final.suffix+'.part')
        if final.exists() and final.stat().st_size == row['Size'] and sha256(final) == row['Sha256']:
            with lock: progress[row['Path']] = row['Size']
            print('VERIFIED existing',row['Path'],flush=True)
            return
        for attempt in range(4):
            offset = partial.stat().st_size if partial.exists() else 0
            if offset == row['Size'] and sha256(partial) == row['Sha256']:
                partial.replace(final)
                with lock: progress[row['Path']] = row['Size']
                return
            if offset >= row['Size']:
                # A corrupt incomplete download is ours to replace on retry.
                partial.unlink(); offset = 0
            try:
                with httpx.Client(follow_redirects=True, timeout=httpx.Timeout(90,connect=30)) as client:
                    with client.stream('GET',API+'/repo',params={'Revision':row['Revision'],'FilePath':row['Path']},
                                       headers={'Range':f'bytes={offset}-'} if offset else {}) as response:
                        response.raise_for_status()
                        if offset and response.status_code != 206:
                            offset = 0
                        if response.status_code == 206 and not response.headers.get('content-range','').startswith(f'bytes {offset}-'):
                            raise RuntimeError('Invalid range response')
                        with partial.open('ab' if offset else 'wb') as out:
                            with lock: progress[row['Path']] = offset
                            for block in response.iter_bytes(4*1024*1024):
                                out.write(block);offset += len(block)
                                with lock: progress[row['Path']] = offset
                if partial.stat().st_size != row['Size'] or sha256(partial) != row['Sha256']:
                    raise RuntimeError('File size or SHA-256 mismatch: '+row['Path'])
                partial.replace(final)
                print('VERIFIED',row['Path'],flush=True)
                return
            except (httpx.HTTPError, RuntimeError) as exc:
                if attempt == 3: raise
                print('RETRY',row['Path'],type(exc).__name__,flush=True)
                time.sleep(2*(attempt+1))
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(download,row) for row in sorted(files,key=lambda row:row['Size'])]
        started = time.monotonic()
        while not all(f.done() for f in futures):
            with lock: done = sum(progress.values())
            print(f'PROGRESS {done/1e9:.2f}/{total/1e9:.2f} GB ({100*done/total:.1f}%) elapsed {time.monotonic()-started:.0f}s',flush=True)
            time.sleep(15)
        for f in futures: f.result()
    report = {'source':API,'complete':True,'bytes':total,'verified_files':len(files),
              'pipeline':str(dest/'pipeline.yaml'),'inference_validated':False,
              'files':[{'path':str(target(r).relative_to(dest)),'sha256':r['Sha256'],'bytes':r['Size'],'revision':r['Revision']} for r in files]}
    (dest/'download_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='files'},indent=2),flush=True)


if __name__=='__main__':
    main()
