"""Export actual PyTorch predictions to static replay assets; no browser inference."""
import sys
from pathlib import Path
import json
import hashlib

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / 'ml'
sys.path.insert(0, str(PROJECT))
from surgtrack.core import TinyUNet, Tracker, seed_all
from surgtrack.motion import FlowPredictor
from surgtrack.milestone2 import VideoStore, MotionTracker, condition_frame, CONDITIONS, METHODS

seed_all(7)
OUT = ROOT / 'web/assets'
OUT.mkdir(parents=True, exist_ok=True)
data = VideoStore(PROJECT / 'milestone2/data', 'test', 64, allow_all=True)
indices = data.sequences['seq06']
predictor = FlowPredictor()
predictor.load_state_dict(torch.load(PROJECT / 'milestone2/run/seed7/predictor.pt', map_location='cpu', weights_only=True)['model'])
predictor.eval()
manifest = dict(size=64, frames=len(indices), sequence='seq06', seed=7, source='SYNTHETIC_TOY_NOT_SURGICAL_DATA',
                methods=list(METHODS), conditions=list(CONDITIONS), modes=['raw','ema','motion'], predictions={}, inputs={})
truth = Image.new('L',(64*len(indices),64))
for ordinal,i in enumerate(indices):
    truth.paste(Image.fromarray((data.mask(i)[0].numpy()*255).astype('uint8')),(ordinal*64,0))
truth.save(OUT/'truth.png')
for condition in CONDITIONS:
    sheet=Image.new('RGB',(64*len(indices),64))
    for ordinal,i in enumerate(indices):
        x,_,_=condition_frame(data.images[i],condition,ordinal)
        sheet.paste(Image.fromarray((x.permute(1,2,0).numpy()*255).astype('uint8')),(ordinal*64,0))
    name=f'input-{condition}.png';sheet.save(OUT/name)
    manifest['inputs'][condition]=name
with torch.inference_mode():
    for method in METHODS:
        checkpoint=PROJECT/f'milestone2/run/seed7/{method}.pt'
        model=TinyUNet()
        model.load_state_dict(torch.load(checkpoint,map_location='cpu',weights_only=True)['model']);model.eval()
        original=json.loads((PROJECT/f'milestone2/run/seed7/{method}_evaluation.json').read_text())
        manifest['predictions'][method]={}
        for condition in CONDITIONS:
            gates={'raw':Tracker(alpha=1.),'ema':Tracker(),'motion':MotionTracker(predictor)}
            sheets={mode:Image.new('RGBA',(64*len(indices),64)) for mode in gates}
            records={mode:[] for mode in gates}
            for ordinal,i in enumerate(indices):
                x,_,_=condition_frame(data.images[i],condition,ordinal)
                probability=model(x[None]).sigmoid()[0,0].numpy()
                for mode,gate in gates.items():
                    p,status=gate.update(x,probability) if mode=='motion' else gate.update(probability)
                    mask=p>=.5
                    pixels=np.zeros((64,64,4),dtype='uint8')
                    pixels[mask]=[60,240,180,255]
                    sheets[mode].paste(Image.fromarray(pixels),(ordinal*64,0))
                    row=next(r for r in original['frames'] if r['sequence']=='seq06' and r['frame']==ordinal and r['condition']==condition and r['mode']==mode)
                    assert status['reliable']==row['reliable']
                    records[mode].append({k:row[k] for k in ['dice','iou','entropy','reliable','reason','centroid_xy','phase','saturated_pixel_fraction','dark_pixel_fraction']})
            manifest['predictions'][method][condition]={}
            for mode,sheet in sheets.items():
                filename=f'mask-{method}-{condition}-{mode}.png';sheet.save(OUT/filename)
                manifest['predictions'][method][condition][mode]=dict(sprite=filename,frames=records[mode])
        print('Exported',method,flush=True)
manifest['checkpoint_sha256']=hashlib.sha256((PROJECT/'milestone2/run/seed7/dual_cycle.pt').read_bytes()).hexdigest()
(OUT/'replay.json').write_text(json.dumps(manifest,separators=(',',':')))
print('Saved static replay assets',flush=True)
