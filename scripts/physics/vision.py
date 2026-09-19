"""Rendered eye observations and an explicit engineered red-target controller."""
import base64
from io import BytesIO
import numpy as np
from PIL import Image
from flygym.utils.mjcf import GEOM_TYPES


def add_target(world, position):
    world.mjcf_root.worldbody.add_geom(name='visual_target',type=GEOM_TYPES['sphere'],pos=[*position,2],size=[2,0,0],rgba=[1,.05,.05,1],contype=0,conaffinity=0)


def observe(sim, fly_name):
    images=sim.get_raw_vision(fly_name)
    rgb=images.astype(np.float32)
    red=(rgb[...,0]>60)&(rgb[...,0]>2*rgb[...,1])&(rgb[...,0]>2*rgb[...,2])
    scores=red.mean(axis=(1,2))
    # Left/right retinal evidence only; no world position or target bearing is supplied.
    imbalance=float((scores[0]-scores[1])/(scores.sum()+.002))
    return images, scores, float(np.clip(-.6*imbalance,-.6,.6))


def encode_eyes(images):
    result=[]
    for image in images:
        buffer=BytesIO()
        # Display thumbnail only. Control uses the unscaled 450 x 512 RGB observation.
        Image.fromarray(image).resize((225,256)).save(buffer,format='JPEG',quality=75)
        result.append('data:image/jpeg;base64,'+base64.b64encode(buffer.getvalue()).decode('ascii'))
    return result
