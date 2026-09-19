"""Run MuJoCo physics and full-connectome rate dynamics on one fixed clock."""
import argparse,json,time,sys
from pathlib import Path
import numpy as np
import mujoco
from flygym.compose import FlatGroundWorld
from flygym.simulation import Simulation
from flygym.utils.math import Rotation3D
from flygym_demo.complex_terrain import make_locomotion_fly,HybridTurningController,HybridControllerObservation,apply_locomotion_action
from brain import ConnectomeBrain,ROOT

def rounded(a): return np.asarray(a).round(6).tolist()

def run(duration=2,drive=1,turn=0,silenced=False,feedback=True,output=None):
    start=time.perf_counter()
    print(json.dumps({'phase':'Loading whole-connectome neural model'}),flush=True)
    brain=ConnectomeBrain()
    fly=make_locomotion_fly();world=FlatGroundWorld()
    world.add_fly(fly,spawn_position=[0,0,0.3],spawn_rotation=Rotation3D('quat',[1,0,0,0]),add_ground_contact_sensors=False)
    sim=Simulation(world);sim.reset();mujoco.mj_forward(sim.mj_model,sim.mj_data)
    ctrl=HybridTurningController(timestep=sim.timestep);ctrl.reset(seed=0)
    body_ids=sim._internal_bodyids_by_fly[fly.name]
    body_lookup={int(b):i for i,b in enumerate(body_ids)}
    geometry=[]
    for g in range(sim.mj_model.ngeom):
        if int(sim.mj_model.geom_bodyid[g]) not in body_lookup: continue
        mesh_id=int(sim.mj_model.geom_dataid[g])
        if sim.mj_model.geom_type[g]!=mujoco.mjtGeom.mjGEOM_MESH: continue
        m=sim.mj_model
        va=int(m.mesh_vertadr[mesh_id]);vn=int(m.mesh_vertnum[mesh_id]);fa=int(m.mesh_faceadr[mesh_id]);fn=int(m.mesh_facenum[mesh_id])
        geometry.append({'name':mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_GEOM,g).split('/')[-1],'body':body_lookup[int(m.geom_bodyid[g])],'vertices':rounded(m.mesh_vert[va:va+vn].ravel()),'faces':m.mesh_face[fa:fa+fn].ravel().tolist(),'position':rounded(m.geom_pos[g]),'quaternion':rounded(m.geom_quat[g]),'color':rounded(m.geom_rgba[g])})
    origin=sim.get_body_positions(fly.name)[0].copy();origin[2]=0
    frames=[];samples=[];times=[];signal=np.zeros(2);contacts=np.zeros(2)
    foot_segments=[f'{leg}_tarsus{i}' for leg in ['lf','lm','lh','rf','rm','rh'] for i in range(1,6)]
    steps=round(duration/sim.timestep);stride=round(1/30/sim.timestep);neural_stride=round(brain.dt/sim.timestep)
    print(json.dumps({'phase':'Simulating forces, contacts and neural feedback','progress':0}),flush=True)
    for step in range(steps+1):
        obs=HybridControllerObservation.from_sim(sim,fly.name)
        # Real ground-contact observation, counted per anatomical side.
        if step%neural_stride==0:
            forces=sim.get_bodysegment_contact_forces(fly.name,foot_segments,ground_only=True).reshape(6,5,3)
            force_by_leg=np.linalg.norm(forces,axis=2).sum(axis=1)
            contacts=np.array([np.count_nonzero(force_by_leg[:3]>1e-6),np.count_nonzero(force_by_leg[3:]>1e-6)])
            signal=brain.step(drive,turn,contacts,silenced,feedback)
        if step%stride==0 or step==steps:
            positions=sim.get_body_positions(fly.name)-origin
            rotations=sim.get_body_rotations(fly.name)
            frames.append({'positions':rounded(positions.ravel()),'quaternions':rounded(rotations.ravel()),'contacts':int(sim.mj_data.ncon),'drive':rounded(signal),'feedback':rounded(contacts)})
            samples.append(brain.rates[brain.recorded].copy());times.append(round(step*sim.timestep,6))
        if step%5000==0: print(json.dumps({'phase':'Simulating forces, contacts and neural feedback','progress':step/steps}),flush=True)
        if step==steps: break
        apply_locomotion_action(sim,fly.name,ctrl.step(signal,obs));sim.step()
        if not np.isfinite(sim.mj_data.qpos).all(): raise ValueError('Physics produced non-finite coordinates')
    values=np.stack(samples)
    activity={'kind':'simulation','unit':'normalized rate (a.u.)','times':times,'values':{brain.rows[int(idx)][0]:rounded(values[:,j]) for j,idx in enumerate(brain.recorded)}}
    first=np.array(frames[0]['positions'][:3]);last=np.array(frames[-1]['positions'][:3])
    result={'schemaVersion':1,'rig':'neuromechfly-2.1.0','datasetId':brain.catalog['id'],'datasetVersion':brain.catalog['version'],'times':times,'frames':frames,'geometry':geometry,'bodyNames':[b.name for b in fly.get_bodysegs_order()],'activity':activity,
      'parameters':{'duration':duration,'drive':drive,'turn':turn,'silenced':silenced,'feedback':feedback,'seed':0},
      'metrics':{'displacementMm':float(np.linalg.norm((last-first)[:2])),'finalHeightMm':float(last[2]),'wallSeconds':time.perf_counter()-start,'neurons':len(brain.rows),'effectiveSignedEdges':brain.weights.nnz,'recordedNeurons':len(brain.recorded),'maxContacts':max(f['contacts'] for f in frames),'physicsStepSeconds':sim.timestep,'neuralStepSeconds':brain.dt},
      'provenance':{'physics':'NeuroMechFly/FlyGym 2.1.0, MuJoCo 3.9.0; position actuators, gravity, collision contacts and adhesion; seed 0','brain':'MaleCNS v1.0 weighted graph; engineered rectified tanh rate dynamics, tau 50 ms, row-normalized signed recurrence gain 0.9. ACh +1, GABA/Glu -1, other/unknown zero: assumed receptor-independent signs, not validated dynamics.','mapping':'Drive stimulates side-labelled descending neurons. Thoracic motor population means x24 feed the published hybrid CPG controller. Ground-contact counts feed side-labelled VNC sensory populations. These are engineered population mappings, not identified sensor-to-cell or muscle-to-cell connections.','recording':'All classified neurons are simulated. Only decoder neurons and 256 distributed input/sensory/intermediate cells are recorded for display. Missing cells have no displayed sample, not measured zero.','geometry':'Actual simulator mesh and body transforms in mm, source X forward/Y left/Z up. Neural overlay registration remains illustrative. Wings are passive; no flight simulation.'}}
    destination=Path(output or ROOT/'data/physics/latest.json');destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix('.tmp');tmp.write_text(json.dumps(result,separators=(',',':'),allow_nan=False));tmp.replace(destination)
    print(json.dumps({'phase':'complete','metrics':result['metrics']}),flush=True);sim.close()
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--duration',type=float,default=2);p.add_argument('--drive',type=float,default=1);p.add_argument('--turn',type=float,default=0);p.add_argument('--silenced',action='store_true');p.add_argument('--no-feedback',action='store_true');p.add_argument('--output');a=p.parse_args()
    if not 0.5<=a.duration<=5 or not 0<=a.drive<=2 or not -1<=a.turn<=1: p.error('Parameters outside supported range')
    run(a.duration,a.drive,a.turn,a.silenced,not a.no_feedback,a.output)
