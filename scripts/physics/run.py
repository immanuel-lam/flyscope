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
from vision import add_target,observe,encode_eyes
from backflip import BackflipAssist
from obstacles import add_obstacles,RangeSensor,LAYOUTS
from odour import sample as sample_odour,steering as odour_steering

sys.path.insert(0,str(ROOT/'scripts/memory'))
from runtime import CueMemory

def rounded(a): return np.asarray(a).round(6).tolist()

def run(duration=2,drive=1,turn=0,silenced=False,feedback=True,output=None,vision=False,vision_control=True,target=(12,4),odour=False,odour_control=True,obstacles=False,avoidance=True,layout="training",memory=False,cue_side="left",reset_memory=False,backflip=False):
    start=time.perf_counter()
    print(json.dumps({'phase':'Loading whole-connectome neural model'}),flush=True)
    brain=ConnectomeBrain()
    memory_model=CueMemory() if memory else None
    memory_frames=[];memory_samples=[];memory_turn=0.;memory_tick=0
    fly=make_locomotion_fly();world=FlatGroundWorld()
    if vision:
        fly.add_vision();add_target(world,target)
    if obstacles:add_obstacles(world,layout)
    world.add_fly(fly,spawn_position=[0,0,0.3],spawn_rotation=Rotation3D('quat',[1,0,0,0]),add_ground_contact_sensors=False)
    sim=Simulation(world);sim.reset();mujoco.mj_forward(sim.mj_model,sim.mj_data)
    ctrl=HybridTurningController(timestep=sim.timestep);ctrl.reset(seed=0)
    body_ids=sim._internal_bodyids_by_fly[fly.name]
    body_names=[b.name for b in fly.get_bodysegs_order()]
    range_sensor=RangeSensor(sim,mujoco.mj_name2id(sim.mj_model,mujoco.mjtObj.mjOBJ_GEOM,fly.name+'/c_head'),int(body_ids[body_names.index('c_thorax')])) if obstacles else None
    flip=BackflipAssist(sim,int(body_ids[0])) if backflip else None
    flip_frames=[];flip_state=None
    obstacle_frames=[];obstacle_turn=0.0;obstacle_contacts=0
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
    visual_frames=[];visual_turn=0.0
    odour_frames=[];odour_turn=0.0
    antenna_indices=[next(i for i,b in enumerate(fly.get_bodysegs_order()) if b.name==name) for name in ['l_funiculus','r_funiculus']] if odour else []
    foot_segments=[f'{leg}_tarsus{i}' for leg in ['lf','lm','lh','rf','rm','rh'] for i in range(1,6)]
    steps=round(duration/sim.timestep);stride=round(1/30/sim.timestep);neural_stride=round(brain.dt/sim.timestep)
    print(json.dumps({'phase':'Simulating forces, contacts and neural feedback','progress':0}),flush=True)
    for step in range(steps+1):
        if range_sensor:obstacle_contacts+=int(any(c.geom1 in range_sensor.obstacle_ids or c.geom2 in range_sensor.obstacle_ids for c in sim.mj_data.contact))
        obs=HybridControllerObservation.from_sim(sim,fly.name)
        if vision and step%round(.1/sim.timestep)==0:
            images,scores,visual_turn=observe(sim,fly.name)
            visual_frames.append({'time':round(step*sim.timestep,6),'eyes':encode_eyes(images),'redFraction':rounded(scores),'turn':visual_turn if vision_control else 0.0})
        if memory_model and step%round(.1/sim.timestep)==0:
            cue=np.zeros(2,np.float32)
            if memory_tick<3:cue[0 if cue_side=='left' else 1]=1.
            if reset_memory and memory_tick==3:memory_model.reset()
            logits=memory_model.step(cue,ablated=silenced)
            if silenced:memory_model.reset();logits=np.zeros(2,np.float32)
            memory_turn=float(.6*np.tanh((logits[1]-logits[0])/4)) if memory_tick>=18 else 0.
            memory_frames.append({'time':round(step*sim.timestep,6),'cue':rounded(cue),'logits':rounded(logits),'turn':memory_turn,'phase':'cue' if memory_tick<3 else 'delay' if memory_tick<18 else 'response'})
            memory_tick+=1
        # Real ground-contact observation, counted per anatomical side.
        if step%neural_stride==0:
            if range_sensor:
                sensed=range_sensor.observe();obstacle_turn=sensed['turn'] if avoidance else 0.0
                sensed.update(time=round(step*sim.timestep,6),turn=obstacle_turn,origin=rounded(np.array(sensed['origin'])-origin));obstacle_frames.append(sensed)
            if odour:
                antenna_positions=sim.get_body_positions(fly.name)[antenna_indices]
                concentration=sample_odour(antenna_positions,target)
                odour_turn=odour_steering(concentration) if odour_control else 0.0
                odour_frames.append({'time':round(step*sim.timestep,6),'antennae':rounded(antenna_positions-origin),'concentration':rounded(concentration),'turn':odour_turn})
            forces=sim.get_bodysegment_contact_forces(fly.name,foot_segments,ground_only=True).reshape(6,5,3)
            force_by_leg=np.linalg.norm(forces,axis=2).sum(axis=1)
            contacts=np.array([np.count_nonzero(force_by_leg[:3]>1e-6),np.count_nonzero(force_by_leg[3:]>1e-6)])
            signal=brain.step(0 if memory_model and step*sim.timestep<1.8 else drive,float(np.clip(turn+(visual_turn if vision and vision_control else 0)+odour_turn+obstacle_turn+memory_turn,-1,1)),contacts,silenced,feedback)
        if flip:flip_state=flip.step(step*sim.timestep,signal)
        if step%stride==0 or step==steps:
            if flip:flip_frames.append({'time':round(step*sim.timestep,6),**flip_state})
            positions=sim.get_body_positions(fly.name)-origin
            rotations=sim.get_body_rotations(fly.name)
            frames.append({'positions':rounded(positions.ravel()),'quaternions':rounded(rotations.ravel()),'contacts':int(sim.mj_data.ncon),'drive':rounded(signal),'feedback':rounded(contacts)})
            if memory_model:memory_samples.append(memory_model.state.copy())
            samples.append(brain.rates[brain.recorded].copy());times.append(round(step*sim.timestep,6))
        if step%5000==0: print(json.dumps({'phase':'Simulating forces, contacts and neural feedback','progress':step/steps}),flush=True)
        if step==steps: break
        apply_locomotion_action(sim,fly.name,ctrl.step(signal,obs));sim.step()
        if not np.isfinite(sim.mj_data.qpos).all(): raise ValueError('Physics produced non-finite coordinates')
    values=np.stack(samples)
    activity={'kind':'simulation','unit':'normalized rate (a.u.)','times':times,'values':{brain.rows[int(idx)][0]:rounded(values[:,j]) for j,idx in enumerate(brain.recorded)}}
    first=np.array(frames[0]['positions'][:3]);last=np.array(frames[-1]['positions'][:3])
    result={'schemaVersion':1,'rig':'neuromechfly-2.1.0','datasetId':brain.catalog['id'],'datasetVersion':brain.catalog['version'],'times':times,'frames':frames,'geometry':geometry,'bodyNames':[b.name for b in fly.get_bodysegs_order()],'activity':activity,
      'parameters':{'duration':duration,'drive':drive,'turn':turn,'silenced':silenced,'feedback':feedback,'seed':0,'vision':vision,'visionControl':vision_control,'target':list(target),'odour':odour,'odourControl':odour_control,'obstacles':obstacles,'avoidance':avoidance,'layout':layout,'memory':memory,'cueSide':cue_side,'resetMemory':reset_memory,'backflip':backflip},
      'metrics':{'displacementMm':float(np.linalg.norm((last-first)[:2])),'finalHeightMm':float(last[2]),'wallSeconds':time.perf_counter()-start,'neurons':len(brain.rows),'effectiveSignedEdges':brain.weights.nnz,'recordedNeurons':len(brain.recorded),'maxContacts':max(f['contacts'] for f in frames),'physicsStepSeconds':sim.timestep,'neuralStepSeconds':brain.dt},
      'provenance':{'physics':'NeuroMechFly/FlyGym 2.1.0, MuJoCo 3.9.0; position actuators, gravity, collision contacts and adhesion; seed 0','brain':'MaleCNS v1.0 weighted graph; engineered rectified tanh rate dynamics, tau 50 ms, row-normalized signed recurrence gain 0.9. ACh +1, GABA/Glu -1, other/unknown zero: assumed receptor-independent signs, not validated dynamics.','mapping':'Drive stimulates side-labelled descending neurons. Thoracic motor population means x24 feed the published hybrid CPG controller. Ground-contact counts feed side-labelled VNC sensory populations. These are engineered population mappings, not identified sensor-to-cell or muscle-to-cell connections.','recording':'All classified neurons are simulated. Only decoder neurons and 256 distributed input/sensory/intermediate cells are recorded for display. Missing cells have no displayed sample, not measured zero.','geometry':'Actual simulator mesh and body transforms in mm, source X forward/Y left/Z up. Neural overlay registration remains illustrative. Wings are passive; no flight simulation.'}}
    if vision:
        result['vision']={'frames':visual_frames,'target':list(target),'targetPosition':rounded(np.array([*target,2])-origin),'sourceSize':[450,512],'control':'Red pixel fraction difference between actual fisheye eye-camera frames; engineered DN stimulus mapping, not biological color vision. Target is visual only and non-colliding.'}
        result['metrics']['targetStartDistanceMm']=float(np.linalg.norm(origin[:2]-target))
        result['metrics']['targetFinalDistanceMm']=float(np.linalg.norm(last[:2]+origin[:2]-target))
    if odour:
        result['odour']={'frames':odour_frames,'sourcePosition':rounded(np.array([*target,0])-origin),'spreadMm':5,'control':'Static Gaussian concentration field sampled at left and right funiculus body positions; normalized bilateral contrast drives engineered descending-neuron stimuli. No fluid dynamics or biological receptor mapping.'}
        result['metrics']['odourStartDistanceMm']=float(np.linalg.norm(origin[:2]-target))
        result['metrics']['odourFinalDistanceMm']=float(np.linalg.norm(last[:2]+origin[:2]-target))
    if obstacles:
        result['obstacles']={'frames':obstacle_frames,'geometry':[{'position':rounded(np.array([x,y,2])-origin),'radius':r,'halfHeight':2} for x,y,r in LAYOUTS[layout]],'control':'Nine artificial horizontal MuJoCo range rays from the head, maximum 8 mm. Obstacle-only ray filter. A 0.5 s clearance hold preserves the last turn after rays clear. Bounded avoidance stimulus enters descending neurons; not biological perception.'}
        result['metrics']['obstacleContactSteps']=obstacle_contacts
        result['metrics']['forwardProgressMm']=float(last[0]-first[0])
    if memory_model:
        memory_values=np.stack(memory_samples)
        result['memory']={'frames':memory_frames,'activity':{'kind':'simulation','unit':'trained cue-memory state (a.u.)','times':times,'values':{row[0]:rounded(memory_values[:,j]) for j,row in enumerate(memory_model.config['neurons'])}},'control':'Separate trained 512-cell MaleCNS cue-memory circuit. Three 100 ms cue updates, 16 blank delay updates, then learned logits supply a bounded DN stimulus to the separate whole-CNS locomotion model. Shared source IDs do not make these two state arrays one biological network.'}
    if flip:result['backflip']={'frames':flip_frames,'control':'Neural motor-output threshold gates an external force and torque controller on the thorax. The assist lifts and rotates the body through contact physics; no pose teleportation. This is an engineered stunt, not a learned or biologically validated backflip. Forces use MuJoCo model units.'}
    destination=Path(output or ROOT/'data/physics/latest.json');destination.parent.mkdir(parents=True,exist_ok=True)
    tmp=destination.with_suffix('.tmp');tmp.write_text(json.dumps(result,separators=(',',':'),allow_nan=False));tmp.replace(destination)
    print(json.dumps({'phase':'complete','metrics':result['metrics']}),flush=True);sim.close()
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--duration',type=float,default=2);p.add_argument('--drive',type=float,default=1);p.add_argument('--turn',type=float,default=0);p.add_argument('--silenced',action='store_true');p.add_argument('--no-feedback',action='store_true');p.add_argument('--output');p.add_argument('--vision',action='store_true');p.add_argument('--no-vision-control',action='store_true');p.add_argument('--target-y',type=float,default=4);p.add_argument('--odour',action='store_true');p.add_argument('--no-odour-control',action='store_true');p.add_argument('--obstacles',action='store_true');p.add_argument('--no-avoidance',action='store_true');p.add_argument('--layout',choices=list(LAYOUTS),default='training');p.add_argument('--memory',action='store_true');p.add_argument('--cue-side',choices=['left','right'],default='left');p.add_argument('--reset-memory',action='store_true');p.add_argument('--backflip',action='store_true');a=p.parse_args()
    if not 0.5<=a.duration<=5 or not 0<=a.drive<=2 or not -1<=a.turn<=1: p.error('Parameters outside supported range')
    run(a.duration,a.drive,a.turn,a.silenced,not a.no_feedback,a.output,a.vision,not a.no_vision_control,(12,a.target_y),a.odour,not a.no_odour_control,a.obstacles,not a.no_avoidance,a.layout,a.memory,a.cue_side,a.reset_memory,a.backflip)
