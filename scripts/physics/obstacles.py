"""Physical obstacles and artificial range sensors; no biological sensor claim."""
import numpy as np
import mujoco
from flygym.utils.mjcf import GEOM_TYPES

LAYOUTS={'training':[(8,1,1.2)],'offset':[(9,-.5,1.2)],'wide':[(10,2,1.6)],'heldout-left':[(11,2.5,1.3)],'heldout-right':[(8,-1.5,1.4)]}

def add_obstacles(world,layout):
    for i,(x,y,radius) in enumerate(LAYOUTS[layout]):
        obstacle=world.mjcf_root.worldbody.add_geom(name=f'avoidance_obstacle_{i}',type=GEOM_TYPES['cylinder'],pos=[x,y,2],size=[radius,2,0],rgba=[.3,.5,.7,1],group=5,contype=1,conaffinity=1)
        # FlyGym uses explicit body/terrain pairs; bit masks alone do not enable contact.
        world.ground_geoms.append(obstacle)

class RangeSensor:
    def __init__(self,sim,head_geom_id,thorax_id):
        if head_geom_id<0 or thorax_id<0:raise ValueError('Missing head geometry or thorax body')
        self.sim=sim;self.head=head_geom_id;self.thorax=thorax_id
        self.angles=np.linspace(-np.pi/3,np.pi/3, 9)
        self.maximum=8.;self.direction=0;self.clear_steps=0;self.last_turn=0.
        self.obstacle_ids={i for i in range(sim.mj_model.ngeom) if (mujoco.mj_id2name(sim.mj_model,mujoco.mjtObj.mjOBJ_GEOM,i) or '').startswith('avoidance_obstacle_')}
        self.group=np.array([0,0,0,0,0,1],dtype=np.uint8)
    def observe(self):
        model,data=self.sim.mj_model,self.sim.mj_data
        origin=data.geom_xpos[self.head].copy();forward=data.xmat[self.thorax].reshape(3,3)[:,0]
        yaw=np.arctan2(forward[1],forward[0]);distances=[];directions=[]
        for angle in self.angles:
            direction=np.array([np.cos(yaw+angle),np.sin(yaw+angle),0.]);hit=np.array([-1],dtype=np.int32)
            distance=mujoco.mj_ray(model,data,origin,direction,self.group,True,-1,hit)
            distances.append(min(distance,self.maximum) if distance>=0 else self.maximum);directions.append(direction.tolist())
        proximity=np.maximum(0,1-np.array(distances)/self.maximum)
        # Choose away from the obstructed side; retain choice until all rays clear.
        if proximity.max()<.02:
            self.clear_steps+=1
            if self.clear_steps>=50:self.direction=0;self.last_turn=0.
            turn=self.last_turn
        else:
            self.clear_steps=0
            if not self.direction:self.direction=1 if proximity[5:].sum()>=proximity[:4].sum() else -1
            turn=self.direction*min(1.,proximity.max()*3);self.last_turn=turn
        contacts=sum(int(c.geom1 in self.obstacle_ids or c.geom2 in self.obstacle_ids) for c in data.contact)
        return {'origin':origin.tolist(),'directions':directions,'distances':distances,'turn':turn,'contacts':contacts}
