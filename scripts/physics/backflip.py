"""Neural-gated external-force backflip assist, not a learned biological skill."""
import numpy as np
import mujoco

class BackflipAssist:
    def __init__(self,sim,thorax_id):
        self.sim=sim;self.body=thorax_id;self.mass=float(sim.mj_model.body_mass.sum())
        self.start=None;self.position=None;self.orientation=None;self.duration=.9
    def step(self,time,neural_drive):
        m,d=self.sim.mj_model,self.sim.mj_data
        d.xfrc_applied[self.body]=0
        if self.start is None and time>=.6 and np.mean(neural_drive)>.1:
            self.start=time;self.position=d.xpos[self.body].copy();self.orientation=d.xquat[self.body].copy()
        if self.start is None or time>self.start+self.duration:
            return {'active':False,'force':[0.,0.,0.],'torque':[0.,0.,0.],'targetAngle':0. if self.start is None else -2*np.pi}
        u=np.clip((time-self.start)/self.duration,0,1);T=self.duration
        blend=10*u**3-15*u**4+6*u**5
        angle=-2*np.pi*blend
        rotation=np.array([np.cos(angle/2),0,np.sin(angle/2),0.])
        desired=np.zeros(4);mujoco.mju_mulQuat(desired,self.orientation,rotation)
        inverse=d.xquat[self.body].copy();inverse[1:]*=-1
        error=np.zeros(4);mujoco.mju_mulQuat(error,desired,inverse)
        if error[0]<0:error=-error
        norm=np.linalg.norm(error[1:]);rotvec=error[1:]*(2*np.arctan2(norm,error[0])/norm) if norm>1e-12 else np.zeros(3)
        velocity=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,self.body,velocity,0)
        target=self.position.copy();target[2]+=8*np.sin(np.pi*u)**2
        target_velocity=np.array([0,0,8*np.pi/T*np.sin(2*np.pi*u)])
        target_acceleration=np.array([0,0,16*np.pi**2/T**2*np.cos(2*np.pi*u)])
        force=self.mass*(target_acceleration+40000*(target-d.xpos[self.body])+400*(target_velocity-velocity[3:])-m.opt.gravity)
        torque=.002*(10000*rotvec-200*velocity[:3])
        force=np.clip(force,-50,50);torque=np.clip(torque,-20,20)
        d.xfrc_applied[self.body,:3]=force;d.xfrc_applied[self.body,3:]=torque
        return {'active':True,'force':force.tolist(),'torque':torque.tolist(),'targetAngle':float(angle)}
