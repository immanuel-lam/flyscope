import type { Activity } from '../data';
export interface PhysicsRun {
  obstacles?:{frames:{time:number;origin:number[];directions:number[][];distances:number[];turn:number;contacts:number}[];geometry:{position:number[];radius:number;halfHeight:number}[];control:string};
  odour?:{frames:{time:number;antennae:number[][];concentration:number[];turn:number}[];sourcePosition:number[];spreadMm:number;control:string};
  vision?:{frames:{time:number;eyes:string[];redFraction:number[];turn:number}[];target:number[];targetPosition?:number[];sourceSize:number[];control:string};
  schemaVersion:1; rig:'neuromechfly-2.1.0'; datasetId:string; datasetVersion:string;
  times:number[];
  frames:{positions:number[];quaternions:number[];contacts:number;drive:number[];feedback:number[]}[];
  geometry:{name?:string;body:number;vertices:number[];faces:number[];position:number[];quaternion:number[];color:number[]}[];
  bodyNames:string[];
  activity:Activity;
  parameters:{duration:number;drive:number;turn:number;silenced:boolean;feedback:boolean;seed:number;vision?:boolean;visionControl?:boolean;target?:number[];odour?:boolean;odourControl?:boolean;obstacles?:boolean;avoidance?:boolean;layout?:string};
  metrics:{displacementMm:number;finalHeightMm:number;wallSeconds:number;neurons:number;effectiveSignedEdges:number;recordedNeurons:number;maxContacts:number;physicsStepSeconds:number;neuralStepSeconds:number;targetStartDistanceMm?:number;targetFinalDistanceMm?:number;odourStartDistanceMm?:number;odourFinalDistanceMm?:number;obstacleContactSteps?:number;forwardProgressMm?:number};
  provenance:Record<string,string>;
}
export function frameIndex(run:PhysicsRun,time:number){
  let low=0,high=run.times.length-1;
  while(low<high){const mid=Math.ceil((low+high)/2);if(run.times[mid]<=time)low=mid;else high=mid-1;}
  return low;
}

/** Root telemetry only; actual body replay uses PhysicsRun geometry/transforms, never these zero joint slots. */
export function physicsTelemetry(run:PhysicsRun):import('../motor/types').MotorTrack{
  let previous=0;
  const poses=run.frames.map(frame=>{
    const [w,x,y,z]=frame.quaternions;
    let heading=-Math.atan2(2*(w*z+x*y),1-2*(y*y+z*z));
    while(heading-previous>Math.PI)heading-=2*Math.PI;
    while(heading-previous< -Math.PI)heading+=2*Math.PI;
    previous=heading;
    const joints=Object.fromEntries(['LF','LM','LH','RF','RM','RH'].flatMap(l=>['sweep','lift','knee'].map(j=>[`${l}.${j}`,0])).concat([['wing.L',0],['wing.R',0]])) as import('../motor/types').JointAngles;
    return {position:[-frame.positions[1],frame.positions[2],frame.positions[0]] as [number,number,number],heading,joints};
  });
  return {schemaVersion:1,rig:'flyscope-procedural-v1',kind:'simulation',source:'Root telemetry only. Export the PhysicsRun for all physical body transforms.',model:'NeuroMechFly 2.1.0 + MuJoCo 3.9.0',datasetId:run.datasetId,datasetVersion:run.datasetVersion,units:{position:'mm',angle:'rad',time:'s'},times:run.times,poses};
}
