import * as THREE from 'three';
import { frameIndex, type PhysicsRun } from './types';
/** Exact compiled MuJoCo mesh-local transforms and recorded body poses; wxyz -> xyzw. */
export function addPhysicsRig(scene:THREE.Scene,run:PhysicsRun){
  const root=new THREE.Group();
  root.matrixAutoUpdate=false;
  root.matrix.set(0,-1,0,0, 0,0,1,0, 1,0,0,0, 0,0,0,1);
  scene.add(root);
  if(run.vision?.targetPosition){
    const target=new THREE.Mesh(new THREE.SphereGeometry(2,24,16),new THREE.MeshStandardMaterial({color:'#ff1111',roughness:.7}));
    target.position.fromArray(run.vision.targetPosition);root.add(target);
  }
  if(run.odour){
    const marker=new THREE.Mesh(new THREE.RingGeometry(.7,1,32),new THREE.MeshBasicMaterial({color:'#b9db79',side:THREE.DoubleSide}));
    marker.position.fromArray(run.odour.sourcePosition);marker.position.z+=.02;root.add(marker);
  }
  for(const obstacle of run.obstacles?.geometry??[]){
    const mesh=new THREE.Mesh(new THREE.CylinderGeometry(obstacle.radius,obstacle.radius,obstacle.halfHeight*2,32),new THREE.MeshStandardMaterial({color:'#4d80b3',roughness:.8}));
    mesh.rotation.x=Math.PI/2;mesh.position.fromArray(obstacle.position);root.add(mesh);
  }
  const bodies=run.bodyNames.map(()=>{const g=new THREE.Group();root.add(g);return g;});
  let headMesh:THREE.Mesh | undefined;
  const headCenter=new THREE.Vector3();
  for(const item of run.geometry){
    const geometry=new THREE.BufferGeometry();
    geometry.setAttribute('position',new THREE.Float32BufferAttribute(item.vertices,3));
    geometry.setIndex(item.faces);geometry.computeVertexNormals();
    const name=item.name ?? run.bodyNames[item.body];
    const material=new THREE.MeshStandardMaterial({color:name.includes('eye')?'#994c3e':name.includes('wing')?'#ccd6c3':'#9caa78',roughness:.7,side:THREE.DoubleSide,transparent:name.includes('wing')||name==='c_head',opacity:name==='c_head'?.35:name.includes('wing')?.35:1});
    const mesh=new THREE.Mesh(geometry,material);mesh.position.fromArray(item.position);
    mesh.quaternion.set(item.quaternion[1],item.quaternion[2],item.quaternion[3],item.quaternion[0]);
    bodies[item.body].add(mesh);
    if(name==='c_head'){headMesh=mesh;geometry.computeBoundingBox();geometry.boundingBox!.getCenter(headCenter);}
  }
  const qa=new THREE.Quaternion(),qb=new THREE.Quaternion();
  const update=(time:number,visible:boolean)=>{
    root.visible=visible;
    const i=frameIndex(run,time),j=Math.min(i+1,run.frames.length-1),a=run.frames[i],b=run.frames[j];
    const t=i===j?0:THREE.MathUtils.clamp((time-run.times[i])/(run.times[j]-run.times[i]),0,1);
    bodies.forEach((body,k)=>{
      body.position.set(...[0,1,2].map(axis=>THREE.MathUtils.lerp(a.positions[k*3+axis],b.positions[k*3+axis],t)) as [number,number,number]);
      const o=k*4;qa.set(a.quaternions[o+1],a.quaternions[o+2],a.quaternions[o+3],a.quaternions[o]);qb.set(b.quaternions[o+1],b.quaternions[o+2],b.quaternions[o+3],b.quaternions[o]);body.quaternion.copy(qa).slerp(qb,t);
    });
    root.updateMatrixWorld(true);
    return {position:new THREE.Vector3(-bodies[0].position.y,bodies[0].position.z,bodies[0].position.x),head:headMesh?.localToWorld(headCenter.clone())};
  };
  return {update};
}
