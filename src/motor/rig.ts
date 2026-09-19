import * as THREE from "three";
import { LEGS } from "./types";
import type { MotorPose } from "./types";
/** Procedural reduced-DOF rig: additive joint angles from this neutral shape. */
export function addLegRig(body: THREE.Group) {
  const rigs = LEGS.map((leg, index) => {
    const side = index < 3 ? -1 : 1;
    const number = index % 3;
    const sweep = new THREE.Group();
    sweep.position.set(side * 0.35, 0.3, 0.18 - number * 0.65);
    body.add(sweep);
    const lift = new THREE.Group();
    sweep.add(lift);
    const upper = new THREE.Vector3(side * 0.55, -0.22, 0.25);
    const knee = new THREE.Group();
    knee.position.copy(upper);
    lift.add(knee);
    const lower = new THREE.Vector3(side * 0.43, -0.56, -0.2);
    const foot = new THREE.Vector3(side * 0.33, -0.34, 0.5 - number * 0.25);
    const segment = (
      parent: THREE.Group,
      start: THREE.Vector3,
      end: THREE.Vector3,
      radius: number,
    ) => {
      const delta = end.clone().sub(start);
      const mesh = new THREE.Mesh(
        new THREE.CylinderGeometry(radius * 0.8, radius, delta.length(), 7),
        new THREE.MeshStandardMaterial({ color: "#8b916d", roughness: 0.7 }),
      );
      mesh.position.copy(start).add(end).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(
        new THREE.Vector3(0, 1, 0),
        delta.normalize(),
      );
      parent.add(mesh);
    };
    segment(lift, new THREE.Vector3(), upper, 0.035);
    segment(knee, new THREE.Vector3(), lower, 0.03);
    segment(knee, lower, lower.clone().add(foot), 0.023);
    return { leg, side, sweep, lift, knee };
  });
  return (pose: MotorPose) =>
    rigs.forEach(({ leg, side, sweep, lift, knee }) => {
      sweep.rotation.y = side * pose.joints[`${leg}.sweep`];
      lift.rotation.z = side * pose.joints[`${leg}.lift`];
      knee.rotation.z = -side * pose.joints[`${leg}.knee`];
    });
}
