import { useEffect, useRef } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { addLegRig } from "./motor/rig";
import type { MotorPose } from "./motor/types";
import type { Detail } from "./large/types";
import type { Dataset, Vec3 } from "./data";
import { activityAt, activityRange, palette } from "./data";

import {
  recordedPeaks,
  relativeRate,
  activityPointOrder,
} from "./physics/activity-display";
import { addPhysicsRig } from "./physics/rig";
import type { PhysicsRun } from "./physics/types";

type Props = {
  activityVisible: boolean;
  physics?: PhysicsRun;
  dataset: Dataset;
  detail?: Detail;
  connectionEdges?: { source: string; target: string; weight: number }[];
  motorPose: MotorPose;
  time: number;
  selected: string | null;
  region: string;
  threshold: number;
  showEdges: boolean;
  showBody: boolean;
  mode: "brain" | "fly";
  reset: number;
  onSelect: (id: string) => void;
};
export default function Scene(props: Props) {
  const host = useRef<HTMLDivElement>(null);
  const current = useRef(props);
  current.current = props;
  useEffect(() => {
    const el = host.current!;
    const d = {
      ...props.dataset,
      neurons: props.dataset.neurons.filter((n) => n.positionKnown !== false),
    };
    const full = props.dataset.id === "male-cns-v1-full";
    let renderer: THREE.WebGLRenderer;
    try {
      renderer = new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
        powerPreference: "high-performance",
      });
    } catch {
      el.textContent =
        "WebGL is unavailable. Use the neuron list and data controls, or open this viewer in a WebGL-enabled browser.";
      return;
    }
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.setClearColor("#151b18", 0);
    el.appendChild(renderer.domElement);
    const scene = new THREE.Scene();
    const camera = new THREE.PerspectiveCamera(38, 1, 0.01, 100);
    camera.position.set(
      props.mode === "fly" ? 6 : 0,
      props.mode === "fly" ? 4 : 1,
      props.mode === "fly" ? 7 : 11,
    );
    let dirty = true;
    let interactionUntil = 0;
    let lastMotorSignature = "";
    let renderedFrames = 0;
    const controls = new OrbitControls(camera, renderer.domElement);
    controls.addEventListener("change", () => {
      dirty = true;
    });
    controls.addEventListener("start", () => {
      interactionUntil = performance.now() + 300;
      dirty = true;
    });
    const markInteraction = () => {
      interactionUntil = performance.now() + 200;
      dirty = true;
    };
    renderer.domElement.addEventListener("pointermove", markInteraction);
    renderer.domElement.addEventListener("wheel", markInteraction);
    controls.enableDamping = true;
    controls.minDistance = full ? 0.025 : 2;
    controls.maxDistance = 25;
    scene.add(new THREE.AmbientLight("#d9e1cd", 1.8));
    const light = new THREE.DirectionalLight("#f0e3c4", 3);
    light.position.set(3, 7, 5);
    scene.add(light);
    const group = new THREE.Group();
    scene.add(group);
    const brain = new THREE.Group();
    group.add(brain);
    const regions = [...new Set(d.neurons.map((n) => n.region))];
    const bounds = new THREE.Box3();
    d.neurons.forEach((n) => {
      bounds.expandByPoint(new THREE.Vector3(...n.position));
      if (n.skeleton)
        for (let i = 0; i < n.skeleton.length; i += 3)
          bounds.expandByPoint(
            new THREE.Vector3(
              n.skeleton[i],
              n.skeleton[i + 1],
              n.skeleton[i + 2],
            ),
          );
    });
    const center = bounds.getCenter(new THREE.Vector3());
    const extent = bounds.getSize(new THREE.Vector3());
    const scale = 5.5 / Math.max(extent.x, extent.y, extent.z, 1e-6);
    const norm = (p: Vec3) => {
      const v = new THREE.Vector3(...p).sub(center).multiplyScalar(scale);
      return full ? new THREE.Vector3(v.x, -v.z, v.y) : v;
    };
    const positions = new Float32Array(d.neurons.length * 3);
    const colors = new Float32Array(d.neurons.length * 3);
    const linePositions: number[] = [];
    const lineColors: number[] = [];
    const owner: number[] = [];
    const bases = d.neurons.map(
      (n) =>
        new THREE.Color(palette[regions.indexOf(n.region) % palette.length]),
    );
    d.neurons.forEach((n, i) => {
      norm(n.position).toArray(positions, i * 3);
      bases[i].toArray(colors, i * 3);
      if (n.skeleton)
        for (let j = 0; j < n.skeleton.length; j += 3) {
          linePositions.push(
            ...norm(n.skeleton.slice(j, j + 3) as Vec3).toArray(),
          );
          lineColors.push(...bases[i].toArray());
          owner.push(i);
        }
    });
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
    const focusActivity = !!props.physics && !!d.activity;
    const peaks = recordedPeaks(focusActivity ? d.activity : undefined);
    if (full) geometry.setIndex(activityPointOrder(d.neurons, peaks, 20000));
    const points = new THREE.Points(
      geometry,
      new THREE.PointsMaterial({
        size: full
          ? props.mode === "fly"
            ? 0.045
            : 0.012
          : props.mode === "fly"
            ? 0.12
            : 0.075,
        vertexColors: true,
        sizeAttenuation: true,
        transparent: true,
        opacity: 0.95,
      }),
    );
    brain.add(points);
    // A separate overlay preserves all recorded cells, even when structural LOD is active.
    const recordedIndices = d.neurons.flatMap((n, i) =>
      peaks.has(n.id) ? [i] : [],
    );
    const glowGeometry = new THREE.BufferGeometry();
    glowGeometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(
        recordedIndices.flatMap((i) =>
          Array.from(positions.slice(i * 3, i * 3 + 3)),
        ),
        3,
      ),
    );
    const strengths = new Float32Array(recordedIndices.length);
    glowGeometry.setAttribute(
      "strength",
      new THREE.BufferAttribute(strengths, 1),
    );
    const glow = new THREE.Points(
      glowGeometry,
      new THREE.ShaderMaterial({
        transparent: true,
        depthWrite: false,
        depthTest: false,
        blending: THREE.AdditiveBlending,
        uniforms: {
          pixelRatio: {
            value:
              renderer.getPixelRatio() * (props.mode === "fly" ? 0.35 : 0.7),
          },
        },
        vertexShader: `attribute float strength; varying float intensity; uniform float pixelRatio;
        void main(){intensity=strength;gl_Position=projectionMatrix*modelViewMatrix*vec4(position,1.0);gl_PointSize=(4.0+10.0*strength)*pixelRatio;}`,
        fragmentShader: `varying float intensity;
        void main(){float radius=length(gl_PointCoord-vec2(0.5))*2.0;if(radius>1.0||intensity<=0.0)discard;
        float halo=pow(1.0-radius,2.0);gl_FragColor=vec4(mix(vec3(0.4,0.8,0.18),vec3(1.0,0.86,0.38),intensity),halo*intensity*0.45);}`,
      }),
    );
    glow.renderOrder = 10;
    brain.add(glow);
    renderer.domElement.dataset.recordedCells = String(recordedIndices.length);

    const lineGeo = new THREE.BufferGeometry();
    lineGeo.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(linePositions, 3),
    );
    lineGeo.setAttribute(
      "color",
      new THREE.Float32BufferAttribute(lineColors, 3),
    );
    const lines = new THREE.LineSegments(
      lineGeo,
      new THREE.LineBasicMaterial({
        vertexColors: true,
        transparent: true,
        opacity: 0.62,
      }),
    );
    brain.add(lines);
    const detailLines = new THREE.LineSegments(
      new THREE.BufferGeometry(),
      new THREE.LineBasicMaterial({
        color: "#ffe7a0",
        transparent: true,
        opacity: 0.95,
      }),
    );
    brain.add(detailLines);
    let lastDetail: Detail | undefined;
    let lastConnections: Props["connectionEdges"];
    const byId = new Map(d.neurons.map((n, i) => [n.id, i]));
    const edgePositions: number[] = [];
    d.edges.forEach((e) => {
      edgePositions.push(
        ...norm(d.neurons[byId.get(e.source)!].position).toArray(),
        ...norm(d.neurons[byId.get(e.target)!].position).toArray(),
      );
    });
    const edgeGeo = new THREE.BufferGeometry();
    edgeGeo.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(edgePositions, 3),
    );
    const edges = new THREE.LineSegments(
      edgeGeo,
      new THREE.LineBasicMaterial({
        color: "#b6cf71",
        transparent: true,
        opacity: 0.12,
      }),
    );
    brain.add(edges);
    const highlight = new THREE.Mesh(
      new THREE.SphereGeometry(full ? 0.025 : 0.105, 16, 12),
      new THREE.MeshBasicMaterial({ color: "#f8eeb8", wireframe: true }),
    );
    brain.add(highlight);
    const body = new THREE.Group();
    group.add(body);
    const sphere = (pos: Vec3, size: Vec3, color: string, opacity = 1) => {
      const mesh = new THREE.Mesh(
        new THREE.SphereGeometry(1, 40, 28),
        new THREE.MeshStandardMaterial({
          color,
          roughness: 0.58,
          metalness: 0.18,
          transparent: opacity < 1,
          opacity,
          depthWrite: opacity === 1,
        }),
      );
      mesh.position.set(...pos);
      mesh.scale.set(...size);
      body.add(mesh);
      return mesh;
    };
    const tube = (pts: Vec3[], radius: number, color: string) => {
      const curve = new THREE.CatmullRomCurve3(
        pts.map((p) => new THREE.Vector3(...p)),
      );
      const m = new THREE.Mesh(
        new THREE.TubeGeometry(curve, 28, radius, 5, false),
        new THREE.MeshStandardMaterial({ color, roughness: 0.7 }),
      );
      body.add(m);
      return m;
    };
    let updateLegs: ((pose: MotorPose) => void) | undefined;
    const wingPivots: THREE.Group[] = [];
    let floor: THREE.GridHelper | undefined;
    if (props.mode === "fly") {
      updateLegs = addLegRig(body);
      sphere([0, 0.45, -0.4], [0.52, 0.55, 0.82], "#747a5a");
      sphere([0, 0.18, -1.55], [0.55, 0.45, 1.05], "#a09057");
      sphere([0, 0.64, 0.65], [0.67, 0.55, 0.5], "#718365", 0.3);
      for (const side of [-1, 1]) {
        sphere([side * 0.48, 0.69, 0.77], [0.27, 0.43, 0.32], "#9e5345", 0.72);
        tube(
          [
            [side * 0.19, 0.93, 1],
            [side * 0.29, 1.12, 1.32],
            [side * 0.44, 1.18, 1.4],
          ],
          0.025,
          "#a8ad8a",
        );
        const wing = sphere(
          [side * 1.08, 0.6, -0.98],
          [0.59, 0.025, 1.6],
          "#ccd6c3",
          0.18,
        );
        wing.rotation.y = side * -0.42;
        const wingPivot = new THREE.Group();
        wingPivot.position.set(side * 0.3, 0.6, -0.2);
        body.add(wingPivot);
        wingPivots.push(wingPivot);
        const wingWorld = wing.position.clone();
        wing.position.sub(wingPivot.position);
        wingPivot.add(wing);
        for (let vein = 0; vein < 5; vein++) {
          const a = -0.9 + vein * 0.45;
          const local: Vec3[] = [
            [0, 0.03, 1.4],
            [Math.sin(a) * 0.32, 0.035, 0],
            [Math.sin(a) * 0.48, 0.03, -1.15],
          ];
          const path = local.map((p) => {
            const v = new THREE.Vector3(...p)
              .applyAxisAngle(new THREE.Vector3(0, 1, 0), side * -0.42)
              .add(wingWorld);
            return v.toArray() as Vec3;
          });
          const veinMesh = tube(path, 0.006, "#8b9b85");
          veinMesh.position.sub(wingPivot.position);
          wingPivot.add(veinMesh);
        }
      }
      for (let i = 0; i < 5; i++) {
        const ring = new THREE.Mesh(
          new THREE.TorusGeometry(0.45 - i * 0.018, 0.018, 5, 48),
          new THREE.MeshStandardMaterial({ color: "#504f39" }),
        );
        ring.position.set(0, 0.18, -1.1 - i * 0.24);
        ring.scale.y = 0.78;
        body.add(ring);
      }
      brain.scale.setScalar(0.17);
      brain.position.set(0, 0.71, 0.78);
      brain.rotation.x = -0.3;
      const grid = new THREE.GridHelper(12, 24, "#394439", "#27312b");
      grid.position.y = -0.86;
      floor = grid;
      scene.add(grid);
      controls.target.set(0, 0.05, -0.45);
    }
    const physical =
      props.mode === "fly" && props.physics
        ? addPhysicsRig(scene, props.physics)
        : undefined;
    if (physical && floor) {
      floor.position.y = 0;
      camera.position.set(9, 6, 10);
      controls.target.set(0, 0, 0);
    }
    const raycaster = new THREE.Raycaster();
    raycaster.params.Points!.threshold = 0.1;
    const pointer = new THREE.Vector2();
    let down = [0, 0];
    const onDown = (e: PointerEvent) => {
      down = [e.clientX, e.clientY];
    };
    const onClick = (e: PointerEvent) => {
      if (Math.hypot(e.clientX - down[0], e.clientY - down[1]) > 5) return;
      const rect = el.getBoundingClientRect();
      pointer.set(
        ((e.clientX - rect.left) / rect.width) * 2 - 1,
        (-(e.clientY - rect.top) / rect.height) * 2 + 1,
      );
      raycaster.setFromCamera(pointer, camera);
      const hit = raycaster
        .intersectObject(points)
        .find((h) => h.index !== undefined && visible[h.index]);
      if (hit?.index !== undefined)
        current.current.onSelect(d.neurons[hit.index].id);
    };
    renderer.domElement.addEventListener("pointerdown", onDown);
    renderer.domElement.addEventListener("pointerup", onClick);
    const fitCamera = () => {
      const target =
        props.mode === "fly"
          ? group.position.clone().add(new THREE.Vector3(0, 0.05, -0.45))
          : new THREE.Vector3();
      const direction =
        props.mode === "fly"
          ? new THREE.Vector3(6, 4, 7).normalize()
          : new THREE.Vector3(0, 0.08, 1).normalize();
      const radius = props.mode === "fly" ? (physical ? 4.0 : 2.6) : 3.1;
      const distance =
        radius /
        Math.tan(THREE.MathUtils.degToRad(19)) /
        Math.min(camera.aspect, 1);
      camera.position.copy(target).addScaledVector(direction, distance);
      controls.target.copy(target);
    };
    const resize = new ResizeObserver(() => {
      const w = el.clientWidth,
        h = el.clientHeight;
      if (w && h) {
        renderer.setSize(w, h);
        camera.aspect = w / h;
        camera.updateProjectionMatrix();
        fitCamera();
        dirty = true;
      }
    });
    resize.observe(el);
    const range = activityRange(d);
    const muted = new THREE.Color("#1b2520");
    const white = new THREE.Color("#ffe7a0");
    const visible: boolean[] = [];
    let frame = 0;
    let lastState = "";
    let lastReset = props.reset;
    const render = () => {
      frame = requestAnimationFrame(render);
      const p = current.current;
      if (p.reset !== lastReset) {
        dirty = true;
        fitCamera();
        lastReset = p.reset;
      }
      const state = [
        p.activityVisible && d.activity ? p.time : 0,
        p.activityVisible,
        p.region,
        p.selected,
        p.threshold,
        p.showEdges,
        p.showBody,
      ].join("|");
      if (p.detail !== lastDetail) {
        dirty = true;
        lastDetail = p.detail;
        detailLines.geometry.dispose();
        const geo = new THREE.BufferGeometry();
        if (p.detail) {
          const a = p.detail.segments;
          const out = new Float32Array(a.length);
          for (let i = 0; i < a.length; i += 3)
            norm([a[i], a[i + 1], a[i + 2]]).toArray(out, i);
          geo.setAttribute("position", new THREE.BufferAttribute(out, 3));
        }
        detailLines.geometry = geo;
      }
      if (p.connectionEdges !== lastConnections) {
        dirty = true;
        lastConnections = p.connectionEdges;
        const positions: number[] = [];
        for (const e of p.connectionEdges ?? []) {
          const a = byId.get(e.source),
            b = byId.get(e.target);
          if (a !== undefined && b !== undefined)
            positions.push(
              ...norm(d.neurons[a].position).toArray(),
              ...norm(d.neurons[b].position).toArray(),
            );
        }
        edges.geometry.dispose();
        edges.geometry = new THREE.BufferGeometry();
        edges.geometry.setAttribute(
          "position",
          new THREE.Float32BufferAttribute(positions, 3),
        );
      }
      if (state !== lastState) {
        dirty = true;
        lastState = state;
        const c = new THREE.Color();
        d.neurons.forEach((n, i) => {
          const raw = p.activityVisible
            ? activityAt(d, n.id, p.time)
            : undefined;
          const value =
            raw === undefined ? 0 : (raw - range[0]) / (range[1] - range[0]);
          const show =
            (p.region === "All regions" || p.region === n.region) &&
            (!p.activityVisible ||
              p.threshold === 0 ||
              (raw !== undefined && value >= p.threshold));
          visible[i] = show;
          c.copy(show ? bases[i] : muted);

          if (show && raw !== undefined)
            c.multiplyScalar(0.3 + value * 0.7).lerp(white, value * 0.6);
          if (n.id === p.selected && show) c.copy(white);
          c.toArray(colors, i * 3);
        });
        glow.visible = p.activityVisible;
        renderer.domElement.dataset.activityVisible = String(p.activityVisible);
        let totalStrength = 0;
        recordedIndices.forEach((ni, j) => {
          const n = d.neurons[ni];
          strengths[j] =
            p.activityVisible && visible[ni]
              ? relativeRate(activityAt(d, n.id, p.time), peaks.get(n.id)!)
              : 0;
          totalStrength += strengths[j];
        });
        glowGeometry.attributes.strength.needsUpdate = true;
        renderer.domElement.dataset.activityStrength = totalStrength.toFixed(5);
        geometry.attributes.color.needsUpdate = true;
        const lc = lineGeo.attributes.color.array as Float32Array;
        owner.forEach((ni, i) => {
          lc[i * 3] = colors[ni * 3];
          lc[i * 3 + 1] = colors[ni * 3 + 1];
          lc[i * 3 + 2] = colors[ni * 3 + 2];
        });
        lineGeo.attributes.color.needsUpdate = true;
        const index = p.selected ? byId.get(p.selected) : undefined;
        highlight.visible = index !== undefined && visible[index];
        if (index !== undefined)
          highlight.position.copy(norm(d.neurons[index].position));
        edges.visible =
          p.showEdges && p.region === "All regions" && p.threshold === 0;
        body.visible = p.showBody && !physical;
      }
      const motorSignature =
        JSON.stringify(p.motorPose) + (physical ? p.time : "");
      if (props.mode === "fly" && motorSignature !== lastMotorSignature) {
        dirty = true;
        lastMotorSignature = motorSignature;
      }
      if (props.mode === "fly" && dirty) {
        const physicalPose = physical?.update(p.time, p.showBody);
        const position =
          physicalPose?.position ?? new THREE.Vector3(...p.motorPose.position);
        const delta = position.clone().sub(group.position);
        group.position.copy(position);
        group.rotation.y = p.motorPose.heading;
        if (physicalPose?.head) {
          group.updateMatrixWorld(true);
          brain.position.copy(group.worldToLocal(physicalPose.head));
        }
        camera.position.add(delta);
        controls.target.add(delta);
        updateLegs?.(p.motorPose);
        wingPivots.forEach((wing, index) => {
          wing.rotation.z =
            (index === 0 ? -1 : 1) *
            p.motorPose.joints[index === 0 ? "wing.L" : "wing.R"];
        });
        if (floor) {
          floor.position.x = Math.round(position.x / 6) * 6;
          floor.position.z = Math.round(position.z / 6) * 6;
        }
      }
      const budget = full
        ? props.mode === "fly"
          ? 20000
          : performance.now() < interactionUntil
            ? 20000
            : d.neurons.length
        : d.neurons.length;
      if (geometry.drawRange.count !== budget) {
        geometry.setDrawRange(0, budget);
        dirty = true;
      }
      controls.update();
      if (!dirty) return;
      dirty = false;
      renderer.render(scene, camera);
      renderer.domElement.dataset.renderedFrames = String(++renderedFrames);
      renderer.domElement.dataset.drawCalls = String(
        renderer.info.render.calls,
      );
      renderer.domElement.dataset.points = String(renderer.info.render.points);
      renderer.domElement.dataset.lines = String(renderer.info.render.lines);
      renderer.domElement.dataset.geometries = String(
        renderer.info.memory.geometries,
      );
    };
    render();
    return () => {
      cancelAnimationFrame(frame);
      resize.disconnect();
      controls.dispose();
      renderer.domElement.removeEventListener("pointermove", markInteraction);
      renderer.domElement.removeEventListener("wheel", markInteraction);
      renderer.domElement.removeEventListener("pointerdown", onDown);
      renderer.domElement.removeEventListener("pointerup", onClick);
      scene.traverse((o) => {
        const m = o as THREE.Mesh;
        if (m.geometry) m.geometry.dispose();
        if (m.material) {
          for (const mat of Array.isArray(m.material)
            ? m.material
            : [m.material])
            mat.dispose();
        }
      });
      renderer.dispose();
      renderer.domElement.remove();
    };
  }, [props.dataset, props.mode, props.physics]);
  return (
    <div
      className="scene-canvas"
      ref={host}
      aria-label={
        props.mode === "brain"
          ? "Interactive 3D neuron view. Drag to orbit; scroll to zoom. Select neurons with the list."
          : "Interactive 3D fly with illustrative brain placement. Drag to orbit; scroll to zoom."
      }
      role="img"
    />
  );
}
