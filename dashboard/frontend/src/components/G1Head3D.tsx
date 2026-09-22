import {
  useEffect,
  useRef,
  useState,
} from "react";

import * as THREE from "three";

import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";


interface LinkTransform {
  xyz: readonly [number, number, number];
  rpy: readonly [number, number, number];
}


const TORSO_TRANSFORM: LinkTransform = {
  xyz: [0, 0, 0],
  rpy: [0, 0, 0],
};


/*
 * Matches the head/logo geometry placement in
 * simulation/vendor/unitree_g1/g1_29dof.xml.
 */
const HEAD_TRANSFORM: LinkTransform = {
  xyz: [
    0.0039635,
    0.0,
    -0.054,
  ],

  rpy: [
    0.0,
    0.0,
    0.0,
  ],
};


function applyTransform(
  object: THREE.Object3D,
  transform: LinkTransform,
) {
  object.position.set(
    transform.xyz[0],
    transform.xyz[1],
    transform.xyz[2],
  );

  object.rotation.set(
    transform.rpy[0],
    transform.rpy[1],
    transform.rpy[2],
  );
}


export function G1Head3D() {
  const mountRef =
    useRef<HTMLDivElement>(null);

  const [loadState, setLoadState] =
    useState<
      "loading" | "ready" | "error"
    >("loading");

  useEffect(() => {
    const mount = mountRef.current;

    if (!mount) return;

    let cancelled = false;

    const scene =
      new THREE.Scene();

    const camera =
      new THREE.PerspectiveCamera(
        29,
        1,
        0.01,
        100,
      );

    camera.position.set(
      0,
      0,
      3.0,
    );

    const renderer =
      new THREE.WebGLRenderer({
        antialias: true,
        alpha: true,
      });

    renderer.setPixelRatio(
      Math.min(
        window.devicePixelRatio,
        1.5,
      ),
    );

    renderer.outputColorSpace =
      THREE.SRGBColorSpace;

    renderer.toneMapping =
      THREE.ACESFilmicToneMapping;

    renderer.toneMappingExposure =
      1.05;

    renderer.setClearColor(
      0x000000,
      0,
    );

    mount.appendChild(
      renderer.domElement,
    );


    // ========================================================
    // LIGHTING
    // ========================================================

    scene.add(
      new THREE.HemisphereLight(
        0xe8eef7,
        0x202731,
        1.5,
      ),
    );

    const keyLight =
      new THREE.DirectionalLight(
        0xffffff,
        3.0,
      );

    keyLight.position.set(
      4,
      5,
      5,
    );

    scene.add(
      keyLight,
    );


    const fillLight =
      new THREE.DirectionalLight(
        0x7ea7e8,
        1.15,
      );

    fillLight.position.set(
      -5,
      1,
      3,
    );

    scene.add(
      fillLight,
    );


    const rimLight =
      new THREE.DirectionalLight(
        0xaec9ff,
        1.4,
      );

    rimLight.position.set(
      1,
      4,
      -4,
    );

    scene.add(
      rimLight,
    );


    // ========================================================
    // ASSEMBLY
    // ========================================================

    const presentation =
      new THREE.Group();

    const assembly =
      new THREE.Group();

    presentation.add(
      assembly,
    );

    scene.add(
      presentation,
    );


    // ========================================================
    // MATERIALS
    // ========================================================

    const headMaterial =
      new THREE.MeshPhysicalMaterial({
        color: 0x242a30,

        metalness: 0.08,
        roughness: 0.30,

        clearcoat: 0.22,
        clearcoatRoughness: 0.38,
      });


    const torsoMaterial =
      new THREE.MeshPhysicalMaterial({
        color: 0xd8dde2,

        metalness: 0.04,
        roughness: 0.39,

        clearcoat: 0.16,
        clearcoatRoughness: 0.48,
      });


    const logoMaterial =
      new THREE.MeshStandardMaterial({
        color: 0x24282d,

        metalness: 0.16,
        roughness: 0.35,
      });


    const loader =
      new STLLoader();

    const geometries:
      THREE.BufferGeometry[] = [];


    async function loadMesh(
      url: string,
      material: THREE.Material,
      transform: LinkTransform,
    ) {
      const geometry =
        await loader.loadAsync(
          url,
        );

      geometry.computeVertexNormals();

      geometries.push(
        geometry,
      );

      const mesh =
        new THREE.Mesh(
          geometry,
          material,
        );

      applyTransform(
        mesh,
        transform,
      );

      assembly.add(
        mesh,
      );

      return mesh;
    }


    async function buildRobot() {
      try {
        await Promise.all([
          loadMesh(
            "/models/g1/torso_link.STL",
            torsoMaterial,
            TORSO_TRANSFORM,
          ),

          loadMesh(
            "/models/g1/head_link.STL",
            headMaterial,
            HEAD_TRANSFORM,
          ),

          loadMesh(
            "/models/g1/logo_link.STL",
            logoMaterial,
            HEAD_TRANSFORM,
          ),
        ]);

        if (cancelled) {
          return;
        }

        assembly.updateMatrixWorld(
          true,
        );


        // ====================================================
        // FIT THE REAL ASSEMBLY INTO THE VIEW
        // ====================================================

        const box =
          new THREE.Box3().setFromObject(
            assembly,
          );

        const center =
          box.getCenter(
            new THREE.Vector3(),
          );

        const size =
          box.getSize(
            new THREE.Vector3(),
          );

        const maxDimension =
          Math.max(
            size.x,
            size.y,
            size.z,
          );

        if (
          !Number.isFinite(
            maxDimension,
          ) ||
          maxDimension <= 0
        ) {
          throw new Error(
            "Invalid G1 assembly bounds",
          );
        }


        /*
         * Center the complete robot bust while
         * preserving all inter-link geometry.
         */
        assembly.position.set(
          -center.x,
          -center.y,
          -center.z,
        );


        const scale =
          1.82 / maxDimension;

        presentation.scale.setScalar(
          scale,
        );


        /*
         * Same presentation orientation that already
         * produced the correct frontal head view.
         */
        presentation.rotation.set(
          -Math.PI / 2,
          0,
          -Math.PI / 2,
        );


        /*
         * Lift the bust slightly inside the viewport.
         */
        presentation.position.y =
          0.04;


        setLoadState(
          "ready",
        );
      } catch (error) {
        console.error(
          "Failed to construct G1 model:",
          error,
        );

        if (!cancelled) {
          setLoadState(
            "error",
          );
        }
      }
    }


    void buildRobot();


    // ========================================================
    // INTERACTION
    // ========================================================

    const controls =
      new OrbitControls(
        camera,
        renderer.domElement,
      );

    controls.enablePan =
      false;

    controls.enableZoom =
      false;

    controls.enableDamping =
      true;

    controls.dampingFactor =
      0.075;

    controls.rotateSpeed =
      0.55;


    /*
     * Keep rotation useful rather than allowing
     * the model to be flipped upside down.
     */
    controls.minPolarAngle =
      Math.PI * 0.25;

    controls.maxPolarAngle =
      Math.PI * 0.75;


    // ========================================================
    // RESPONSIVE CANVAS
    // ========================================================

    const resize = () => {
      const width =
        mount.clientWidth;

      const height =
        mount.clientHeight;

      if (
        width < 1 ||
        height < 1
      ) {
        return;
      }

      renderer.setSize(
        width,
        height,
        false,
      );

      camera.aspect =
        width / height;

      camera.updateProjectionMatrix();
    };


    const observer =
      new ResizeObserver(
        resize,
      );

    observer.observe(
      mount,
    );

    resize();


    // ========================================================
    // RENDER LOOP
    // ========================================================

    let frame = 0;

    const render = () => {
      frame =
        requestAnimationFrame(
          render,
        );

      controls.update();

      renderer.render(
        scene,
        camera,
      );
    };

    render();


    // ========================================================
    // CLEANUP
    // ========================================================

    return () => {
      cancelled = true;

      cancelAnimationFrame(
        frame,
      );

      observer.disconnect();

      controls.dispose();

      geometries.forEach(
        (geometry) =>
          geometry.dispose(),
      );

      headMaterial.dispose();
      torsoMaterial.dispose();
      logoMaterial.dispose();

      renderer.dispose();

      renderer.domElement.remove();
    };
  }, []);


  return (
    <div className="g1-head-3d g1-bust-3d">
      <div className="g1-model-backdrop">
        <span className="model-orbit orbit-1" />
        <span className="model-orbit orbit-2" />
      </div>

      <div
        ref={mountRef}
        className="g1-three-mount"
      />

      {loadState === "loading" && (
        <div className="g1-model-state">
          Loading G1 model…
        </div>
      )}

      {loadState === "error" && (
        <div className="g1-model-state error">
          Model unavailable
        </div>
      )}

      <div className="g1-model-label">
        <strong>G1 EDU U7</strong>

        <span>
          {loadState === "ready"
            ? "UNITREE HUMANOID"
            : "UNITREE MODEL"}
        </span>
      </div>
    </div>
  );
}
