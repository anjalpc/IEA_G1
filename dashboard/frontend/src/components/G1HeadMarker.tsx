import {
  useEffect,
  useRef,
} from "react";

import * as THREE from "three";

import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";


export function G1HeadMarker() {
  const mountRef =
    useRef<HTMLDivElement>(null);

  useEffect(() => {
    const mount =
      mountRef.current;

    if (!mount) {
      return;
    }

    let cancelled = false;

    const scene =
      new THREE.Scene();


    const camera =
      new THREE.PerspectiveCamera(
        28,
        1,
        0.01,
        10,
      );

    camera.position.set(
      0,
      0,
      2.8,
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

    renderer.setSize(
      44,
      44,
      false,
    );

    renderer.setClearColor(
      0x000000,
      0,
    );

    renderer.outputColorSpace =
      THREE.SRGBColorSpace;

    renderer.toneMapping =
      THREE.ACESFilmicToneMapping;

    renderer.toneMappingExposure =
      1.15;


    mount.appendChild(
      renderer.domElement,
    );


    scene.add(
      new THREE.AmbientLight(
        0xffffff,
        1.4,
      ),
    );


    const key =
      new THREE.DirectionalLight(
        0xffffff,
        2.6,
      );

    key.position.set(
      3,
      4,
      4,
    );

    scene.add(key);


    const rim =
      new THREE.DirectionalLight(
        0x8ba9d4,
        1.1,
      );

    rim.position.set(
      -3,
      2,
      -3,
    );

    scene.add(rim);


    const material =
      new THREE.MeshPhysicalMaterial({
        color: 0x7d8792,

        metalness: 0.08,
        roughness: 0.34,

        clearcoat: 0.18,
        clearcoatRoughness: 0.42,
      });


    const loader =
      new STLLoader();

    let geometry:
      THREE.BufferGeometry | null = null;


    loader
      .loadAsync(
        "/models/g1/head_link.STL",
      )
      .then((loaded) => {
        if (cancelled) {
          loaded.dispose();
          return;
        }

        geometry = loaded;

        geometry.computeVertexNormals();
        geometry.center();


        const mesh =
          new THREE.Mesh(
            geometry,
            material,
          );


        /*
         * Same basic presentation orientation used by the
         * dashboard G1 viewer.
         */
        mesh.rotation.set(
          -Math.PI / 2,
          0,
          -Math.PI / 2,
        );


        const bounds =
          new THREE.Box3()
            .setFromObject(mesh);

        const size =
          bounds.getSize(
            new THREE.Vector3(),
          );

        const maxDimension =
          Math.max(
            size.x,
            size.y,
            size.z,
          );


        if (
          Number.isFinite(
            maxDimension,
          ) &&
          maxDimension > 0
        ) {
          mesh.scale.setScalar(
            1.55 / maxDimension,
          );
        }


        scene.add(mesh);
      })
      .catch((error) => {
        console.error(
          "Failed to load G1 head marker:",
          error,
        );
      });


    let frame = 0;

    const render = () => {
      frame =
        requestAnimationFrame(
          render,
        );

      renderer.render(
        scene,
        camera,
      );
    };

    render();


    return () => {
      cancelled = true;

      cancelAnimationFrame(
        frame,
      );

      geometry?.dispose();
      material.dispose();
      renderer.dispose();

      renderer.domElement.remove();
    };
  }, []);


  return (
    <div
      ref={mountRef}
      className="g1-head-marker"
      aria-hidden="true"
    />
  );
}
