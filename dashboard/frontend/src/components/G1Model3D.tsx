import {
  useEffect,
  useRef,
  useState,
} from "react";

import * as THREE from "three";

import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { STLLoader } from "three/examples/jsm/loaders/STLLoader.js";


type LoadState =
  | "loading"
  | "ready"
  | "error";


type G1ModelView =
  | "control"
  | "overview"
  | "overview-full";


interface G1Model3DProps {
  view?: G1ModelView;
  showLabel?: boolean;
}


/*
 * Dashboard presentation pose only.
 *
 * These angles are NEVER sent to the physical robot.
 *
 * The Unitree locomotion home pose normally keeps the elbows
 * noticeably bent. For the Control-page digital twin we want
 * a calmer product-style stance with both arms hanging closer
 * to the torso.
 */
const CONTROL_DISPLAY_JOINT_POS: Record<string, number> = {
  /*
   * Viewer-only relaxed presentation pose.
   *
   * Unitree's official G1 geometry places the elbow-to-wrist
   * link roughly 73 degrees away from the upper-arm direction
   * at q = 0. Rotating about the elbow Y axis by ~1.28 rad
   * visually straightens the arm.
   *
   * These values are never sent to the physical robot.
   */

  // Left arm
  left_shoulder_pitch_joint: 0.0,
  left_shoulder_roll_joint: 0.0,
  left_shoulder_yaw_joint: 0.0,
  left_elbow_joint: 1.28,

  // Right arm
  right_shoulder_pitch_joint: 0.0,
  right_shoulder_roll_joint: 0.0,
  right_shoulder_yaw_joint: 0.0,
  right_elbow_joint: 1.28,

  // Neutral wrists
  left_wrist_roll_joint: 0.0,
  left_wrist_pitch_joint: 0.0,
  left_wrist_yaw_joint: 0.0,

  right_wrist_roll_joint: 0.0,
  right_wrist_pitch_joint: 0.0,
  right_wrist_yaw_joint: 0.0,
};


interface MeshAsset {
  file: string;
  scale: THREE.Vector3;
}


function parseNumbers(
  value: string | null,
  fallback: number[],
) {
  if (!value) {
    return fallback;
  }

  const result = value
    .trim()
    .split(/\s+/)
    .map(Number);

  return result.every(Number.isFinite)
    ? result
    : fallback;
}


function applyTransform(
  element: Element,
  object: THREE.Object3D,
  angleInRadians: boolean,
) {
  const pos = parseNumbers(
    element.getAttribute("pos"),
    [0, 0, 0],
  );

  object.position.set(
    pos[0],
    pos[1],
    pos[2],
  );


  /*
   * MuJoCo quaternion order:
   *
   *     w x y z
   *
   * Three.js expects:
   *
   *     x y z w
   */
  const quatText =
    element.getAttribute("quat");

  if (quatText) {
    const quat = parseNumbers(
      quatText,
      [1, 0, 0, 0],
    );

    object.quaternion.set(
      quat[1],
      quat[2],
      quat[3],
      quat[0],
    );

    return;
  }


  /*
   * Fallback for any elements that use Euler angles.
   */
  const eulerText =
    element.getAttribute("euler");

  if (eulerText) {
    const euler = parseNumbers(
      eulerText,
      [0, 0, 0],
    );

    const factor =
      angleInRadians
        ? 1
        : Math.PI / 180;

    object.rotation.set(
      euler[0] * factor,
      euler[1] * factor,
      euler[2] * factor,
      "XYZ",
    );
  }
}


function applyDisplayJointPose(
  bodyElement: Element,
  group: THREE.Group,
) {
  const jointElement =
    Array.from(
      bodyElement.children,
    ).find(
      (child) =>
        child.tagName === "joint",
    );

  if (!jointElement) {
    return;
  }

  const jointName =
    jointElement.getAttribute(
      "name",
    );

  if (!jointName) {
    return;
  }

  const angle =
    CONTROL_DISPLAY_JOINT_POS[
      jointName
    ];

  if (angle === undefined) {
    return;
  }

  const axisValues =
    parseNumbers(
      jointElement.getAttribute(
        "axis",
      ),
      [0, 0, 1],
    );

  const axis =
    new THREE.Vector3(
      axisValues[0],
      axisValues[1],
      axisValues[2],
    );

  if (axis.lengthSq() === 0) {
    return;
  }

  axis.normalize();

  const jointRotation =
    new THREE.Quaternion()
      .setFromAxisAngle(
        axis,
        angle,
      );

  /*
   * Body transform first, then revolute-joint rotation in
   * the body's local frame.
   */
  group.quaternion.multiply(
    jointRotation,
  );
}


function createMaterial(
  rgbaText: string | null,
) {
  const rgba = parseNumbers(
    rgbaText,
    [0.7, 0.7, 0.7, 1],
  );

  const color =
    new THREE.Color(
      rgba[0],
      rgba[1],
      rgba[2],
    );

  return new THREE.MeshPhysicalMaterial({
    color,

    transparent:
      rgba[3] < 0.999,

    opacity:
      rgba[3],

    metalness: 0.04,
    roughness: 0.4,

    clearcoat: 0.12,
    clearcoatRoughness: 0.5,
  });
}


export function G1Model3D({
  view = "control",
  showLabel = true,
}: G1Model3DProps) {
  const mountRef =
    useRef<HTMLDivElement>(null);

  const [loadState, setLoadState] =
    useState<LoadState>("loading");


  useEffect(() => {
    const mount =
      mountRef.current;

    if (!mount) {
      return;
    }


    let cancelled = false;


    // ========================================================
    // THREE SCENE
    // ========================================================

    const scene =
      new THREE.Scene();


    const camera =
      new THREE.PerspectiveCamera(
        28,
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
        0xf1f5fa,
        0x20252d,
        1.5,
      ),
    );


    const key =
      new THREE.DirectionalLight(
        0xffffff,
        3.0,
      );

    key.position.set(
      4,
      6,
      5,
    );

    scene.add(key);


    const fill =
      new THREE.DirectionalLight(
        0x86a9e5,
        1.15,
      );

    fill.position.set(
      -5,
      2,
      3,
    );

    scene.add(fill);


    const rim =
      new THREE.DirectionalLight(
        0xd1e0ff,
        1.0,
      );

    rim.position.set(
      1,
      4,
      -5,
    );

    scene.add(rim);


    // ========================================================
    // MODEL CONTAINERS
    // ========================================================

    /*
     * assembly preserves the actual MuJoCo hierarchy.
     *
     * presentation is only for screen orientation,
     * centering and scaling.
     */
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


    const loader =
      new STLLoader();


    const geometries =
      new Set<THREE.BufferGeometry>();

    const materials =
      new Set<THREE.Material>();


    async function buildRobot() {
      try {
        // ----------------------------------------------------
        // FETCH MUJOCO MODEL
        // ----------------------------------------------------

        const response =
          await fetch(
            "/models/g1/g1_29dof.xml",
          );

        if (!response.ok) {
          throw new Error(
            `G1 XML request failed: ${response.status}`,
          );
        }


        const xmlText =
          await response.text();


        if (cancelled) {
          return;
        }


        const xml =
          new DOMParser().parseFromString(
            xmlText,
            "application/xml",
          );


        const parserError =
          xml.querySelector(
            "parsererror",
          );

        if (parserError) {
          throw new Error(
            "Could not parse G1 MuJoCo XML",
          );
        }


        // ----------------------------------------------------
        // COMPILER SETTINGS
        // ----------------------------------------------------

        const compiler =
          xml.querySelector(
            "mujoco > compiler",
          );


        const meshDirectory =
          compiler?.getAttribute(
            "meshdir",
          ) ?? "meshes";


        const angleInRadians =
          compiler?.getAttribute(
            "angle",
          ) === "radian";


        // ----------------------------------------------------
        // ASSET LOOKUP
        // ----------------------------------------------------

        const meshAssets =
          new Map<
            string,
            MeshAsset
          >();


        for (
          const element
          of Array.from(
            xml.querySelectorAll(
              "asset > mesh",
            ),
          )
        ) {
          const name =
            element.getAttribute(
              "name",
            );

          const file =
            element.getAttribute(
              "file",
            );

          if (
            !name ||
            !file
          ) {
            continue;
          }


          const scaleValues =
            parseNumbers(
              element.getAttribute(
                "scale",
              ),
              [1, 1, 1],
            );


          meshAssets.set(
            name,
            {
              file,

              scale:
                new THREE.Vector3(
                  scaleValues[0],
                  scaleValues[1],
                  scaleValues[2],
                ),
            },
          );
        }


        // ----------------------------------------------------
        // MATERIAL CACHE
        // ----------------------------------------------------

        const materialCache =
          new Map<
            string,
            THREE.MeshPhysicalMaterial
          >();


        function getMaterial(
          rgba:
            string | null,
        ) {
          const key =
            rgba ??
            "0.7 0.7 0.7 1";


          const cached =
            materialCache.get(
              key,
            );

          if (cached) {
            return cached;
          }


          const material =
            createMaterial(
              rgba,
            );

          materialCache.set(
            key,
            material,
          );

          materials.add(
            material,
          );

          return material;
        }


        // ----------------------------------------------------
        // MESH LOADS
        // ----------------------------------------------------

        const pendingLoads:
          Promise<void>[] = [];


        function addVisualGeom(
          geomElement: Element,
          parent: THREE.Group,
        ) {
          const meshName =
            geomElement.getAttribute(
              "mesh",
            );

          if (!meshName) {
            return;
          }


          const asset =
            meshAssets.get(
              meshName,
            );

          if (!asset) {
            console.warn(
              "Unknown G1 mesh asset:",
              meshName,
            );

            return;
          }


          const task =
            loader
              .loadAsync(
                `/models/g1/${meshDirectory}/${asset.file}`,
              )
              .then(
                (geometry) => {
                  if (cancelled) {
                    geometry.dispose();
                    return;
                  }


                  geometry.computeVertexNormals();

                  geometries.add(
                    geometry,
                  );


                  const material =
                    getMaterial(
                      geomElement.getAttribute(
                        "rgba",
                      ),
                    );


                  const mesh =
                    new THREE.Mesh(
                      geometry,
                      material,
                    );


                  mesh.name =
                    meshName;


                  applyTransform(
                    geomElement,
                    mesh,
                    angleInRadians,
                  );


                  mesh.scale.copy(
                    asset.scale,
                  );


                  parent.add(
                    mesh,
                  );
                },
              );


          pendingLoads.push(
            task,
          );
        }


        // ----------------------------------------------------
        // RECURSIVE BODY TREE
        // ----------------------------------------------------

        function buildBody(
          bodyElement: Element,
          parent: THREE.Group,
        ) {
          const group =
            new THREE.Group();


          group.name =
            bodyElement.getAttribute(
              "name",
            ) ?? "body";


          applyTransform(
            bodyElement,
            group,
            angleInRadians,
          );

          /*
           * Apply only the dashboard presentation pose.
           * When live joint-state rendering is connected,
           * this will be replaced by robot-reported q values.
           */
          applyDisplayJointPose(
            bodyElement,
            group,
          );


          parent.add(
            group,
          );


          const children =
            Array.from(
              bodyElement.children,
            );


          for (
            const child
            of children
          ) {
            /*
             * Use MuJoCo's visual geometry only.
             *
             * The file contains duplicate collision
             * geometry for many links. group="1"
             * is the visual mesh representation.
             */
            if (
              child.tagName ===
              "geom"
            ) {
              if (
                child.getAttribute(
                  "type",
                ) === "mesh" &&
                child.getAttribute(
                  "group",
                ) === "1"
              ) {
                addVisualGeom(
                  child,
                  group,
                );
              }

              continue;
            }


            if (
              child.tagName ===
              "body"
            ) {
              buildBody(
                child,
                group,
              );
            }
          }
        }


        const worldbody =
          xml.querySelector(
            "worldbody",
          );


        if (!worldbody) {
          throw new Error(
            "G1 worldbody not found",
          );
        }


        for (
          const child
          of Array.from(
            worldbody.children,
          )
        ) {
          if (
            child.tagName ===
            "body"
          ) {
            buildBody(
              child,
              assembly,
            );
          }
        }


        await Promise.all(
          pendingLoads,
        );


        if (cancelled) {
          return;
        }


        /*
         * Sanity-check critical visual links.
         * This does not affect rendering.
         */
        const headVisual =
          assembly.getObjectByName(
            "head_link",
          );

        if (!headVisual) {
          console.warn(
            "G1 visual hierarchy loaded without head_link",
          );
        }


        // ----------------------------------------------------
        // FIT COMPLETE ROBOT INTO VIEW
        // ----------------------------------------------------

        assembly.updateMatrixWorld(
          true,
        );


        const bounds =
          new THREE.Box3()
            .setFromObject(
              assembly,
            );


        const center =
          bounds.getCenter(
            new THREE.Vector3(),
          );


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
          !Number.isFinite(
            maxDimension,
          ) ||
          maxDimension <= 0
        ) {
          throw new Error(
            "Invalid G1 model bounds",
          );
        }


        /*
         * Center the complete hierarchy without
         * changing link-to-link transforms.
         */
        assembly.position.set(
          -center.x,
          -center.y,
          -center.z,
        );


        /*
         * Presentation orientation matching the
         * frontal view we already established.
         */
        presentation.rotation.set(
          -Math.PI / 2,
          0,
          -Math.PI / 2,
        );


        const scale =
          view === "control"
            ? 3.15 / maxDimension
            : 2.72 / maxDimension;

        presentation.scale.setScalar(
          scale,
        );

        /*
         * Calculate bounds after the presentation rotation
         * and scale. This lets each dashboard page frame the
         * same real robot model differently.
         */
        presentation.updateMatrixWorld(
          true,
        );

        const displayBounds =
          new THREE.Box3().setFromObject(
            presentation,
          );

        const displaySize =
          displayBounds.getSize(
            new THREE.Vector3(),
          );

        if (view === "control") {
          /*
           * Frozen Control framing.
           */
          const focusY =
            displayBounds.min.y +
            displaySize.y * 0.84;

          controls.target.set(
            0,
            focusY,
            0,
          );

          camera.position.set(
            0,
            focusY + 0.06,
            2.45,
          );

        } else if (view === "overview") {
          /*
           * Primary Overview framing:
           * head / shoulders / torso dominant.
           */
          const focusY =
            displayBounds.min.y +
            displaySize.y * 0.84;

          controls.target.set(
            0,
            focusY,
            0,
          );

          camera.position.set(
            0,
            focusY + 0.06,
            2.65,
          );

        } else {
          /*
           * Full-body Overview inspection mode.
           *
           * Keep the automatic fit distance, but bias the
           * camera target slightly ABOVE the true model center.
           * That moves the robot lower in the frame so it does
           * not feel stuck to the top edge.
           */
          const displayCenter =
            displayBounds.getCenter(
              new THREE.Vector3(),
            );

          const verticalFov =
            THREE.MathUtils.degToRad(
              camera.fov,
            );

          const fitDistance =
            (displaySize.y * 0.5) /
            Math.tan(
              verticalFov * 0.5,
            );

          const fullBodyDistance =
            fitDistance * 1.12;

          /*
           * Full-body inspection should be geometrically
           * centered in the viewport.
           */
          controls.target.set(
            displayCenter.x,
            displayCenter.y,
            displayCenter.z,
          );

          camera.position.set(
            displayCenter.x,
            displayCenter.y,
            displayCenter.z +
              fullBodyDistance,
          );
        }

        camera.lookAt(
          controls.target,
        );

        controls.update();

        setLoadState(
          "ready",
        );
      } catch (error) {
        console.error(
          "Failed to construct full G1:",
          error,
        );


        if (!cancelled) {
          setLoadState(
            "error",
          );
        }
      }
    }


    // ========================================================
    // ORBIT CONTROLS
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

    /*
     * You said tilt is fine, so deliberately do not
     * constrain azimuth or polar rotation.
     */
    controls.enableDamping =
      true;

    controls.dampingFactor =
      0.075;

    controls.rotateSpeed =
      0.5;

    /*
     * Load after controls exist because the final framing
     * target is calculated from the completed robot model.
     */
    void buildRobot();




    // ========================================================
    // RESIZE
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
    // RENDER
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


      materials.forEach(
        (material) =>
          material.dispose(),
      );


      renderer.dispose();


      renderer.domElement.remove();
    };
  }, [view]);


  return (
    <div
      className={`g1-head-3d g1-fullbody-3d g1-model-${view}`}
    >
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
          Loading complete G1…
        </div>
      )}

      {loadState === "error" && (
        <div className="g1-model-state error">
          Model unavailable
        </div>
      )}

      {showLabel && (
        <div className="g1-model-label">
          <strong>G1 EDU U7</strong>

          <span>
            {loadState === "ready"
              ? "29 DOF HUMANOID"
              : "UNITREE MODEL"}
          </span>
        </div>
      )}
    </div>
  );
}
