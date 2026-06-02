#!/usr/bin/env python3
"""Compare native OmniPlanCarryBoxWTACO against the current web rule planner.

Both planners are fed from the same MuJoCo reset scene and the same target box
goal.  This intentionally mirrors the simplified planner currently implemented
in carryBoxWTACPolicyRunner.js so we can see where the webpage diverges from the
local Python runner.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import mujoco
import numpy as np


NATIVE_REPO = Path("/home/lightcone/nair_sim-feat-runyi")
if str(NATIVE_REPO) not in sys.path:
    sys.path.insert(0, str(NATIVE_REPO))

from common.utils import quat_apply, quat_mul, quat_slerp, yaw_quat, yaw_to_quat  # noqa: E402
from policy.carrybox_bbox_manager.omniplan_carrybox import OmniPlanCarryBoxWTACO  # noqa: E402


def normalize_quat(q):
    q = np.asarray(q, dtype=np.float32).reshape(4)
    n = float(np.linalg.norm(q))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return (q / n).astype(np.float32)


def slerp(q0, q1, t):
    return normalize_quat(quat_slerp(normalize_quat(q0), normalize_quat(q1), float(t)))


def lerp_vec(a, b, t):
    a = np.asarray(a, dtype=np.float32).reshape(3)
    b = np.asarray(b, dtype=np.float32).reshape(3)
    return (a + (b - a) * float(t)).astype(np.float32)


def interp_count(a, b, step, min_count=2):
    d = float(np.linalg.norm(np.asarray(b, dtype=np.float32) - np.asarray(a, dtype=np.float32)))
    return max(int(min_count), int(math.floor(d / float(step))) + 1)


def body_pose(model, data, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise KeyError(name)
    return data.xpos[body_id].copy().astype(np.float32), data.xquat[body_id].copy().astype(np.float32)


def web_like_planner(scene, goal, dims):
    torso_pos, torso_quat = scene["torso_link"]
    left_hand_pos, left_hand_quat = scene["left_palm_link"]
    right_hand_pos, right_hand_quat = scene["right_palm_link"]
    left_ankle_pos, left_ankle_quat = scene["left_ankle_pitch_link"]
    right_ankle_pos, right_ankle_quat = scene["right_ankle_pitch_link"]
    obj_pos, obj_quat = scene["box"]

    obj = obj_pos.astype(np.float32)
    goal = np.asarray(goal, dtype=np.float32).reshape(3)
    dims = np.asarray(dims, dtype=np.float32).reshape(3)
    torso_start = torso_pos.astype(np.float32)
    yaw0 = normalize_quat(yaw_quat(torso_quat))
    to_obj_yaw = yaw_to_quat(math.atan2(float(obj[1] - torso_start[1]), float(obj[0] - torso_start[0])))

    ax = quat_apply(obj_quat, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    ay = quat_apply(obj_quat, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    candidates = [
        (obj + ax * dims[0], ax, float(dims[1])),
        (obj - ax * dims[0], -ax, float(dims[1])),
        (obj + ay * dims[1], ay, float(dims[0])),
        (obj - ay * dims[1], -ay, float(dims[0])),
    ]
    candidates.sort(key=lambda c: float(np.linalg.norm(c[0][:2] - torso_start[:2])))
    _face_pos, face_normal, face_width = candidates[0]
    approach_dir = -face_normal
    face_yaw = yaw_to_quat(math.atan2(float(approach_dir[1]), float(approach_dir[0])))
    pitch_down = np.array([0.7017, 0.0, 0.7017, 0.0], dtype=np.float32)
    q_pre = normalize_quat(quat_mul(face_yaw, pitch_down))
    q_grasp = normalize_quat(quat_mul(face_yaw, np.array([0.717, 0.0, 0.717, 0.0], dtype=np.float32)))

    fwd = quat_apply(face_yaw, np.array([1.0, 0.0, 0.0], dtype=np.float32))
    stand = np.array([obj[0] - fwd[0] * 0.62, obj[1] - fwd[1] * 0.62, 0.793], dtype=np.float32)
    pre_center = np.array([obj[0] - fwd[0] * 0.38, obj[1] - fwd[1] * 0.38, max(float(obj[2]), 0.7)], dtype=np.float32)
    grasp_center = obj.copy()
    lift_obj = np.array([obj[0], obj[1], max(float(obj[2] + 0.35), 0.85)], dtype=np.float32)
    carry_goal = np.array([goal[0], goal[1], max(float(goal[2]), float(lift_obj[2]))], dtype=np.float32)
    place_obj = goal.copy()

    frames = []

    def hands_from_center(center, yaw_q, half_width):
        y = quat_apply(yaw_q, np.array([0.0, 1.0, 0.0], dtype=np.float32))
        return center + y * float(half_width), center - y * float(half_width)

    def push(phase, t_pos, t_quat, l_pos, l_quat, r_pos, r_quat, o_pos, o_quat, contact):
        frames.append(
            {
                "phase": int(phase),
                "ref_torso_future_pos": np.asarray(t_pos, dtype=np.float32),
                "ref_torso_future_quat": normalize_quat(t_quat),
                "ref_left_ankle_future_pos": left_ankle_pos.copy(),
                "ref_left_ankle_future_quat": normalize_quat(left_ankle_quat),
                "ref_right_ankle_future_pos": right_ankle_pos.copy(),
                "ref_right_ankle_future_quat": normalize_quat(right_ankle_quat),
                "ref_left_wrist_pos": np.asarray(l_pos, dtype=np.float32),
                "ref_left_wrist_quat": normalize_quat(l_quat),
                "ref_right_wrist_pos": np.asarray(r_pos, dtype=np.float32),
                "ref_right_wrist_quat": normalize_quat(r_quat),
                "ref_object_pos": np.asarray(o_pos, dtype=np.float32),
                "ref_object_quat": normalize_quat(o_quat),
                "ref_contact": np.asarray(contact, dtype=np.float32),
            }
        )

    def add_segment(phase, n, fn):
        for i in range(int(n)):
            fn(1.0 if n <= 1 else i / (n - 1), i)

    start_l = left_hand_pos.copy()
    start_r = right_hand_pos.copy()
    start_lq = left_hand_quat.copy()
    start_rq = right_hand_quat.copy()
    open_width = max(0.24, float(np.linalg.norm(start_l - start_r)) * 0.5)
    grasp_width = max(face_width - 0.02, 0.1)

    def phase11(u, _i):
        q = slerp(yaw0, to_obj_yaw, u)
        tp = lerp_vec(torso_start, [torso_start[0], torso_start[1], 0.793], u)
        c = lerp_vec((start_l + start_r) * 0.5, [tp[0], tp[1], tp[2] - 0.13], u)
        lp, rp = hands_from_center(c, q, open_width)
        lq = slerp(start_lq, normalize_quat(quat_mul(q, pitch_down)), u)
        rq = slerp(start_rq, normalize_quat(quat_mul(q, pitch_down)), u)
        push(11, tp, q, lp, lq, rp, rq, obj, obj_quat, [0, 0, 0, 0])

    add_segment(11, 40, phase11)

    def phase12(u, _i):
        tp = lerp_vec(torso_start, stand, u)
        q = slerp(to_obj_yaw, face_yaw, u)
        c = lerp_vec([tp[0], tp[1], tp[2] - 0.13], pre_center, u)
        lp, rp = hands_from_center(c, q, open_width)
        lq = normalize_quat(quat_mul(q, pitch_down))
        push(12, tp, q, lp, lq, rp, lq, obj, obj_quat, [0, 0, 0, 0])

    add_segment(12, interp_count(torso_start, stand, 0.02, 30), phase12)

    def phase14(u, _i):
        c = lerp_vec(pre_center, grasp_center, u)
        lp, rp = hands_from_center(c, face_yaw, open_width + (grasp_width - open_width) * u)
        q = slerp(q_pre, q_grasp, u)
        tp = lerp_vec(stand, [stand[0], stand[1], min(float(stand[2]), 0.62)], u)
        push(14, tp, face_yaw, lp, q, rp, q, obj, obj_quat, [0, 0, 0, 0])

    add_segment(14, 55, phase14)
    for _ in range(30):
        lp, rp = hands_from_center(grasp_center, face_yaw, grasp_width)
        push(15, [stand[0], stand[1], min(float(stand[2]), 0.62)], face_yaw, lp, q_grasp, rp, q_grasp, obj, obj_quat, [0, 0, 1, 1])

    def phase21(u, _i):
        o = lerp_vec(obj, lift_obj, u)
        lp, rp = hands_from_center(o, face_yaw, grasp_width)
        tp = lerp_vec([stand[0], stand[1], min(float(stand[2]), 0.62)], stand, u)
        push(21, tp, face_yaw, lp, q_grasp, rp, q_grasp, o, obj_quat, [0, 0, 1, 1])

    add_segment(21, 60, phase21)

    def phase23(u, _i):
        o = lerp_vec(lift_obj, carry_goal, u)
        q = yaw_to_quat(math.atan2(float(goal[1] - obj[1]), float(goal[0] - obj[0])))
        lp, rp = hands_from_center(o, q, grasp_width)
        tp = np.array([o[0] - fwd[0] * 0.62, o[1] - fwd[1] * 0.62, 0.793], dtype=np.float32)
        push(23, tp, q, lp, q_grasp, rp, q_grasp, o, obj_quat, [0, 0, 1, 1])

    add_segment(23, interp_count(lift_obj, carry_goal, 0.02, 50), phase23)

    def phase25(u, _i):
        o = lerp_vec(carry_goal, place_obj, u)
        lp, rp = hands_from_center(o, face_yaw, grasp_width + 0.12 * u)
        push(25, [stand[0], stand[1], 0.793], face_yaw, lp, q_grasp, rp, q_grasp, o, obj_quat, [0, 0, 0, 0] if u > 0.5 else [0, 0, 1, 1])

    add_segment(25, 45, phase25)
    for _ in range(40):
        lp, rp = hands_from_center(place_obj, face_yaw, open_width)
        push(26, [stand[0], stand[1], 0.793], face_yaw, lp, q_pre, rp, q_pre, place_obj, obj_quat, [0, 0, 0, 0])

    out = {}
    for key in frames[0].keys():
        if key == "phase":
            out["ref_phase"] = np.asarray([f[key] for f in frames], dtype=np.int32)
        else:
            out[key] = np.asarray([f[key] for f in frames], dtype=np.float32)
    out["ref_table_1_pos"] = np.asarray([[obj[0], obj[1], obj[2] - dims[2] - 0.01]], dtype=np.float32).repeat(len(frames), axis=0)
    out["ref_table_2_pos"] = np.asarray([[goal[0], goal[1], goal[2] - dims[2] - 0.01]], dtype=np.float32).repeat(len(frames), axis=0)
    return out


def quat_angle_error(a, b):
    a = normalize_quat(a)
    b = normalize_quat(b)
    dot = abs(float(np.dot(a, b)))
    return 2.0 * math.acos(max(-1.0, min(1.0, dot)))


def summarize_array(name, native, web, n):
    if name not in native or name not in web:
        print(f"{name:34s} skipped (missing: native={name not in native}, web={name not in web})")
        return
    a = native[name][:n]
    b = web[name][:n]
    if a.shape[-1] == 4 and "quat" in name:
        err = np.asarray([quat_angle_error(x, y) for x, y in zip(a, b)], dtype=np.float32)
        unit = "rad"
    else:
        err = np.linalg.norm(a.reshape(n, -1) - b.reshape(n, -1), axis=1)
        unit = "m/L2"
    imax = int(np.argmax(err))
    print(f"{name:34s} mean={float(np.mean(err)):.6f} {unit}  max={float(err[imax]):.6f} @frame={imax}")


def phase_counts(x):
    vals, counts = np.unique(x, return_counts=True)
    return " ".join(f"{int(v)}:{int(c)}" for v, c in zip(vals, counts))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene",
        default="/home/lightcone/workspace/projects/Achievements/Paper-web/OmniContact/ours/omnicontact.github.io/humanoid-policy-viewer-src/public/examples/scenes/g1/carrybox_manager_carry.xml",
    )
    parser.add_argument("--goal", nargs=3, type=float, default=[1.0, 1.0, 0.55])
    parser.add_argument("--dims", nargs=3, type=float, default=[0.15, 0.15, 0.15])
    parser.add_argument("--emit-fixture", action="store_true", help="Emit scene/native planner JSON for JS parity tests.")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(args.scene)
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    names = [
        "torso_link",
        "left_palm_link",
        "right_palm_link",
        "left_ankle_pitch_link",
        "right_ankle_pitch_link",
        "box",
    ]
    scene = {name: body_pose(model, data, name) for name in names}
    goal = np.asarray(args.goal, dtype=np.float32)
    dims = np.asarray(args.dims, dtype=np.float32)

    native_planner = OmniPlanCarryBoxWTACO()
    native, native_target_yaw = native_planner.generate(
        torso_pos=scene["torso_link"][0],
        torso_quat=scene["torso_link"][1],
        left_ankle_pos=scene["left_ankle_pitch_link"][0],
        right_ankle_pos=scene["right_ankle_pitch_link"][0],
        obj_pos=scene["box"][0],
        obj_quat=scene["box"][1],
        box_half_dims=dims,
        target_obj_pos=goal,
        task="carrybox",
        left_wrist_pos=scene["left_palm_link"][0],
        left_wrist_quat=scene["left_palm_link"][1],
        right_wrist_pos=scene["right_palm_link"][0],
        right_wrist_quat=scene["right_palm_link"][1],
    )
    if args.emit_fixture:
        def arr(x):
            return np.asarray(x).tolist()

        payload = {
            "scene": {
                name: {
                    "pos": arr(scene[name][0]),
                    "quat": arr(scene[name][1]),
                }
                for name in names
            },
            "goal": arr(goal),
            "dims": arr(dims),
            "nativeTargetYaw": float(native_target_yaw),
            "native": {
                key: arr(value)
                for key, value in native.items()
            },
        }
        print(json.dumps(payload))
        return

    web = web_like_planner(scene, goal, dims)

    n = min(len(native["ref_phase"]), len(web["ref_phase"]))
    print(f"scene={args.scene}")
    print(f"goal={goal.tolist()} dims={dims.tolist()}")
    print(f"native_target_yaw={float(native_target_yaw):.6f} rad")
    print(f"native_len={len(native['ref_phase'])} web_len={len(web['ref_phase'])} compared_len={n}")
    print(f"native_phase_counts={phase_counts(native['ref_phase'])}")
    print(f"web_phase_counts   ={phase_counts(web['ref_phase'])}")
    print()

    for frame in [0, 1, 10, 39, 40, 80, 120, 200, min(n - 1, 300), min(n - 1, 500), n - 1]:
        print(
            f"frame {frame:4d}: native_phase={int(native['ref_phase'][frame]) if frame < len(native['ref_phase']) else -1:2d} "
            f"web_phase={int(web['ref_phase'][frame]) if frame < len(web['ref_phase']) else -1:2d} "
            f"torso_native={native['ref_torso_future_pos'][frame].round(4).tolist() if frame < len(native['ref_phase']) else None} "
            f"torso_web={web['ref_torso_future_pos'][frame].round(4).tolist() if frame < len(web['ref_phase']) else None}"
        )
    print()

    keys = [
        "ref_torso_future_pos",
        "ref_torso_future_quat",
        "ref_left_ankle_future_pos",
        "ref_right_ankle_future_pos",
        "ref_left_wrist_pos",
        "ref_right_wrist_pos",
        "ref_left_wrist_quat",
        "ref_right_wrist_quat",
        "ref_object_pos",
        "ref_contact",
        "ref_table_1_pos",
        "ref_table_2_pos",
    ]
    for key in keys:
        summarize_array(key, native, web, n)

    phase_mismatch = native["ref_phase"][:n] != web["ref_phase"][:n]
    print()
    print(f"phase_mismatch_frames={int(np.sum(phase_mismatch))}/{n}")
    first = np.flatnonzero(phase_mismatch)
    if len(first):
        i = int(first[0])
        print(f"first_phase_mismatch={i}: native={int(native['ref_phase'][i])} web={int(web['ref_phase'][i])}")


if __name__ == "__main__":
    main()
