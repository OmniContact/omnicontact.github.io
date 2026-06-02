#!/usr/bin/env python3
import json
import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path("/home/lightcone/nair_sim-feat-runyi")
sys.path.insert(0, str(PROJECT_ROOT))

from common.ctrlcomp import PolicyOutput, StateAndCmd
from common.utils import (
    get_gravity_orientation,
    matrix_from_quat,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_to_6d_batch,
    quat_rotate_inverse,
    subtract_frame_transforms,
    yaw_quat,
)
from policy.carrybox_bbox_manager.CarryBox_BBox_Manager_WTAC import CarryBox_BBox_Manager_WTAC


MJ2LAB = np.array([0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28], dtype=np.int32)
LAB2MJ = np.array([0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28], dtype=np.int32)
FUTURE_FRAMES = np.array([0, 1, 2, 3, 4, 8, 12, 16, 24, 32, 50], dtype=np.int32)
HISTORY_SLICES = ((0, 15), (15, 18), (18, 21), (21, 50), (50, 79), (79, 108), (108, 111), (111, 117), (117, 141))


def norm_stats(a, b):
    d = np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)
    return {
        "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
        "mean_abs": float(np.mean(np.abs(d))) if d.size else 0.0,
        "l2": float(np.linalg.norm(d)),
    }


def build_policy(reference_source="onnx_ref"):
    state = StateAndCmd(29)
    out = PolicyOutput(29)
    policy = CarryBox_BBox_Manager_WTAC(state, out)
    policy.reference_source = reference_source
    policy.task = "carrybox"
    return policy, state, out


def load_mj(xml_path):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def sync_state_cmd(model, data, policy, state):
    n = model.nu
    state.q = data.qpos[7:7+n].copy().astype(np.float32)
    state.dq = data.qvel[6:6+n].copy().astype(np.float32)
    state.base_pos = data.qpos[:3].copy().astype(np.float32)
    state.base_quat = data.qpos[3:7].copy().astype(np.float32)
    state.lin_vel = data.qvel[:3].copy().astype(np.float32)
    state.ang_vel = data.qvel[3:6].copy().astype(np.float32)
    state.gravity_ori = get_gravity_orientation(state.base_quat).astype(np.float32)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "box")
    state.obj_pos = data.xpos[body_id].copy().astype(np.float32)
    state.obj_quat = data.xquat[body_id].copy().astype(np.float32)


def reset_object_to_ref(model, data, policy):
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "box")
    qadr = int(model.jnt_qposadr[joint_id])
    vadr = int(model.jnt_dofadr[joint_id])
    data.qpos[qadr:qadr+3] = policy.ref_object_pos[0]
    data.qpos[qadr+3:qadr+7] = policy.ref_object_quat[0]
    data.qvel[vadr:vadr+6] = 0
    mujoco.mj_forward(model, data)


def py_segments(policy, state):
    fk = policy._get_fk_info()
    policy._update_ghost_robot()
    qj = (state.q[policy.mj2lab] - policy.default_angles_lab).astype(np.float32)
    dqj = state.dq[policy.mj2lab].astype(np.float32)
    robot_heading = yaw_quat(policy.torso_quat)
    obj_pos_rel, obj_rot_rel = subtract_frame_transforms(policy.torso_pos, robot_heading, state.obj_pos, state.obj_quat)
    obj_pos_rel = policy._clip_norm(obj_pos_rel)
    obj_rot_6d = matrix_from_quat(obj_rot_rel)[:, :2].reshape(-1).astype(np.float32)
    bbox_rel = policy._build_bbox_rel(robot_heading)
    tracking, *_ = policy._get_future_state()
    curr_prop = np.concatenate([
        policy.ee_pos,
        state.ang_vel.astype(np.float32).reshape(-1),
        state.gravity_ori.astype(np.float32),
        qj,
        dqj,
        policy.action,
        obj_pos_rel.astype(np.float32),
        obj_rot_6d,
        bbox_rel,
    ]).astype(np.float32)
    hist = np.zeros((policy.history_len, policy.single_obs_dim), dtype=np.float32)
    hist[-1] = curr_prop
    hist_flat = np.concatenate([hist[:, a:b].reshape(-1) for a, b in HISTORY_SLICES]).astype(np.float32)
    return {
        "tracking": tracking.astype(np.float32),
        "ee_pos": policy.ee_pos.astype(np.float32),
        "ang_vel": state.ang_vel.astype(np.float32),
        "gravity_ori": state.gravity_ori.astype(np.float32),
        "qj": qj,
        "dqj": dqj,
        "prev_action": policy.action.astype(np.float32),
        "obj_pos_rel": obj_pos_rel.astype(np.float32),
        "obj_rot_6d": obj_rot_6d,
        "bbox_rel": bbox_rel.astype(np.float32),
        "curr_prop": curr_prop,
        "history_flat": hist_flat,
        "full_obs": np.concatenate([tracking, hist_flat]).astype(np.float32),
    }


def pose(model, data, name, fallback=None):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        if fallback is None:
            raise KeyError(name)
        return fallback
    return data.xpos[body_id].copy().astype(np.float32), data.xquat[body_id].copy().astype(np.float32)


def clip_norm(v, max_norm=4.0):
    v = np.asarray(v, dtype=np.float32)
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, max_norm / np.maximum(norm, 1e-8))


def web_like_segments(model, data, policy, use_current_js_names=True, gravity_sign="js", bbox_half=0.3):
    torso_pos, torso_quat = pose(model, data, "torso_link")
    if use_current_js_names:
        lh_pos, _ = pose(model, data, "left_rubber_hand", pose(model, data, "left_wrist_yaw_link"))
        rh_pos, _ = pose(model, data, "right_rubber_hand", pose(model, data, "right_wrist_yaw_link"))
        mid_pos, _ = pose(model, data, "head_link", (state.base_pos, state.base_quat))
    else:
        lh_pos, _ = pose(model, data, "left_palm_link")
        rh_pos, _ = pose(model, data, "right_palm_link")
        mid_pos, _ = pose(model, data, "mid360_link")
    la_pos, _ = pose(model, data, "left_ankle_pitch_link")
    ra_pos, _ = pose(model, data, "right_ankle_pitch_link")
    obj_pos, obj_quat = pose(model, data, "box")

    ee_pos = np.concatenate([
        quat_rotate_inverse(torso_quat, lh_pos - torso_pos),
        quat_rotate_inverse(torso_quat, rh_pos - torso_pos),
        quat_rotate_inverse(torso_quat, la_pos - torso_pos),
        quat_rotate_inverse(torso_quat, ra_pos - torso_pos),
        quat_rotate_inverse(torso_quat, mid_pos - torso_pos),
    ]).astype(np.float32)

    state_q = data.qpos[7:7+model.nu].copy().astype(np.float32)
    state_dq = data.qvel[6:6+model.nu].copy().astype(np.float32)
    qj = np.zeros(29, dtype=np.float32)
    dqj = np.zeros(29, dtype=np.float32)
    for i in range(29):
        # Matches current JS loop, which treats mj2lab as lab->mj by mistake.
        mj_idx = int(MJ2LAB[i])
        qj[i] = state_q[mj_idx] - policy.default_angles_lab[i]
        dqj[i] = state_dq[mj_idx]

    if gravity_sign == "js":
        gravity_ori = quat_rotate_inverse(data.qpos[3:7].copy().astype(np.float32), np.array([0, 0, -1], dtype=np.float32))
    else:
        gravity_ori = get_gravity_orientation(data.qpos[3:7].copy().astype(np.float32)).astype(np.float32)

    robot_heading = yaw_quat(torso_quat)
    heading_conj = quat_conjugate(robot_heading)

    def rel_pose(pos_ref, quat_ref, idx):
        rel_pos = quat_apply(heading_conj, pos_ref[idx] - torso_pos)
        rel_quat = quat_mul(heading_conj, quat_ref[idx])
        rel_6d = quat_to_6d_batch(rel_quat[None, :])[0]
        return rel_pos.astype(np.float32), rel_6d.astype(np.float32)

    tracking_pose_chunks = []
    tracking_contact_chunks = []
    for off in FUTURE_FRAMES:
        idx = min(int(off), len(policy.ref_left_wrist_pos) - 1)
        # Matches current JS: left/right wrist, torso, ankles, contact; no object future.
        for p, q in [
            (policy.ref_left_wrist_pos, policy.ref_left_wrist_quat),
            (policy.ref_right_wrist_pos, policy.ref_right_wrist_quat),
            (policy.ref_torso_future_pos, policy.ref_torso_future_quat),
            (policy.ref_left_ankle_future_pos, policy.ref_left_ankle_future_quat),
            (policy.ref_right_ankle_future_pos, policy.ref_right_ankle_future_quat),
        ]:
            rp, r6 = rel_pose(p, q, idx)
            tracking_pose_chunks.extend([rp, r6])
        tracking_contact_chunks.append(policy.ref_contact[idx].astype(np.float32))
    tracking = np.concatenate([*tracking_pose_chunks, *tracking_contact_chunks]).astype(np.float32)

    obj_pos_rel, obj_rot_rel = subtract_frame_transforms(torso_pos, robot_heading, obj_pos, obj_quat)
    obj_pos_rel = clip_norm(obj_pos_rel).reshape(-1)
    obj_rot_6d = matrix_from_quat(obj_rot_rel)[:, :2].reshape(-1).astype(np.float32)

    bbox_offsets = np.array([[1, 1, 1], [1, 1, -1], [1, -1, 1], [1, -1, -1], [-1, 1, 1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]], dtype=np.float32) * float(bbox_half)
    corners = np.array([quat_apply(obj_quat, c) + obj_pos for c in bbox_offsets], dtype=np.float32)
    bbox_rel = clip_norm(np.array([quat_apply(heading_conj, c - torso_pos) for c in corners], dtype=np.float32)).reshape(-1).astype(np.float32)

    curr_prop = np.concatenate([
        ee_pos,
        data.qvel[3:6].copy().astype(np.float32),
        gravity_ori.astype(np.float32),
        qj,
        dqj,
        np.zeros(29, dtype=np.float32),
        obj_pos_rel.astype(np.float32),
        obj_rot_6d.astype(np.float32),
        bbox_rel,
    ]).astype(np.float32)
    hist = np.zeros((5, 141), dtype=np.float32)
    hist[-1] = curr_prop
    hist_flat = np.concatenate([hist[:, a:b].reshape(-1) for a, b in HISTORY_SLICES]).astype(np.float32)
    return {
        "tracking": tracking,
        "ee_pos": ee_pos,
        "ang_vel": data.qvel[3:6].copy().astype(np.float32),
        "gravity_ori": gravity_ori.astype(np.float32),
        "qj": qj,
        "dqj": dqj,
        "prev_action": np.zeros(29, dtype=np.float32),
        "obj_pos_rel": obj_pos_rel.astype(np.float32),
        "obj_rot_6d": obj_rot_6d.astype(np.float32),
        "bbox_rel": bbox_rel,
        "curr_prop": curr_prop,
        "history_flat": hist_flat,
        "full_obs": np.concatenate([tracking, hist_flat]).astype(np.float32),
    }


if __name__ == "__main__":
    xml = PROJECT_ROOT / "g1_description" / "carrybox_manager_carry.xml"
    model, data = load_mj(xml)
    policy, state, out = build_policy("onnx_ref")
    sync_state_cmd(model, data, policy, state)
    policy.enter()
    reset_object_to_ref(model, data, policy)
    sync_state_cmd(model, data, policy, state)

    py = py_segments(policy, state)
    current = web_like_segments(model, data, policy, use_current_js_names=True, gravity_sign="js", bbox_half=0.3)
    fixed = web_like_segments(model, data, policy, use_current_js_names=False, gravity_sign="py", bbox_half=0.15)

    report = {
        "dims": {k: int(v.size) for k, v in py.items()},
        "current_js_vs_python": {k: norm_stats(py[k], current[k]) for k in py.keys()},
        "after_all_current_fixes_vs_python": {k: norm_stats(py[k], fixed[k]) for k in py.keys()},
        "notes": {
            "current_js_names": "left_rubber_hand/right_rubber_hand/head_link fallback, gravity=[0,0,-1], qj loop uses MJ2LAB as if lab->mj",
            "after_all_current_fixes": "uses left_palm_link/right_palm_link/mid360_link, Python-compatible rotation 6D, contact-at-tail tracking layout, and bbox half-size 0.15",
        },
    }
    print(json.dumps(report, indent=2))
