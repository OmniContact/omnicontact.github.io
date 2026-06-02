/**
 * SDF voxel grid loading and trilinear interpolation lookup.
 * Loads precomputed unsigned distance fields and computes distance + gradient
 * for arbitrary query points via trilinear interpolation + central differences.
 */

export class SDFVoxelGrid {
  /**
   * @param {object} data - Parsed JSON with grid_size, bbox_min, bbox_max, distances_b64
   */
  constructor(data) {
    this.gridSize = data.grid_size; // [nx, ny, nz]
    this.bboxMin = data.bbox_min;   // [x, y, z]
    this.bboxMax = data.bbox_max;   // [x, y, z]

    // Decode base64 Float32 distances
    const binaryStr = atob(data.distances_b64);
    const bytes = new Uint8Array(binaryStr.length);
    for (let i = 0; i < binaryStr.length; i++) {
      bytes[i] = binaryStr.charCodeAt(i);
    }
    this.distances = new Float32Array(bytes.buffer);

    // Precompute voxel sizes
    this.voxelSize = [
      (this.bboxMax[0] - this.bboxMin[0]) / (this.gridSize[0] - 1),
      (this.bboxMax[1] - this.bboxMin[1]) / (this.gridSize[1] - 1),
      (this.bboxMax[2] - this.bboxMin[2]) / (this.gridSize[2] - 1),
    ];

    this.nx = this.gridSize[0];
    this.ny = this.gridSize[1];
    this.nz = this.gridSize[2];
  }

  /**
   * Get distance at integer grid coordinates (clamped).
   */
  _getDistance(ix, iy, iz) {
    ix = Math.max(0, Math.min(this.nx - 1, ix));
    iy = Math.max(0, Math.min(this.ny - 1, iy));
    iz = Math.max(0, Math.min(this.nz - 1, iz));
    return this.distances[ix * this.ny * this.nz + iy * this.nz + iz];
  }

  /**
   * Trilinear interpolation of distance at a continuous grid coordinate.
   */
  _interpolateDistance(gx, gy, gz) {
    const ix = Math.floor(gx);
    const iy = Math.floor(gy);
    const iz = Math.floor(gz);

    const fx = gx - ix;
    const fy = gy - iy;
    const fz = gz - iz;

    const d000 = this._getDistance(ix, iy, iz);
    const d001 = this._getDistance(ix, iy, iz + 1);
    const d010 = this._getDistance(ix, iy + 1, iz);
    const d011 = this._getDistance(ix, iy + 1, iz + 1);
    const d100 = this._getDistance(ix + 1, iy, iz);
    const d101 = this._getDistance(ix + 1, iy, iz + 1);
    const d110 = this._getDistance(ix + 1, iy + 1, iz);
    const d111 = this._getDistance(ix + 1, iy + 1, iz + 1);

    const c00 = d000 * (1 - fz) + d001 * fz;
    const c01 = d010 * (1 - fz) + d011 * fz;
    const c10 = d100 * (1 - fz) + d101 * fz;
    const c11 = d110 * (1 - fz) + d111 * fz;

    const c0 = c00 * (1 - fy) + c01 * fy;
    const c1 = c10 * (1 - fy) + c11 * fy;

    return c0 * (1 - fx) + c1 * fx;
  }

  /**
   * Query the SDF at a point in the object's local frame.
   * Writes results into the provided `out` object to avoid per-call allocation.
   * @param {number} px - x coordinate in object local frame
   * @param {number} py - y coordinate in object local frame
   * @param {number} pz - z coordinate in object local frame
   * @param {number} maxDist - maximum valid distance (default 1.0)
   * @param {{distance: number, gx: number, gy: number, gz: number}} out - reusable result object
   */
  query(px, py, pz, maxDist, out) {
    // Convert to grid coordinates
    const gx = (px - this.bboxMin[0]) / this.voxelSize[0];
    const gy = (py - this.bboxMin[1]) / this.voxelSize[1];
    const gz = (pz - this.bboxMin[2]) / this.voxelSize[2];

    // Check if outside grid bounds
    if (gx < 0 || gx > this.nx - 1 || gy < 0 || gy > this.ny - 1 || gz < 0 || gz > this.nz - 1) {
      out.distance = maxDist; out.gx = 0; out.gy = 0; out.gz = 0;
      return;
    }

    // Interpolated distance
    const dist = this._interpolateDistance(gx, gy, gz);

    if (dist >= maxDist) {
      out.distance = maxDist; out.gx = 0; out.gy = 0; out.gz = 0;
      return;
    }

    // Compute gradient via central differences
    const h = 1.0; // 1 grid cell step
    const dxp = this._interpolateDistance(Math.min(gx + h, this.nx - 1), gy, gz);
    const dxm = this._interpolateDistance(Math.max(gx - h, 0), gy, gz);
    const dyp = this._interpolateDistance(gx, Math.min(gy + h, this.ny - 1), gz);
    const dym = this._interpolateDistance(gx, Math.max(gy - h, 0), gz);
    const dzp = this._interpolateDistance(gx, gy, Math.min(gz + h, this.nz - 1));
    const dzm = this._interpolateDistance(gx, gy, Math.max(gz - h, 0));

    let gradX = (dxp - dxm) / (2.0 * this.voxelSize[0]);
    let gradY = (dyp - dym) / (2.0 * this.voxelSize[1]);
    let gradZ = (dzp - dzm) / (2.0 * this.voxelSize[2]);

    // Normalize gradient
    const norm = Math.sqrt(gradX * gradX + gradY * gradY + gradZ * gradZ);
    if (norm > 1e-6) {
      gradX /= norm;
      gradY /= norm;
      gradZ /= norm;
    }

    out.distance = dist; out.gx = gradX; out.gy = gradY; out.gz = gradZ;
  }
}


/**
 * Manages multiple SDF grids for scene objects.
 * Computes closest-object SDF for a set of query points.
 */
export class SDFManager {
  constructor() {
    /** @type {Map<string, SDFVoxelGrid>} */
    this.grids = new Map();
    /** @type {Map<string, {pos: Float32Array, quat: Float32Array}>} */
    this.objectPoses = new Map();
    this.maxValidDistance = 1.0;
    // Pre-allocated query output buffers (resized as needed)
    this._queryNumPoints = 0;
    this._queryDistances = null;
    this._queryGradients = null;
    this._queryResult = { distances: null, gradients: null };
    // Pre-allocated workspace for queryClosest inner loop
    this._localPt = new Float32Array(3);
    this._worldGrad = new Float32Array(3);
    this._queryOut = { distance: 0, gx: 0, gy: 0, gz: 0 };
  }

  /**
   * Load an SDF grid from a URL.
   * @param {string} name - Object name
   * @param {string} url - URL to the SDF JSON file
   */
  async loadGrid(name, url) {
    const response = await fetch(url);
    if (!response.ok) {
      throw new Error(`Failed to load SDF grid from ${url}: ${response.status}`);
    }
    const data = await response.json();
    this.grids.set(name, new SDFVoxelGrid(data));
    // Initialize pose to origin, uniform scale 1
    this.objectPoses.set(name, {
      pos: new Float32Array([0, 0, 0]),
      quat: new Float32Array([1, 0, 0, 0]),  // w, x, y, z
      scale: 1.0,
    });
    console.log(`SDF grid loaded: ${name} (${data.grid_size.join('x')})`);
  }

  /**
   * Update the world-frame pose (and optionally scale) of a scene object.
   * @param {string} name
   * @param {Float32Array|number[]} pos - [x, y, z]
   * @param {Float32Array|number[]} quat - [w, x, y, z]
   * @param {number} [scale] - uniform scale factor (default: unchanged)
   */
  setObjectPose(name, pos, quat, scale) {
    const pose = this.objectPoses.get(name);
    if (!pose) return;
    pose.pos[0] = pos[0]; pose.pos[1] = pos[1]; pose.pos[2] = pos[2];
    pose.quat[0] = quat[0]; pose.quat[1] = quat[1]; pose.quat[2] = quat[2]; pose.quat[3] = quat[3];
    if (scale !== undefined) pose.scale = scale;
  }

  /**
   * Update only the uniform scale of a scene object.
   * @param {string} name
   * @param {number} scale
   */
  setObjectScale(name, scale) {
    const pose = this.objectPoses.get(name);
    if (pose) pose.scale = scale;
  }

  /**
   * Query the SDF for multiple points (in world frame).
   * For each query point, finds the closest object and returns
   * distance and gradient (in world frame).
   *
   * @param {Float32Array} queryPositions - Flat array [x0,y0,z0, x1,y1,z1, ...]
   * @param {number} numPoints
   * @returns {{ distances: Float32Array, gradients: Float32Array }}
   *   distances: (numPoints,), gradients: (numPoints*3,)
   */
  queryClosest(queryPositions, numPoints) {
    // Resize pre-allocated buffers if needed
    if (this._queryNumPoints !== numPoints) {
      this._queryNumPoints = numPoints;
      this._queryDistances = new Float32Array(numPoints);
      this._queryGradients = new Float32Array(numPoints * 3);
    }
    const distances = this._queryDistances;
    const gradients = this._queryGradients;
    distances.fill(this.maxValidDistance);
    gradients.fill(0);

    const localPt = this._localPt;
    const worldGrad = this._worldGrad;
    const queryOut = this._queryOut;

    for (const [name, grid] of this.grids) {
      const pose = this.objectPoses.get(name);
      if (!pose) continue;

      const s = pose.scale ?? 1.0;
      const invS = 1.0 / s;
      const localMaxDist = this.maxValidDistance * invS;
      const pq = pose.quat;
      const qw = pq[0], qx = pq[1], qy = pq[2], qz = pq[3];
      const px = pose.pos[0], py = pose.pos[1], pz = pose.pos[2];

      for (let i = 0; i < numPoints; i++) {
        // Inline _worldToLocal: translate then rotate by inverse quat
        const dx = queryPositions[i * 3] - px;
        const dy = queryPositions[i * 3 + 1] - py;
        const dz = queryPositions[i * 3 + 2] - pz;
        // Inline _quatApplyInverse (conjugate rotation)
        const tx = 2.0 * (-qy * dz + qz * dy);
        const ty = 2.0 * (-qz * dx + qx * dz);
        const tz = 2.0 * (-qx * dy + qy * dx);
        localPt[0] = dx + qw * tx + (-qy * tz + qz * ty);
        localPt[1] = dy + qw * ty + (-qz * tx + qx * tz);
        localPt[2] = dz + qw * tz + (-qx * ty + qy * tx);

        grid.query(localPt[0] * invS, localPt[1] * invS, localPt[2] * invS, localMaxDist, queryOut);
        const dist = queryOut.distance * s;

        if (dist < distances[i]) {
          distances[i] = dist;
          // Inline _localToWorldDir: rotate gradient from local to world
          const lgx = queryOut.gx, lgy = queryOut.gy, lgz = queryOut.gz;
          const rtx = 2.0 * (qy * lgz - qz * lgy);
          const rty = 2.0 * (qz * lgx - qx * lgz);
          const rtz = 2.0 * (qx * lgy - qy * lgx);
          gradients[i * 3]     = lgx + qw * rtx + (qy * rtz - qz * rty);
          gradients[i * 3 + 1] = lgy + qw * rty + (qz * rtx - qx * rtz);
          gradients[i * 3 + 2] = lgz + qw * rtz + (qx * rty - qy * rtx);
        }
      }
    }

    this._queryResult.distances = distances;
    this._queryResult.gradients = gradients;
    return this._queryResult;
  }
}


// Quaternion math functions (_worldToLocal, _localToWorldDir, _quatApply,
// _quatApplyInverse) have been inlined directly into queryClosest() to
// avoid per-call array/object allocations in the hot path.
