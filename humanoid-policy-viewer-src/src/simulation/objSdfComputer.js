/**
 * Client-side OBJ parser and SDF voxel grid computation.
 *
 * Ports the Python precompute_sdf.py logic to JavaScript so users
 * can upload OBJ files and get SDF grids computed in the browser.
 */

/**
 * Transform OBJ vertices from OBJ space (Z-up, Y-forward) to MuJoCo space (Z-up, X-forward).
 * Applies -90° rotation around Z-axis in-place:
 *   mj_x =  obj_y
 *   mj_y = -obj_x
 *   mj_z =  obj_z
 * @param {Float64Array} vertices - flat [x,y,z, ...] array, modified in-place
 */
export function transformObjToMujoco(vertices) {
  for (let i = 0; i < vertices.length; i += 3) {
    const ox = vertices[i], oy = vertices[i + 1];
    vertices[i]     =  oy;   // mj_x = obj_y
    vertices[i + 1] = -ox;   // mj_y = -obj_x
    // vertices[i+2] unchanged (mj_z = obj_z)
  }
}

/**
 * Parse OBJ text into vertices and triangulated faces.
 * @param {string} text - Raw OBJ file content
 * @returns {{ vertices: Float64Array, faces: Int32Array, numVerts: number, numFaces: number }}
 */
export function parseObj(text) {
  const verts = [];
  const faces = [];

  const lines = text.split('\n');
  for (const line of lines) {
    const trimmed = line.trim();
    if (!trimmed || trimmed[0] === '#') continue;
    const parts = trimmed.split(/\s+/);

    if (parts[0] === 'v' && parts.length >= 4) {
      verts.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
    } else if (parts[0] === 'f') {
      const numVerts = verts.length / 3;
      const faceVerts = [];
      for (let i = 1; i < parts.length; i++) {
        if (!parts[i]) continue;
        let idx = parseInt(parts[i].split('/')[0], 10);
        if (isNaN(idx)) continue;
        // Handle negative (relative) indices: -1 = last vertex, -2 = second-to-last, etc.
        if (idx < 0) idx = numVerts + idx;
        else idx = idx - 1; // OBJ is 1-indexed
        if (idx >= 0 && idx < numVerts) {
          faceVerts.push(idx);
        }
      }
      // Fan-triangulate polygons (need at least 3 vertices)
      for (let i = 1; i < faceVerts.length - 1; i++) {
        faces.push(faceVerts[0], faceVerts[i], faceVerts[i + 1]);
      }
    }
  }

  const numVerts = verts.length / 3;
  const numFaces = faces.length / 3;

  return {
    vertices: new Float64Array(verts),
    faces: new Int32Array(faces),
    numVerts,
    numFaces,
  };
}


/**
 * Closest point on triangle (a, b, c) for a single query point p.
 * Uses the Voronoi region method.
 * All inputs are [x, y, z] as plain arrays or typed-array offsets.
 *
 * @returns {number[]} [cx, cy, cz] closest point
 */
function closestPointOnTriangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz) {
  const abx = bx - ax, aby = by - ay, abz = bz - az;
  const acx = cx - ax, acy = cy - ay, acz = cz - az;
  const apx = px - ax, apy = py - ay, apz = pz - az;

  const d1 = abx * apx + aby * apy + abz * apz;
  const d2 = acx * apx + acy * apy + acz * apz;

  // Vertex A
  if (d1 <= 0 && d2 <= 0) return [ax, ay, az];

  const bpx = px - bx, bpy = py - by, bpz = pz - bz;
  const d3 = abx * bpx + aby * bpy + abz * bpz;
  const d4 = acx * bpx + acy * bpy + acz * bpz;

  // Vertex B
  if (d3 >= 0 && d4 <= d3) return [bx, by, bz];

  // Edge AB
  const vc = d1 * d4 - d3 * d2;
  if (vc <= 0 && d1 >= 0 && d3 <= 0) {
    const denom = d1 - d3 || 1e-12;
    const t = d1 / denom;
    return [ax + t * abx, ay + t * aby, az + t * abz];
  }

  const cpx = px - cx, cpy = py - cy, cpz = pz - cz;
  const d5 = abx * cpx + aby * cpy + abz * cpz;
  const d6 = acx * cpx + acy * cpy + acz * cpz;

  // Vertex C
  if (d6 >= 0 && d5 <= d6) return [cx, cy, cz];

  // Edge AC
  const vb = d5 * d2 - d1 * d6;
  if (vb <= 0 && d2 >= 0 && d6 <= 0) {
    const denom = d2 - d6 || 1e-12;
    const t = d2 / denom;
    return [ax + t * acx, ay + t * acy, az + t * acz];
  }

  // Edge BC
  const va = d3 * d6 - d5 * d4;
  if (va <= 0 && (d4 - d3) >= 0 && (d5 - d6) >= 0) {
    const denom = (d4 - d3) + (d5 - d6) || 1e-12;
    const t = (d4 - d3) / denom;
    return [bx + t * (cx - bx), by + t * (cy - by), bz + t * (cz - bz)];
  }

  // Inside triangle
  const denomAll = va + vb + vc || 1e-12;
  const v = vb / denomAll;
  const w = vc / denomAll;
  return [
    ax + v * abx + w * acx,
    ay + v * aby + w * acy,
    az + v * abz + w * acz,
  ];
}


/**
 * Compute a 3D unsigned distance field on a regular grid.
 *
 * @param {Float64Array} vertices  – flat [x,y,z, ...] with numVerts entries
 * @param {Int32Array}   faces     – flat [i0,i1,i2, ...] with numFaces triangles
 * @param {number} numFaces
 * @param {number} [gridSize=32]
 * @param {number} [padding=1.0]
 * @param {function} [onProgress] – called with (currentFace, totalFaces)
 * @returns {{ grid_size: number[], bbox_min: number[], bbox_max: number[], distances_b64: string }}
 */
export function computeSdfGrid(vertices, faces, numFaces, gridSize = 32, padding = 1.0, onProgress) {
  // Compute bounding box
  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < vertices.length; i += 3) {
    const x = vertices[i], y = vertices[i + 1], z = vertices[i + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  minX -= padding; minY -= padding; minZ -= padding;
  maxX += padding; maxY += padding; maxZ += padding;

  const totalPts = gridSize * gridSize * gridSize;
  const distances = new Float32Array(totalPts);
  distances.fill(Infinity);

  // Precompute face vertex positions
  const fv = new Float64Array(numFaces * 9);
  for (let f = 0; f < numFaces; f++) {
    const i0 = faces[f * 3] * 3;
    const i1 = faces[f * 3 + 1] * 3;
    const i2 = faces[f * 3 + 2] * 3;
    fv[f * 9]     = vertices[i0];     fv[f * 9 + 1] = vertices[i0 + 1]; fv[f * 9 + 2] = vertices[i0 + 2];
    fv[f * 9 + 3] = vertices[i1];     fv[f * 9 + 4] = vertices[i1 + 1]; fv[f * 9 + 5] = vertices[i1 + 2];
    fv[f * 9 + 6] = vertices[i2];     fv[f * 9 + 7] = vertices[i2 + 1]; fv[f * 9 + 8] = vertices[i2 + 2];
  }

  const stepX = (maxX - minX) / (gridSize - 1);
  const stepY = (maxY - minY) / (gridSize - 1);
  const stepZ = (maxZ - minZ) / (gridSize - 1);

  // For each face, compute distance for every grid point
  for (let fi = 0; fi < numFaces; fi++) {
    const fo = fi * 9;
    const ax = fv[fo], ay = fv[fo + 1], az = fv[fo + 2];
    const bx = fv[fo + 3], by = fv[fo + 4], bz = fv[fo + 5];
    const cx = fv[fo + 6], cy = fv[fo + 7], cz = fv[fo + 8];

    let idx = 0;
    for (let ix = 0; ix < gridSize; ix++) {
      const px = minX + ix * stepX;
      for (let iy = 0; iy < gridSize; iy++) {
        const py = minY + iy * stepY;
        for (let iz = 0; iz < gridSize; iz++) {
          const pz = minZ + iz * stepZ;
          const cp = closestPointOnTriangle(px, py, pz, ax, ay, az, bx, by, bz, cx, cy, cz);
          const dx = px - cp[0], dy = py - cp[1], dz = pz - cp[2];
          const distSq = dx * dx + dy * dy + dz * dz;
          if (distSq < distances[idx] * distances[idx]) {
            distances[idx] = Math.sqrt(distSq);
          }
          idx++;
        }
      }
    }

    if (onProgress) onProgress(fi + 1, numFaces);
  }

  // Clamp to padding
  for (let i = 0; i < totalPts; i++) {
    if (distances[i] > padding) distances[i] = padding;
  }

  // Encode distances as base64
  const bytes = new Uint8Array(distances.buffer);
  let binary = '';
  for (let i = 0; i < bytes.length; i++) {
    binary += String.fromCharCode(bytes[i]);
  }
  const b64 = btoa(binary);

  return {
    grid_size: [gridSize, gridSize, gridSize],
    bbox_min: [minX, minY, minZ],
    bbox_max: [maxX, maxY, maxZ],
    distances_b64: b64,
  };
}


/**
 * Parse an STL file (binary or ASCII) into vertices and indexed faces.
 * Deduplicates vertices so the output matches parseObj's format.
 * @param {ArrayBuffer} buffer - Raw STL file contents
 * @returns {{ vertices: Float64Array, faces: Int32Array, numVerts: number, numFaces: number }}
 */
export function parseStl(buffer) {
  // Detect format: binary STL has 80-byte header + 4-byte face count.
  // ASCII STL starts with "solid " but some binary files do too.
  // Use the size check: binary size = 84 + numFaces * 50
  const byteLen = buffer.byteLength;
  let isBinary = true;

  if (byteLen >= 84) {
    const view = new DataView(buffer);
    const faceCount = view.getUint32(80, true);
    if (faceCount * 50 + 84 !== byteLen) {
      // Size doesn't match binary format — check if it's ASCII
      const header = new Uint8Array(buffer, 0, Math.min(80, byteLen));
      const headerStr = String.fromCharCode(...header).trim();
      if (headerStr.startsWith('solid')) {
        isBinary = false;
      }
    }
  } else {
    isBinary = false;
  }

  // Raw (non-deduplicated) vertex array: 3 per face
  let rawVerts;
  let rawFaceCount;

  if (isBinary) {
    const view = new DataView(buffer);
    rawFaceCount = view.getUint32(80, true);
    rawVerts = new Float64Array(rawFaceCount * 9);
    for (let f = 0; f < rawFaceCount; f++) {
      const off = 84 + f * 50;
      // Skip 12-byte normal, read 3 vertices (each 3 × float32)
      for (let v = 0; v < 3; v++) {
        const vOff = off + 12 + v * 12;
        rawVerts[f * 9 + v * 3]     = view.getFloat32(vOff, true);
        rawVerts[f * 9 + v * 3 + 1] = view.getFloat32(vOff + 4, true);
        rawVerts[f * 9 + v * 3 + 2] = view.getFloat32(vOff + 8, true);
      }
    }
  } else {
    // ASCII STL
    const text = new TextDecoder().decode(buffer);
    const vertexRegex = /vertex\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)\s+([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)/gi;
    const coords = [];
    let match;
    while ((match = vertexRegex.exec(text)) !== null) {
      coords.push(parseFloat(match[1]), parseFloat(match[2]), parseFloat(match[3]));
    }
    rawFaceCount = Math.floor(coords.length / 9);
    rawVerts = new Float64Array(coords.slice(0, rawFaceCount * 9));
  }

  if (rawFaceCount === 0) {
    return { vertices: new Float64Array(0), faces: new Int32Array(0), numVerts: 0, numFaces: 0 };
  }

  // Deduplicate vertices using coordinate string keys
  const vertMap = new Map();
  const uniqueVerts = [];
  const faceIndices = new Int32Array(rawFaceCount * 3);

  for (let f = 0; f < rawFaceCount; f++) {
    for (let v = 0; v < 3; v++) {
      const x = rawVerts[f * 9 + v * 3];
      const y = rawVerts[f * 9 + v * 3 + 1];
      const z = rawVerts[f * 9 + v * 3 + 2];
      const key = `${x},${y},${z}`;
      let idx = vertMap.get(key);
      if (idx === undefined) {
        idx = uniqueVerts.length / 3;
        vertMap.set(key, idx);
        uniqueVerts.push(x, y, z);
      }
      faceIndices[f * 3 + v] = idx;
    }
  }

  return {
    vertices: new Float64Array(uniqueVerts),
    faces: faceIndices,
    numVerts: uniqueVerts.length / 3,
    numFaces: rawFaceCount,
  };
}

/**
 * Build a THREE.js BufferGeometry from parsed OBJ data.
 * @param {Float64Array} vertices
 * @param {Int32Array} faces
 * @param {number} numFaces
 * @param {object} THREE - three.js namespace
 * @returns {THREE.BufferGeometry}
 */
export function buildThreeGeometry(vertices, faces, numFaces, THREE) {
  // Vertices are in MuJoCo space (Z-up). Three.js is Y-up.
  // MuJoCo (x,y,z) → Three.js (x, z, -y)
  const positions = new Float32Array(numFaces * 9);
  for (let f = 0; f < numFaces; f++) {
    const i0 = faces[f * 3] * 3;
    const i1 = faces[f * 3 + 1] * 3;
    const i2 = faces[f * 3 + 2] * 3;
    positions[f * 9]     = vertices[i0];     positions[f * 9 + 1] = vertices[i0 + 2]; positions[f * 9 + 2] = -vertices[i0 + 1];
    positions[f * 9 + 3] = vertices[i1];     positions[f * 9 + 4] = vertices[i1 + 2]; positions[f * 9 + 5] = -vertices[i1 + 1];
    positions[f * 9 + 6] = vertices[i2];     positions[f * 9 + 7] = vertices[i2 + 2]; positions[f * 9 + 8] = -vertices[i2 + 1];
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  geometry.computeVertexNormals();
  return geometry;
}
