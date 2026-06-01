"""
Point-cloud file I/O, voxelization, and file-discovery utilities.

Supports PCD v0.7 files in both ASCII and binary (little-endian) formats.
"""

import glob
import os
import re
import struct
from typing import List, Optional, Set, Tuple

import numpy as np

_STAMP_RE_SUFFIX = re.compile(r"cloud_(?P<stamp>[0-9]+(?:\.[0-9]+)?)_gt\.pcd$")
_STAMP_RE_PREFIX = re.compile(r"^cloud_gt_(?P<stamp>[0-9]+(?:\.[0-9]+)?)\.pcd$")
_STAMP_RE_PLAIN = re.compile(r"cloud_(?P<stamp>[0-9]+(?:\.[0-9]+)?)\.pcd$")


def read_pcd_xyz(path: str) -> np.ndarray:
    """
    Read XYZ points from a PCD file (ASCII or binary little-endian).

    Returns an (N, 3) float64 array.  Returns shape (0, 3) if the file
    contains no points.
    """
    with open(path, "rb") as f:
        header: List[str] = []
        while True:
            ln = f.readline()
            if not ln:
                raise ValueError(f"PCD missing DATA line: {path}")
            s = ln.decode("utf-8", errors="replace").strip()
            header.append(s)
            if s.lower().startswith("data "):
                break

        hdr: dict[str, List[str]] = {}
        for h in header:
            if not h or h.startswith("#"):
                continue
            parts = h.split()
            if len(parts) >= 2:
                hdr[parts[0].upper()] = parts[1:]

        fields = [x.strip() for x in hdr.get("FIELDS", [])]
        sizes = [int(x) for x in hdr.get("SIZE", [])]
        types = [x.strip().upper() for x in hdr.get("TYPE", [])]
        counts = (
            [int(x) for x in hdr.get("COUNT", [])]
            if "COUNT" in hdr
            else [1] * len(fields)
        )
        n_pts = int(hdr.get("POINTS", ["0"])[0])
        data_spec = hdr.get("DATA", ["ascii"])[0].lower()

        if (
            not fields
            or len(fields) != len(sizes)
            or len(fields) != len(types)
            or len(fields) != len(counts)
        ):
            raise ValueError(f"PCD header incomplete in {path}")
        if "x" not in fields or "y" not in fields or "z" not in fields:
            raise ValueError(f"PCD missing x/y/z fields: {path} fields={fields}")

        ix = fields.index("x")
        iy = fields.index("y")
        iz = fields.index("z")
        for i in (ix, iy, iz):
            if counts[i] != 1 or types[i] != "F" or sizes[i] not in (4, 8):
                raise ValueError(
                    f"Unsupported x/y/z format in {path}: "
                    f"SIZE={sizes[i]} TYPE={types[i]} COUNT={counts[i]}"
                )

        offsets: List[int] = []
        off = 0
        for sz, ct in zip(sizes, counts):
            offsets.append(off)
            off += int(sz) * int(ct)
        point_step = off

        if data_spec == "ascii":
            return _read_ascii(f, fields, ix, iy, iz)

        if data_spec != "binary":
            raise ValueError(f"Unsupported DATA={data_spec!r} in {path}")

        return _read_binary(f, n_pts, point_step, offsets, sizes, ix, iy, iz)


def _read_ascii(f, fields, ix, iy, iz) -> np.ndarray:
    raw = f.read().decode("utf-8", errors="replace").strip()
    if not raw:
        return np.zeros((0, 3), dtype=np.float64)
    arr = np.fromstring(raw, sep=" ", dtype=np.float64)
    if arr.size % len(fields) == 0:
        rec = arr.reshape((-1, len(fields)))
        return rec[:, [ix, iy, iz]].astype(np.float64, copy=False)
    pts = []
    for ln in raw.splitlines():
        parts = ln.split()
        if len(parts) < len(fields):
            continue
        pts.append((float(parts[ix]), float(parts[iy]), float(parts[iz])))
    return np.asarray(pts, dtype=np.float64)


def _read_binary(f, n_pts, point_step, offsets, sizes, ix, iy, iz) -> np.ndarray:
    payload = f.read()
    if n_pts <= 0:
        n_pts = len(payload) // max(1, point_step)
    need = n_pts * point_step
    if len(payload) < need:
        n_pts = len(payload) // max(1, point_step)
        need = n_pts * point_step
    payload = payload[:need]

    def unpack(buf: bytes, sz: int) -> float:
        return float(struct.unpack_from("<f" if sz == 4 else "<d", buf)[0])

    out = np.empty((n_pts, 3), dtype=np.float64)
    for i in range(n_pts):
        base = i * point_step
        out[i, 0] = unpack(payload[base + offsets[ix]:base + offsets[ix] + sizes[ix]], sizes[ix])
        out[i, 1] = unpack(payload[base + offsets[iy]:base + offsets[iy] + sizes[iy]], sizes[iy])
        out[i, 2] = unpack(payload[base + offsets[iz]:base + offsets[iz] + sizes[iz]], sizes[iz])
    return out


def write_pcd_xyz_ascii(path: str, pts_xyz: np.ndarray) -> None:
    """Write an (N, 3) point array as a PCD v0.7 ASCII file."""
    pts = np.asarray(pts_xyz, dtype=np.float32)
    n = int(pts.shape[0])
    with open(path, "w") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {n}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {n}\n")
        f.write("DATA ascii\n")
        for i in range(n):
            x, y, z = pts[i]
            f.write(f"{x:.6f} {y:.6f} {z:.6f}\n")


def voxel_ids(points_xyz: np.ndarray, voxel_m: float) -> Set[bytes]:
    """
    Quantize points to integer voxel indices and return as a set of
    hashable byte keys (packed int32 triples).
    """
    pts = np.asarray(points_xyz, dtype=np.float64)
    if pts.size == 0:
        return set()
    ijk = np.floor(pts / float(voxel_m)).astype(np.int32, copy=False)
    ijk = np.ascontiguousarray(ijk)
    packed = ijk.view(np.dtype((np.void, ijk.dtype.itemsize * 3))).reshape(-1)
    return {bytes(v) for v in packed}


def stamp_from_path(path: str) -> Optional[float]:
    """
    Extract a numeric timestamp from a cloud filename.

    Recognises:
      - cloud_<stamp>_gt.pcd
      - cloud_gt_<stamp>.pcd
      - cloud_<stamp>.pcd

    Returns None if no pattern matches.
    """
    base = os.path.basename(path)
    for regex in (_STAMP_RE_SUFFIX, _STAMP_RE_PREFIX, _STAMP_RE_PLAIN):
        m = regex.search(base)
        if m:
            return float(m.group("stamp"))
    return None


def sorted_clouds(
    run_dir: str,
    pattern: str = "cloud_*_gt.pcd",
) -> List[Tuple[float, str]]:
    """
    Discover PCD files in *run_dir* matching *pattern*, sorted by timestamp.

    Returns a list of (stamp, path) tuples.
    """
    paths = glob.glob(os.path.join(run_dir, pattern))
    result: List[Tuple[float, str]] = []
    for p in paths:
        s = stamp_from_path(p)
        if s is not None:
            result.append((s, p))
    result.sort(key=lambda t: t[0])
    return result
