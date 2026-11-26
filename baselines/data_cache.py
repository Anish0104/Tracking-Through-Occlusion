"""Exact, memory-efficient on-disk cache for generated data splits.

Goal: train the three model seeds on 5,000 + 1,000 videos without holding the
split in RAM (or ever putting it on the GPU), while guaranteeing that the cached
arrays are bit-for-bit what `make_batch(n, seed)` returns, in the same order.

How a split is produced (first that applies)
    1. public hook   the generator module defines iter_videos(n, seed), yielding one
                     {frames [T,64,64], xy [T,2], visible_frac [T]} dict per video in
                     exactly make_batch(n, seed) order. Incremental: peak RAM ~ one video.
                     (Suggested to Krisha for data/generator.py; nothing requires it.)
    2. adapter       an explicitly registered, OPTIONAL adapter in ADAPTERS below. The
                     only one is for data/starter.py and it relies on that module's
                     private _one_video(rng); see _iter_starter. Incremental.
    3. fallback      make_batch(n, seed) once, written out, then freed. Always exact,
                     but peak RAM is whatever the generator itself needs.
    Paths 1 and 2 are checked against make_batch(k, seed) on a small k before every
    build, so a generator change that breaks the hook/adapter fails loudly instead of
    silently caching different data.

Storage (one directory per split)
    frames         "auto" (default): lossless uint8 palette codes when the frames hold at
                   most 256 distinct values (the starter has 3: 0, 0.4, 1), else the
                   original dtype. The palette is kept as raw bit patterns, so decoding
                   is bitwise exact (-0.0, NaN, every float32 value preserved). If a later
                   video exceeds 256 values the codes written so far are decoded into a
                   raw array and writing continues raw; nothing is regenerated.
                   "uint8": palette or error. "raw": original dtype always.
    xy, visible_frac   original dtype, as .npy.
    Everything is a .npy file opened with mmap_mode="r"; pages are read on demand.

Validity
    A split is built in a hidden ".<name>.partial-<pid>-<id>" directory. Checksums and
    a manifest are written last, then the directory is renamed into place in one
    atomic step, so a directory with the final name is always complete. An interrupted
    build leaves only a partial directory, which is never read; the next build removes
    partial directories of the same split whose process is no longer running.
    Loading re-checks the manifest, shapes, dtypes and (by default) SHA-256 checksums.

Identity
    The directory name contains n, seed and a fingerprint (SHA-256) of the generator's
    source file. Editing the generator therefore yields a new cache directory; old ones
    are never deleted automatically. The model seed is not part of the key, so seeds
    0, 1 and 2 share one cached dataset.

CLI (build the shared train/validation caches once, before training):
    python -m baselines.data_cache --data starter --cache-dir checkpoints/data_cache
"""

import argparse
import hashlib
import importlib
import inspect
import json
import os
import shutil
import time
import uuid
from pathlib import Path

import numpy as np

DATA_KEYS = ("frames", "xy", "visible_frac")
DATA_SOURCES = {"starter": "data.starter", "generator": "data.generator"}
TRAIN_SEED, VAL_SEED, TEST_SEED = 100, 200, 300      # CONTRACT.md section 4
TRAIN_SIZE, VAL_SIZE = 5000, 1000
FORMATS = ("auto", "uint8", "raw")
FORMAT_VERSION = 1
MAX_CODES = 256
VERIFY_PREFIX = 4          # videos compared against make_batch before an incremental build


class CacheError(RuntimeError):
    pass


# ---------------------------------------------------------------- sources

def resolve_module(source):
    """'starter', 'generator' or a dotted module path -> the generator module."""
    name = DATA_SOURCES.get(source, source)
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as e:
        if e.name != name:
            raise                                  # a dependency is missing: keep the real error
        hint = " (data/generator.py has not been merged yet)" if name == "data.generator" else ""
        raise ModuleNotFoundError(f"data source '{source}': no module {name}{hint}", name=name) from None


def source_fingerprint(module):
    """SHA-256 of the generator's source file. Helpers imported from other files are
    not covered; bump something in the generator file if only those change."""
    path = inspect.getsourcefile(module)
    if path is None:
        raise CacheError(f"cannot fingerprint {module.__name__}: no source file")
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _iter_starter(module, n, seed):
    """OPTIONAL ADAPTER for data/starter.py (private API).

    data.starter.make_batch(n, seed) is literally
        rng = np.random.default_rng(seed); [_one_video(rng) for _ in range(n)]
    so drawing the videos one at a time from the same generator reproduces it
    exactly with one video in memory. If starter.py changes, the prefix check in
    _checked_iterator catches the mismatch before anything is cached.
    """
    rng = np.random.default_rng(seed)
    for _ in range(n):
        frames, xy, vis = module._one_video(rng)
        yield {"frames": frames, "xy": xy, "visible_frac": vis}


ADAPTERS = {"data.starter": _iter_starter}


def incremental_producer(module):
    """Return (name, iterator factory) or (None, None) when only make_batch exists."""
    if callable(getattr(module, "iter_videos", None)):
        return "iter_videos", lambda n, seed: module.iter_videos(n, seed)
    adapter = ADAPTERS.get(module.__name__)
    if adapter is not None:
        return f"adapter:{module.__name__}", lambda n, seed: adapter(module, n, seed)
    return None, None


def _checked_iterator(module, factory, n, seed):
    """The incremental producer must agree exactly with make_batch on a small prefix."""
    k = min(n, VERIFY_PREFIX)
    ref = module.make_batch(n=k, seed=seed)
    got = list(factory(k, seed))
    if len(got) != k:
        raise CacheError(f"incremental producer yielded {len(got)} videos, expected {k}")
    for key in DATA_KEYS:
        stacked = np.stack([np.asarray(v[key]) for v in got])
        if stacked.dtype != np.asarray(ref[key]).dtype or not _bitwise_equal(stacked, ref[key]):
            raise CacheError(f"incremental producer disagrees with make_batch on '{key}'; "
                             "refusing to cache (use --cache-format raw with no adapter, or fix it)")
    return factory(n, seed)


def _batch_videos(batch):
    for i in range(len(batch["frames"])):
        yield {k: batch[k][i] for k in DATA_KEYS}


# ---------------------------------------------------------------- encoding

def _bits_dtype(dtype):
    return np.dtype(f"u{np.dtype(dtype).itemsize}")


def _bitwise_equal(a, b):
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    if a.dtype.kind in "fc":
        bits = _bits_dtype(a.dtype)
        return np.array_equal(np.ascontiguousarray(a).view(bits), np.ascontiguousarray(b).view(bits))
    return np.array_equal(a, b)


class PaletteEncoder:
    """Maps each distinct bit pattern to a uint8 code (palette in discovery order, so
    codes already written never change)."""

    def __init__(self, dtype):
        self.dtype = np.dtype(dtype)
        self.bits_dtype = _bits_dtype(self.dtype)
        self.bits = np.empty(0, self.bits_dtype)

    def encode(self, x):
        """Return uint8 codes for x, or None if the palette would exceed 256 entries."""
        b = np.ascontiguousarray(x).view(self.bits_dtype)
        values = np.unique(b)
        new = values[~np.isin(values, self.bits)]
        if len(self.bits) + len(new) > MAX_CODES:
            return None
        self.bits = np.concatenate([self.bits, new])
        order = np.argsort(self.bits, kind="stable")
        codes = order[np.searchsorted(self.bits[order], b)].astype(np.uint8)
        if not np.array_equal(self.bits[codes], b):            # cannot happen; belt and braces
            raise CacheError("palette encoding is not lossless")
        return codes

    @property
    def values(self):
        return self.bits.view(self.dtype)


class PaletteArray:
    """Read-only array view: uint8 codes (memmap) + palette -> original values on access."""

    def __init__(self, codes, palette):
        self.codes, self.palette = codes, np.asarray(palette)
        self.shape, self.ndim = codes.shape, codes.ndim
        self.dtype = self.palette.dtype

    def __len__(self):
        return len(self.codes)

    def __getitem__(self, idx):
        return self.palette[np.asarray(self.codes[idx])]

    def __array__(self, dtype=None, copy=None):
        out = self.palette[np.asarray(self.codes)]
        return out if dtype is None else out.astype(dtype)


# ---------------------------------------------------------------- build

def _sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while block := f.read(chunk):
            h.update(block)
    return h.hexdigest()


def _pid_alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _remove_dead_partials(cache_dir, name):
    """Remove our own interrupted build directories for this split (dead process only)."""
    removed = []
    for p in Path(cache_dir).glob(f".{name}.partial-*"):
        try:
            pid = int(p.name.rsplit(".partial-", 1)[1].split("-")[0])
        except ValueError:
            continue
        if not _pid_alive(pid):
            shutil.rmtree(p, ignore_errors=True)
            removed.append(p.name)
    return removed


def _write_split(videos, n, out_dir, fmt):
    """Stream videos into .npy memmaps in out_dir. Returns per-array manifest entries."""
    mm, info, enc = {}, {}, None
    count = 0
    for i, video in enumerate(videos):
        if i >= n:
            raise CacheError(f"producer yielded more than {n} videos")
        for key in DATA_KEYS:
            x = np.asarray(video[key])
            if i == 0:
                info[key] = {"dtype": x.dtype.str, "shape": [n, *x.shape], "encoding": "raw"}
                if key == "frames" and fmt in ("auto", "uint8"):
                    enc = PaletteEncoder(x.dtype)
                    info[key]["encoding"] = "palette_u8"
                    mm[key] = np.lib.format.open_memmap(out_dir / "frames_codes.npy", "w+",
                                                        np.uint8, (n, *x.shape))
                else:
                    mm[key] = np.lib.format.open_memmap(out_dir / f"{key}.npy", "w+",
                                                        x.dtype, (n, *x.shape))
            if x.dtype.str != info[key]["dtype"] or list(x.shape) != info[key]["shape"][1:]:
                raise CacheError(f"video {i} '{key}': {x.dtype} {x.shape} differs from video 0")
            if key == "frames" and info[key]["encoding"] == "palette_u8":
                codes = enc.encode(x)
                if codes is not None:
                    mm[key][i] = codes
                    continue
                if fmt == "uint8":
                    raise CacheError(f"frames have more than {MAX_CODES} distinct values; "
                                     "uint8 storage would be lossy (use --cache-format auto/raw)")
                mm[key] = _switch_to_raw(mm[key], enc, i, out_dir, n, x)
                info[key]["encoding"] = "raw"
            mm[key][i] = x
        count = i + 1
    if count != n:
        raise CacheError(f"producer yielded {count} videos, expected {n}")
    for key, a in mm.items():
        a.flush()
    entries = {}
    for key in DATA_KEYS:
        e = dict(info[key])
        if e["encoding"] == "palette_u8":
            np.save(out_dir / "frames_palette.npy", enc.values)
            e.update(file="frames_codes.npy", palette_file="frames_palette.npy")
        else:
            e["file"] = f"{key}.npy"
        entries[key] = e
    del mm
    return entries


def _switch_to_raw(codes_mm, enc, n_done, out_dir, n, first_raw, chunk=256):
    """Palette overflowed at video n_done: decode what is written (exact) into a raw array."""
    raw = np.lib.format.open_memmap(out_dir / "frames.npy", "w+", first_raw.dtype,
                                    (n, *first_raw.shape))
    values = enc.values
    for s in range(0, n_done, chunk):
        raw[s:min(s + chunk, n_done)] = values[np.asarray(codes_mm[s:min(s + chunk, n_done)])]
    del codes_mm
    (out_dir / "frames_codes.npy").unlink()
    return raw


def split_name(module, n, seed):
    return f"{module.__name__.replace('.', '_')}_n{n}_s{seed}_{source_fingerprint(module)[:12]}"


def build_split(source, n, seed, cache_dir, fmt="auto", log=print):
    """Build (or find) the cache for make_batch(n, seed). Returns its directory."""
    if fmt not in FORMATS:
        raise ValueError(f"cache format must be one of {FORMATS}")
    if seed == TEST_SEED:
        raise CacheError("data seed 300 is the held-out test bank; it is never cached here")
    module = resolve_module(source)
    cache_dir = Path(cache_dir)
    name = split_name(module, n, seed)
    final = cache_dir / name
    if (final / "manifest.json").exists():
        return final
    cache_dir.mkdir(parents=True, exist_ok=True)
    removed = _remove_dead_partials(cache_dir, name)
    if removed:
        log(f"removed interrupted cache build(s): {', '.join(removed)}")

    producer, factory = incremental_producer(module)
    t0 = time.time()
    if producer is not None:
        videos = _checked_iterator(module, factory, n, seed)
    else:
        producer = "make_batch"
        batch = module.make_batch(n=n, seed=seed)       # generator's own peak memory
        videos = _batch_videos(batch)
    log(f"building cache {name} ({producer}, format {fmt}) ...")

    tmp = cache_dir / f".{name}.partial-{os.getpid()}-{uuid.uuid4().hex[:8]}"
    tmp.mkdir()
    try:
        arrays = _write_split(videos, n, tmp, fmt)
        videos = batch = None                            # free the fallback batch early
        for e in arrays.values():
            e["sha256"] = _sha256(tmp / e["file"])
            if "palette_file" in e:
                e["palette_sha256"] = _sha256(tmp / e["palette_file"])
        manifest = {"format_version": FORMAT_VERSION, "module": module.__name__,
                    "fingerprint": source_fingerprint(module), "n": n, "seed": seed,
                    "producer": producer, "format": fmt, "arrays": arrays,
                    "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "complete": True}
        with open(tmp / "manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
            f.flush()
            os.fsync(f.fileno())
        try:
            os.rename(tmp, final)                        # atomic publish
        except OSError:
            if not (final / "manifest.json").exists():
                raise
            shutil.rmtree(tmp, ignore_errors=True)       # another process won the race
    except BaseException:
        # Leave nothing that could be mistaken for a cache. If this process is killed
        # instead, the partial directory is removed by the next build.
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    size = sum(p.stat().st_size for p in final.iterdir()) / 1e9
    log(f"cached {name}: {size:.2f} GB in {time.time() - t0:.1f}s")
    return final


# ---------------------------------------------------------------- load

def open_split(path, verify=True):
    """Open a complete cache directory as {frames, xy, visible_frac} (memmap-backed)."""
    path = Path(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text())
    except (OSError, ValueError) as e:
        raise CacheError(f"{path}: missing or unreadable manifest ({e})") from None
    if manifest.get("format_version") != FORMAT_VERSION or not manifest.get("complete"):
        raise CacheError(f"{path}: incomplete or incompatible cache")
    out = {}
    for key in DATA_KEYS:
        e = manifest["arrays"][key]
        files = [(e["file"], e["sha256"])]
        if "palette_file" in e:
            files.append((e["palette_file"], e["palette_sha256"]))
        for fname, digest in files:
            if not (path / fname).exists():
                raise CacheError(f"{path}: missing {fname}")
            if verify and _sha256(path / fname) != digest:
                raise CacheError(f"{path}: checksum mismatch in {fname}; delete this directory to rebuild")
        try:
            arr = np.load(path / e["file"], mmap_mode="r")
        except ValueError as err:
            raise CacheError(f"{path}: corrupt {e['file']} ({err})") from None
        if list(arr.shape) != e["shape"]:
            raise CacheError(f"{path}: {e['file']} has shape {arr.shape}, expected {e['shape']}")
        if e["encoding"] == "palette_u8":
            palette = np.load(path / e["palette_file"])
            if palette.dtype.str != e["dtype"] or arr.size and int(arr.max(initial=0)) >= len(palette):
                raise CacheError(f"{path}: invalid frame palette")
            out[key] = PaletteArray(arr, palette)
        else:
            if arr.dtype.str != e["dtype"]:
                raise CacheError(f"{path}: {e['file']} has dtype {arr.dtype}, expected {e['dtype']}")
            out[key] = arr
    return out


def get_split(source, n, seed, cache_dir, fmt="auto", verify=True, log=print):
    """Cached make_batch(n, seed): build once if needed, then open read-only."""
    return open_split(build_split(source, n, seed, cache_dir, fmt, log), verify=verify)


# ---------------------------------------------------------------- CLI

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="starter")
    p.add_argument("--cache-dir", default="checkpoints/data_cache")
    p.add_argument("--n-train", type=int, default=TRAIN_SIZE)
    p.add_argument("--n-val", type=int, default=VAL_SIZE)
    p.add_argument("--format", default="auto", choices=FORMATS)
    p.add_argument("--verify-full", action="store_true",
                   help="also compare every cached array with make_batch(n, seed) (needs the "
                        "generator's full memory once per split)")
    a = p.parse_args(argv)
    for n, seed in ((a.n_train, TRAIN_SEED), (a.n_val, VAL_SEED)):
        path = build_split(a.data, n, seed, a.cache_dir, a.format)
        split = open_split(path)                      # checksum verification
        print(f"verified checksums: {path}")
        if a.verify_full:
            ref = resolve_module(a.data).make_batch(n=n, seed=seed)
            ok = all(_bitwise_equal(np.asarray(split[k]), ref[k]) for k in DATA_KEYS)
            del ref
            print(f"bitwise equal to make_batch({n}, {seed}): {ok}")
            if not ok:
                raise SystemExit(1)


if __name__ == "__main__":
    main()
