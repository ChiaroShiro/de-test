#!/usr/bin/env python3
"""GB200 双卡：原始 P2P、先复制后 DE、远端输入直接 DE。

Python 只使用标准库；GPU 工作由同目录 bench_peer_de.cu 完成。
首次运行自动用 nvcc 构建。需要 Python >= 3.9、CUDA Toolkit >= 12.8，
以及完整 nvCOMP C/C++ SDK（建议 >= 5.2；头文件与动态库必须匹配）。当前算法为 LZ4。

示例（Linux / GB200）：
  export NVCOMP_ROOT=/opt/nvidia/nvcomp
  python bench_peer_de.py --pattern repeated --mb 64 --src 0 --dst 1
  python bench_peer_de.py --model-dir "$MODEL" --max-shards 1 --max-segments 2
  python bench_peer_de.py --model-dir "$MODEL" --chunks 65536,262144 --csv peer.csv

不做软件后端回退。直接 peer-DE 失败会保留后端错误并以非零状态退出。
模型按 safetensors 的原始 tensor payload 分段读取；磁盘和 stdin 传输不计入 GPU 后端计时。
全模型汇总是各段测量中位数之和，不是跨段流水实测。

CSV 中各耗时均为秒、中位数，并附 min/max：
  raw_copy       原始数据直接 P2P 复制
  prepare        GPU 压缩 + 长度回读 + 偏移准备 + GPU 压缩块打包
  peer_de        GPU1 DE 直接读取 GPU0 输入，无接收端压缩载荷暂存
  staged_de      压缩载荷 P2P 复制 + GPU1 本地 DE（串行对照）
  peer_e2e       每轮 prepare + peer_de 的实测总时间
  staged_e2e     每轮 prepare + staged_de 的实测总时间
不包含一次性的内存分配、文件读取、CPU→GPU 原始输入准备和结果校验。
peer 源映射不被接受时，进程失败退出，不报告假的融合成绩。
"""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import re
import shutil
import struct
import subprocess
import sys
import sysconfig


MIB = 1 << 20
STAGES = ("raw_copy", "prepare", "peer_de", "staged_de", "compressed_copy",
          "local_de", "peer_e2e", "staged_e2e")
METRIC_FIELDS = [
    "raw_bytes", "payload_bytes", "staged_wire_bytes", "chunk_bytes", "chunks",
    "ratio", "peer_de_output_GBps", "raw_copy_GBps", "peer_receive_speedup",
    "peer_payload_staging_bytes", "peer_payload_memcpy_calls", "verified",
] + [stage + suffix for stage in STAGES for suffix in ("_s", "_min_s", "_max_s")]
CSV_FIELDS = ["input", "segment", "payload_offset", "src", "dst", "src_gpu",
              "dst_gpu", "cuda_headers", "cuda_runtime", "cuda_driver", "iters",
              "warmup", "nvcomp_headers"] + METRIC_FIELDS + ["result_json"]


def positive(value):
    value = int(value)
    if value <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return value


def nonnegative(value):
    value = int(value)
    if value < 0:
        raise argparse.ArgumentTypeError("不能为负数")
    return value


def parse_chunks(value):
    try:
        chunks = list(dict.fromkeys(int(x.strip()) for x in value.split(",")))
    except ValueError as ex:
        raise argparse.ArgumentTypeError("--chunks 使用逗号分隔的字节数") from ex
    if not chunks or any(x < 16 or x % 16 != 0 or x > 4 * MIB for x in chunks):
        raise argparse.ArgumentTypeError("每个 chunk 必须为 16 字节倍数，范围为 16 字节～4 MiB")
    return chunks


def parser():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-dir", default=os.environ.get("MODEL"),
                   help="safetensors 模型目录；默认 MODEL 环境变量")
    p.add_argument("--pattern", choices=("zeros", "random", "repeated", "sparse"),
                   help="合成输入；显式指定时忽略 MODEL 环境变量")
    p.add_argument("--mb", type=positive, default=256, help="合成输入总量，MiB")
    p.add_argument("--seg-mb", type=positive, default=256, help="外层段大小，MiB")
    p.add_argument("--max-shards", type=nonnegative, default=0,
                   help="最多均匀抽取多少分片；0=全部")
    p.add_argument("--max-segments", type=nonnegative, default=0,
                   help="所有分片合计最多处理多少段；0=全部")
    p.add_argument("--src", type=nonnegative, default=0)
    p.add_argument("--dst", type=nonnegative, default=1)
    p.add_argument("--chunks", type=parse_chunks, default=parse_chunks("65536"))
    p.add_argument("--iters", type=positive, default=5)
    p.add_argument("--warmup", type=nonnegative, default=2)
    p.add_argument("--no-e2e", action="store_true", help="不测每轮重新压缩的端到端路径")
    p.add_argument("--csv", default="peer_de_results.csv")
    p.add_argument("--cuda-home", help="CUDA Toolkit 根目录，默认 CUDA_HOME/CUDA_PATH 或 PATH")
    p.add_argument("--nvcomp-root", help="nvCOMP C/C++ SDK 根目录，默认 NVCOMP_ROOT")
    p.add_argument("--nvcomp-include", help="包含 nvcomp/lz4.h 的目录")
    p.add_argument("--nvcomp-lib", help="包含 libnvcomp.so 的目录，或该库的绝对路径")
    p.add_argument("--binary", help="使用已编译的后端，跳过自动构建")
    p.add_argument("--build-only", action="store_true", help="只构建后端，无须模型或 GPU 运行")
    return p


def unique_paths(items):
    seen = set()
    for item in items:
        if not item:
            continue
        path = Path(item).expanduser().resolve()
        if path not in seen:
            seen.add(path)
            yield path


def sdk_roots(args):
    candidates = [args.nvcomp_root, os.environ.get("NVCOMP_ROOT"),
                  os.environ.get("CONDA_PREFIX"), sys.prefix,
                  "/opt/nvidia/nvcomp", "/usr/local/nvcomp", "/usr/local", "/usr"]
    package_dirs = list(sys.path) + list(sysconfig.get_paths().values())
    for parent in unique_paths(package_dirs):
        candidates.extend([parent / "nvidia" / "nvcomp", parent / "nvidia" / "libnvcomp",
                           parent / "nvcomp", parent / "nvidia" / "nvcomp" / "libnvcomp"])
    return list(unique_paths(candidates))


def find_nvcomp(args):
    roots = sdk_roots(args)
    include_candidates = ([Path(args.nvcomp_include).expanduser()] if args.nvcomp_include else
                          [r / sub for r in roots for sub in ("include", "")])
    include = next((p.resolve() for p in include_candidates
                    if (p / "nvcomp" / "lz4.h").is_file()), None)
    if include is None:
        raise RuntimeError("找不到 nvCOMP 头文件 nvcomp/lz4.h。请安装 C/C++ SDK，"
                           "并指定 --nvcomp-root 或 --nvcomp-include；只有 Python 绑定可能不含头文件。")
    missing_headers = [name for name in ("nvcomp.h", "nvcomp/version.h", "nvcomp/export.h")
                       if not (include / name).is_file()]
    if missing_headers:
        raise RuntimeError("nvCOMP 头文件不完整，缺少 " + ", ".join(missing_headers) +
                           "。请使用完整 C/C++ SDK（建议 >=5.2），或指定正确的 --nvcomp-include。")
    if args.nvcomp_lib:
        lib_candidates = [Path(args.nvcomp_lib).expanduser()]
    else:
        lib_candidates = [r / sub for r in roots for sub in
                          ("lib", "lib64", "lib/aarch64-linux-gnu", "lib/x86_64-linux-gnu", "")]
    for loc in unique_paths(lib_candidates):
        if loc.is_file() and loc.name.startswith("libnvcomp.so"):
            return include, loc
        if loc.is_dir():
            libraries = sorted(loc.glob("libnvcomp.so*"),
                               key=lambda x: (x.name != "libnvcomp.so", len(x.name), x.name))
            for lib in libraries:
                if lib.is_file():
                    return include, lib.absolute()
    raise RuntimeError("找不到 libnvcomp.so。请用 --nvcomp-lib 指定库目录或 .so 文件。")


def build_backend(args):
    if args.binary:
        executable = Path(args.binary).expanduser().resolve()
        if not executable.is_file():
            raise RuntimeError("找不到 --binary: " + str(executable))
        return executable
    if sys.platform != "linux":
        raise RuntimeError("自动构建面向 Linux GB200 服务器；请将 .py 和 .cu 一起复制到服务器运行。")
    cuda_home = args.cuda_home or os.environ.get("CUDA_HOME") or os.environ.get("CUDA_PATH")
    nvcc = str(Path(cuda_home).expanduser() / "bin" / "nvcc") if cuda_home else shutil.which("nvcc")
    if not nvcc or not Path(nvcc).is_file():
        raise RuntimeError("找不到 nvcc；请配置 CUDA_HOME/PATH，或使用 --cuda-home。")
    nvcc = str(Path(nvcc).resolve())
    probe = subprocess.run([nvcc, "--version"], capture_output=True, text=True)
    version = re.search(r"release\s+(\d+)\.(\d+)", probe.stdout + probe.stderr)
    if probe.returncode or not version or tuple(map(int, version.groups())) < (12, 8):
        raise RuntimeError("需要 CUDA Toolkit >= 12.8（含 Blackwell / DE 接口）。\n" +
                           probe.stdout + probe.stderr)
    include, library = find_nvcomp(args)
    toolkit = Path(nvcc).parent.parent
    source = Path(__file__).resolve().with_suffix(".cu")
    if not source.is_file():
        raise RuntimeError("请将同目录 bench_peer_de.cu 一起复制到服务器。")
    builddir = source.parent / ".build_peer_de"
    builddir.mkdir(exist_ok=True)
    executable = builddir / "bench_peer_de"
    cmd = [nvcc, "-std=c++17", "-O3", "-arch=sm_100", str(source),
           "-I" + str(include), "-L" + str(library.parent),
           "-Xlinker", "-rpath", "-Xlinker", str(library.parent)]
    # CUDA stubs 仅供链接，不加入运行期 rpath；运行时必须使用服务器的真实驱动。
    for stub in (toolkit / "lib64/stubs", toolkit / "targets/aarch64-linux/lib/stubs",
                 toolkit / "targets/sbsa-linux/lib/stubs", toolkit / "targets/x86_64-linux/lib/stubs"):
        if (stub / "libcuda.so").is_file():
            cmd.append("-L" + str(stub))
            break
    cmd.extend(["-lnvcomp" if library.name == "libnvcomp.so" else str(library),
                "-lcuda", "-o", str(executable)])
    library_stamp = (str(library), library.stat().st_size, library.stat().st_mtime_ns)
    fingerprint = hashlib.sha256((source.read_text(encoding="utf-8") + "\n" +
                                  json.dumps(cmd) + "\n" + json.dumps(library_stamp) +
                                  "\n" + probe.stdout).encode())
    for relpath in ("nvcomp/lz4.h", "nvcomp.h", "nvcomp/version.h",
                    "nvcomp/shared_types.h", "nvcomp/nvcomp.h"):
        header = include / relpath
        fingerprint.update(relpath.encode())
        fingerprint.update(header.read_bytes() if header.is_file() else b"<absent>")
    signature = fingerprint.hexdigest()
    stamp = builddir / "build.sha256"
    if executable.is_file() and stamp.is_file() and stamp.read_text().strip() == signature:
        print("复用后端:", executable, flush=True)
        return executable
    print("构建后端:", " ".join(cmd), flush=True)
    proc = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    if proc.stdout:
        print(proc.stdout, end="" if proc.stdout.endswith("\n") else "\n")
    if proc.stderr:
        print(proc.stderr, file=sys.stderr, end="" if proc.stderr.endswith("\n") else "\n")
    if proc.returncode:
        raise RuntimeError("后端编译失败。请检查 CUDA/nvCOMP 版本以及头文件与库是否来自同一安装。")
    stamp.write_text(signature + "\n", encoding="ascii")
    return executable


def no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("safetensors header 存在重复键: " + str(key))
        result[key] = value
    return result


def safetensors_payload(path):
    """只读 header，返回 tensor 数据区起点和长度；验证所有范围恰好覆盖该区域。"""
    size = path.stat().st_size
    with path.open("rb") as f:
        first = f.read(8)
        if len(first) != 8:
            raise ValueError(f"{path}: safetensors 文件头不足 8 字节")
        header_size = struct.unpack("<Q", first)[0]
        if not 2 <= header_size <= 100_000_000 or 8 + header_size > size:
            raise ValueError(f"{path}: 非法 safetensors header 长度")
        header = json.loads(f.read(header_size).decode("utf-8"), object_pairs_hook=no_duplicate_keys)
    if not isinstance(header, dict):
        raise ValueError(f"{path}: header 不是 JSON 对象")
    ranges = []
    for name, item in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(item, dict):
            raise ValueError(f"{path}: tensor {name} 信息不合法")
        offsets = item.get("data_offsets")
        if (not isinstance(offsets, list) or len(offsets) != 2 or
                any(type(x) is not int for x in offsets) or
                not 0 <= offsets[0] <= offsets[1]):
            raise ValueError(f"{path}: tensor {name} 的 data_offsets 不合法")
        ranges.append((offsets[0], offsets[1], name))
    expected = 0
    for start, end, name in sorted(ranges):
        if start != expected:
            raise ValueError(f"{path}: tensor {name} 存在重叠/空洞，预期偏移 {expected}，实际 {start}")
        expected = end
    if expected != size - 8 - header_size:
        raise ValueError(f"{path}: tensor 范围没有完整覆盖文件 payload")
    return 8 + header_size, expected


def model_segments(args):
    root = Path(args.model_dir).expanduser()
    files = sorted(p for p in root.glob("*.safetensors") if p.is_file())
    if not files:
        raise RuntimeError(f"{root}: 没有 .safetensors 分片")
    if 0 < args.max_shards < len(files):
        files = [files[i * len(files) // args.max_shards] for i in range(args.max_shards)]
    count = 0
    for path in files:
        base, length = safetensors_payload(path)
        with path.open("rb") as f:
            f.seek(base)
            for offset in range(0, length, args.seg_mb * MIB):
                if args.max_segments and count >= args.max_segments:
                    return
                nbytes = min(args.seg_mb * MIB, length - offset)
                payload = f.read(nbytes)
                if len(payload) != nbytes:
                    raise RuntimeError(f"{path}: 读取期间文件被截断")
                count += 1
                yield str(path), count, offset, payload


def synthetic_segments(args):
    pattern = args.pattern or "repeated"
    rng = random.Random(20261115)
    period = b"Blackwell peer DE LZ4 benchmark. "
    total = args.mb * MIB
    for number, offset in enumerate(range(0, total, args.seg_mb * MIB), 1):
        if args.max_segments and number > args.max_segments:
            return
        size = min(args.seg_mb * MIB, total - offset)
        if pattern == "zeros":
            payload = bytes(size)
        elif pattern == "random":
            payload = rng.randbytes(size)
        elif pattern == "repeated":
            shift = offset % len(period)
            payload = (period * ((size + shift + len(period) - 1) // len(period)))[shift:shift + size]
        else:
            data = bytearray(size)
            for pos in rng.sample(range(size), max(1, size // 1000)):
                data[pos] = rng.randrange(1, 256)
            payload = bytes(data)
        yield "synthetic:" + pattern, number, offset, payload


def run_segment(binary, args, chunk, payload):
    cmd = [str(binary), "--stdin-bytes", str(len(payload)), "--src", str(args.src),
           "--dst", str(args.dst), "--chunk", str(chunk), "--iters", str(args.iters),
           "--warmup", str(args.warmup)]
    if args.no_e2e:
        cmd.append("--no-e2e")
    result = subprocess.run(cmd, input=payload, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stderr = result.stderr.decode("utf-8", errors="replace")
    stdout = result.stdout.decode("utf-8", errors="replace")
    if stderr:
        print(stderr, file=sys.stderr, end="" if stderr.endswith("\n") else "\n", flush=True)
    if result.returncode:
        raise RuntimeError(f"后端退出码 {result.returncode}；未回退。\n{stdout}".rstrip())
    try:
        report = json.loads(stdout)
    except json.JSONDecodeError as ex:
        raise RuntimeError("后端未返回单个有效 JSON：\n" + stdout[-8000:]) from ex
    if not isinstance(report, dict) or report.get("status") != "ok" or report.get("verified") is not True:
        raise RuntimeError("后端没有确认成功与正确性：" + stdout)
    if report.get("raw_bytes") != len(payload) or report.get("chunk_bytes") != chunk:
        raise RuntimeError("后端报告的输入大小/分块与请求不一致")
    if report.get("peer_payload_memcpy_calls") != 0 or report.get("peer_payload_staging_bytes") != 0:
        raise RuntimeError("后端 peer 路径包含 payload 拷贝/暂存，不计为直接解压")
    return report


def open_csv(path):
    path = Path(path).expanduser()
    if path.exists() and path.stat().st_size:
        with path.open(newline="", encoding="utf-8") as f:
            if next(csv.reader(f), None) != CSV_FIELDS:
                raise RuntimeError(f"{path}: CSV 表头不同，请换一个输出文件名")
        new = False
    else:
        new = True
    f = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(f, fieldnames=CSV_FIELDS, extrasaction="ignore")
    if new:
        writer.writeheader()
        f.flush()
    return f, writer


def main(argv=None):
    args = parser().parse_args(argv)
    if args.src == args.dst:
        raise RuntimeError("--src 与 --dst 必须是不同 GPU")
    binary = build_backend(args)
    if args.build_only:
        print("后端就绪:", binary)
        return 0
    print("GPU 索引受 CUDA_VISIBLE_DEVICES 影响；P2P 成功不等于已经证明链路为 NVLink。", flush=True)
    print("计时为后端主机墙钟中位数，含对应操作的提交/同步；不含读盘和 stdin。", flush=True)
    segments = synthetic_segments(args) if args.pattern or not args.model_dir else model_segments(args)
    sums = {}
    csv_file, writer = open_csv(args.csv)
    try:
        for label, index, offset, payload in segments:
            for chunk in args.chunks:
                print(f"[{index}] {label} offset={offset} bytes={len(payload)} chunk={chunk}", flush=True)
                report = run_segment(binary, args, chunk, payload)
                row = dict(report)
                row.update(input=label, segment=index, payload_offset=offset,
                           result_json=json.dumps(report, ensure_ascii=False, separators=(",", ":")))
                writer.writerow(row)
                csv_file.flush()
                total = sums.setdefault(chunk, {"raw_bytes": 0, "payload_bytes": 0, "segments": 0})
                total["segments"] += 1
                for key in ("raw_bytes", "payload_bytes"):
                    total[key] += report[key]
                for stage in STAGES:
                    key = stage + "_s"
                    if key in report:
                        total[key] = total.get(key, 0.0) + report[key]
                print(f"  PASS ratio={report['ratio']:.3f} "
                      f"raw={report['raw_copy_s'] * 1000:.3f}ms "
                      f"peer-DE={report['peer_de_s'] * 1000:.3f}ms "
                      f"staged-DE={report['staged_de_s'] * 1000:.3f}ms", flush=True)
                if "peer_e2e_s" in report:
                    print(f"  在线压缩+peer-DE={report['peer_e2e_s'] * 1000:.3f}ms "
                          f"在线压缩+staged-DE={report['staged_e2e_s'] * 1000:.3f}ms", flush=True)
    finally:
        csv_file.close()
    if not sums:
        raise RuntimeError("没有非空 tensor payload 可测试")
    print("\n分段汇总（各段中位数之和；不是跨段流水实测）：")
    for chunk, total in sums.items():
        raw = total["raw_bytes"]
        print(f"chunk={chunk} segments={total['segments']} raw={raw / 1e9:.3f}GB "
              f"ratio={raw / total['payload_bytes']:.3f}")
        for stage in STAGES:
            duration = total.get(stage + "_s")
            if duration is not None:
                print(f"  {stage:16s} {duration:.6f}s")
        print(f"  接收路径 speedup(raw/peer-DE)={total['raw_copy_s'] / total['peer_de_s']:.3f}")
        if "peer_e2e_s" in total:
            print(f"  在线总路径 speedup(raw/peer-e2e)={total['raw_copy_s'] / total['peer_e2e_s']:.3f}")
    print("CSV:", Path(args.csv).resolve())
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print("ERROR:", exc, file=sys.stderr)
        raise SystemExit(2)
