#!/usr/bin/env python3
# A) NVLink 直传权重   vs   B) SM 压缩 -> NVLink 传压缩数据 -> DE（或 SM）解压
#
# 用法示例：
#   冒烟测试: python bench_de_xfer.py --model-dir /models/xxx --algos LZ4 --chunks 65536 --backends hw --max-shards 1 --iters 3
#   抽样扫参: python bench_de_xfer.py --model-dir /models/xxx --max-shards 4 --iters 3 --csv sweep.csv
#   全模型:   python bench_de_xfer.py --model-dir /models/xxx --algos LZ4,Deflate --chunks 1048576 --csv full.csv
import argparse, csv, glob, itertools, os, time
import torch
from safetensors.torch import load_file
from nvidia import nvcomp

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", required=True)
ap.add_argument("--algos", default="LZ4,Snappy,Deflate,Gzip")       # 只有这四种能走 DE
ap.add_argument("--chunks", default="65536,262144,1048576,4194304")  # DE 单块上限 4 MiB
ap.add_argument("--backends", default="hw,cuda")   # hw=只走 DE，不满足条件直接报错；cuda=强制用 SM（对照组）
ap.add_argument("--src", type=int, default=0)
ap.add_argument("--dst", type=int, default=1)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--max-shards", type=int, default=0)  # 0=全部分片；>0 则均匀抽取这么多个分片
ap.add_argument("--no-out", action="store_true",
                help="decode 不传 out，由 nvcomp 自己分配输出（hw 后端报错时可以试试）")
ap.add_argument("--seg-mb", type=int, default=512,
                help="每个分片再切成多大的段去压缩/解压（MB），避免单次调用申请过多显存")
ap.add_argument("--csv", default="results.csv")
args = ap.parse_args()

G0, G1 = torch.device(f"cuda:{args.src}"), torch.device(f"cuda:{args.dst}")
assert torch.cuda.can_device_access_peer(args.src, args.dst), "两卡之间不能 P2P，先检查 NVLink fabric"


def find_backend(key):            # 不同 nvcomp 版本的枚举名可能不同，这里自动查找
    for name in dir(nvcomp):
        members = getattr(getattr(nvcomp, name), "__members__", None)
        if members and "backend" in name.lower():
            for k, v in members.items():
                if key in k.upper():
                    return v
    raise RuntimeError(f"找不到 {key} 后端枚举，用 help(nvcomp.Codec) 查看后手动填写")


BK = {"hw": "HARDWARE", "cuda": "CUDA"}
algos = args.algos.split(",")
chunks = [int(c) for c in args.chunks.split(",")]
backends = args.backends.split(",")
enc = {(a, c): nvcomp.Codec(algorithm=a, uncomp_chunk_size=c, device_id=args.src)
       for a in algos for c in chunks}
dec = {(a, c, b): nvcomp.Codec(algorithm=a, uncomp_chunk_size=c, device_id=args.dst,
                               decompress_backend=find_backend(BK[b]))
       for a in algos for c in chunks for b in backends}
print("nvcomp", nvcomp.__version__, "| torch", torch.__version__, "| CUDA", torch.version.cuda)
for g in (G0, G1):
    free, total = torch.cuda.mem_get_info(g)
    print(f"{g}: 空闲显存 {free/2**30:.1f} / {total/2**30:.1f} GiB")


def sync():
    torch.cuda.synchronize(G0); torch.cuda.synchronize(G1)


def bench(fn):                    # 预热 1 次，然后取 iters 次的中位数（秒）
    out = fn(); sync(); ts = []
    for _ in range(args.iters):
        out = None                # 先释放上一轮的输出，避免显存占用翻倍（不计时）
        sync(); t0 = time.perf_counter(); out = fn(); sync()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2], out


def as_u8(arr, dev):              # nvcomp.Array 转成 torch uint8 张量（零拷贝）
    try:
        t = torch.from_dlpack(arr)
    except Exception:
        t = torch.as_tensor(arr, device=dev)
    return t.view(torch.uint8).reshape(-1)


shards = sorted(glob.glob(os.path.join(args.model_dir, "*.safetensors")))
assert shards, f"{args.model_dir} 下没有 .safetensors 文件"
if args.max_shards:
    shards = shards[::max(1, len(shards) // args.max_shards)][:args.max_shards]

raw_total, tA_total, acc = 0, 0.0, {}
for i, f in enumerate(shards):
    ts = load_file(f)                                                    # 读盘不计时
    src = torch.cat([t.reshape(-1).view(torch.uint8) for t in ts.values()]).to(G0)
    del ts
    ref = src.to(G1)                                                     # 校验用的参考数据
    dst = torch.empty_like(ref)
    tA, _ = bench(lambda: dst.copy_(src, non_blocking=True))             # 方案 A：NVLink 直传
    raw_total += src.numel(); tA_total += tA
    print(f"[{i+1}/{len(shards)}] {os.path.basename(f)} {src.numel()/1e9:.2f} GB  A={tA*1e3:.2f} ms")

    seg = args.seg_mb << 20                                              # 每段字节数
    offs = range(0, src.numel(), seg)
    ssegs = [src[o:o + seg] for o in offs]                               # 发送端原始数据分段（视图，不拷贝）
    dsegs = [dst[o:o + seg] for o in offs]                               # 接收端输出分段（视图，不拷贝）

    for a, c in itertools.product(algos, chunks):
        e = enc[a, c]
        tc, comps = bench(lambda: [e.encode(nvcomp.as_array(x)) for x in ssegs])        # B1：SM 压缩
        csrcs = [as_u8(x, G0) for x in comps]
        cdsts = [torch.empty_like(x, device=G1) for x in csrcs]
        tx, _ = bench(lambda: [y.copy_(x, non_blocking=True) for x, y in zip(csrcs, cdsts)])  # B2：传压缩数据
        cbytes = sum(x.numel() for x in csrcs)
        s = acc.setdefault((a, c), [0, 0.0, 0.0]); s[0] += cbytes; s[1] += tc; s[2] += tx
        line = f"    {a:7s} chunk={c:>7d} ratio={src.numel()/cbytes:.3f} c={tc*1e3:.1f} x={tx*1e3:.2f}"
        for b in backends:
            d = dec[a, c, b]
            if args.no_out:                                               # B3：解压（nvcomp 自己分配输出）
                td, res = bench(lambda: [d.decode(nvcomp.as_array(x)) for x in cdsts])
                out_t = torch.cat([as_u8(r, G1) for r in res])[:ref.numel()]
            else:                                                         # B3：解压到预分配的 dst
                dst.zero_()                                               # 清零，防止旧数据掩盖解压失败
                td, _ = bench(lambda: [d.decode(nvcomp.as_array(x), out=nvcomp.as_array(y))
                                       for x, y in zip(cdsts, dsegs)])
                out_t = dst
            assert torch.equal(out_t, ref), f"{a}/{c}/{b} 解压结果和原始数据不一致"   # 无损校验
            acc[(a, c, b)] = acc.get((a, c, b), 0.0) + td
            line += f" d[{b}]={td*1e3:.1f}"
        print(line + " ms")
        del comps, csrcs, cdsts
    del src, ref, dst, ssegs, dsegs
    torch.cuda.empty_cache()

GB = raw_total / 1e9
print(f"\n总量 {GB:.1f} GB | A 直传 {tA_total:.3f}s ({GB/tA_total:.0f} GB/s)")
new = not os.path.exists(args.csv)
with open(args.csv, "a", newline="") as fp:
    w = csv.writer(fp)
    if new:
        w.writerow(["algo", "chunk", "backend", "raw_GB", "ratio", "A_s", "comp_s", "xfer_s", "decomp_s",
                    "B_serial_s", "B_pipe_s", "speedup_serial", "speedup_pipe", "comp_GBps", "decomp_GBps"])
    for a, c in itertools.product(algos, chunks):
        cb, tc, tx = acc[(a, c)]
        for b in backends:
            td = acc[(a, c, b)]
            ser, pipe = tc + tx + td, max(tc, tx, td)
            print(f"{a:7s} chunk={c:>7d} {b:4s} 压缩率={raw_total/cb:.3f}  B串行={ser:.3f}s  B流水≈{pipe:.3f}s  "
                  f"加速比(串行/流水)={tA_total/ser:.2f}/{tA_total/pipe:.2f}  "
                  f"压缩 {GB/tc:.0f} GB/s  解压 {GB/td:.0f} GB/s")
            w.writerow([a, c, b, f"{GB:.3f}", f"{raw_total/cb:.4f}", f"{tA_total:.4f}", f"{tc:.4f}",
                        f"{tx:.4f}", f"{td:.4f}", f"{ser:.4f}", f"{pipe:.4f}",
                        f"{tA_total/ser:.3f}", f"{tA_total/pipe:.3f}", f"{GB/tc:.1f}", f"{GB/td:.1f}"])
