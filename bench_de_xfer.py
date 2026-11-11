#!/usr/bin/env python3
# A) NVLink 直传权重   vs   B) SM 压缩 -> NVLink 传压缩数据 -> DE（或 SM）解压
#
# 用法示例（模型目录默认读环境变量 $MODEL，要先 export）：
#   export MODEL=/data/models/xxx
#   冒烟测试: python bench_de_xfer.py --algos LZ4 --chunks 65536 --backends hw --max-shards 1 --iters 3
#   抽样扫参: python bench_de_xfer.py --max-shards 4 --iters 3 --csv sweep.csv
#   全模型:   python bench_de_xfer.py --algos Snappy,LZ4,Deflate --chunks 65536 --iters 3 --csv full.csv
import argparse, csv, gc, glob, itertools, os, time
import torch
from safetensors.torch import load_file
from nvidia import nvcomp

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=os.environ.get("MODEL"), help="模型目录，默认读环境变量 $MODEL")
ap.add_argument("--algos", default="LZ4,Snappy,Deflate,Gzip")       # 只有这四种能走 DE
# NVIDIA 文档：各算法都推荐 64KiB 分块；Deflate 分块超过 64KiB 时压缩可能很慢、临时显存很大
ap.add_argument("--chunks", default="65536,262144,1048576")
ap.add_argument("--backends", default="hw,cuda")   # hw=只走 DE，不满足条件直接报错；cuda=强制用 SM（对照组）
ap.add_argument("--src", type=int, default=0)
ap.add_argument("--dst", type=int, default=1)
ap.add_argument("--iters", type=int, default=5)
ap.add_argument("--max-shards", type=int, default=0)  # 0=全部分片；>0 则均匀抽取这么多个分片
ap.add_argument("--seg-mb", type=int, default=4096,
                help="每个分片切成多大的段去压缩/解压（MB）。默认 4096 即整个分片一次处理，块数最多、并行度最高")
ap.add_argument("--no-out", action="store_true",
                help="decode 不传 out，由 nvcomp 自己分配输出（hw 后端报错时可以试试）")
ap.add_argument("--no-e2e", action="store_true", help="跳过 B 方案端到端实测（可以省一些时间）")
ap.add_argument("--csv", default="results.csv")
args = ap.parse_args()
if not args.model_dir:
    raise SystemExit("没有模型目录：先 export MODEL=模型目录，或者用 --model-dir 指定")

G0, G1 = torch.device(f"cuda:{args.src}"), torch.device(f"cuda:{args.dst}")
if not torch.cuda.can_device_access_peer(args.src, args.dst):
    raise SystemExit("两卡之间不能 P2P，先检查 NVLink fabric")


def find_backend(key):            # 不同 nvcomp 版本的枚举名可能不同，这里自动查找
    for name in dir(nvcomp):
        members = getattr(getattr(nvcomp, name), "__members__", None)
        if members and "backend" in name.lower():
            for k, v in members.items():
                if key in k.upper():
                    return v
    raise RuntimeError(f"找不到 {key} 后端枚举，用 help(nvcomp.Codec) 查看后手动填写")


BK = {"hw": find_backend("HARDWARE"), "cuda": find_backend("CUDA")}
algos = args.algos.split(",")
chunks = [int(c) for c in args.chunks.split(",")]
backends = args.backends.split(",")
for b in backends:
    if b not in BK:
        raise SystemExit(f"未知后端 {b}，只能是 hw 或 cuda")
print("nvcomp", nvcomp.__version__, "| torch", torch.__version__, "| CUDA", torch.version.cuda)


def mem():                        # 两卡当前空闲显存，用来观察有没有跨组合累积
    return " ".join(f"{g}空闲{torch.cuda.mem_get_info(g)[0] / 2**30:.1f}G" for g in (G0, G1))


print("启动时:", mem())


def sync():
    torch.cuda.synchronize(G0); torch.cuda.synchronize(G1)


def bench(fn):                    # 预热 1 次，然后取 iters 次的中位数（秒）
    # 开头先同步：nvCOMP 用自己的内部 stream，和 PyTorch 的 stream 之间没有自动依赖，
    # 必须等之前排队的 PyTorch 操作（拷贝、填充等）做完，再开始调用 nvCOMP
    sync()
    out = fn(); sync(); ts = []
    for _ in range(args.iters):
        out = None                # 先释放上一轮的输出，避免显存占用翻倍（不计时）
        sync(); t0 = time.perf_counter(); out = fn(); sync()
        ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2], out


def as_u8(arr):                   # nvcomp.Array -> torch uint8 一维张量（DLPack 零拷贝，带设备信息）
    return torch.from_dlpack(arr).view(torch.uint8).reshape(-1)


def on(dev, fn):                  # 把 fn 包装成"先把当前 GPU 切到 dev 再执行"
    # 实测：不切换的话，nvcomp.as_array 会把 GPU1 上的张量当成当前 GPU（GPU0）的，
    # 解压时报 "Input array and Codec device id mismatched"
    def run():
        with torch.cuda.device(dev):
            return fn()
    return run


cur = {"stage": ""}               # 记录当前进行到哪一步，显存不足时用来报告位置


def verify(decode, ref, dst):
    """计时之外单独解压一次并逐字节校验，避免计时轮次里旧的正确数据掩盖错误"""
    if args.no_out:
        sync(); res = decode(); sync()
        off = 0
        for r in res:
            t = as_u8(r)
            if not torch.equal(t, ref[off:off + t.numel()]):
                raise RuntimeError("解压结果和原始数据不一致")
            off += t.numel()
        if off != ref.numel():
            raise RuntimeError(f"解压总长度 {off} 和原始长度 {ref.numel()} 不一致")
    else:
        torch.bitwise_not(ref, out=dst)          # 先把目标填成与正确结果逐位相反的内容
        sync(); decode(); sync()
        if not torch.equal(dst, ref):
            raise RuntimeError("解压结果和原始数据不一致")


def b_path(enc, dec, ssegs, dsegs):
    """B 方案端到端：压缩 -> 取实际长度 -> 申请接收缓冲 -> NVLink 传输 -> 解压，完整串行做一遍"""
    with torch.cuda.device(G0):
        comps = [enc.encode(nvcomp.as_array(x)) for x in ssegs]
    torch.cuda.synchronize(G0)                   # 压缩在 nvCOMP 内部 stream 上，做完才能拷贝
    csrcs = [as_u8(x)[:x.buffer_size] for x in comps]
    cdsts = [torch.empty_like(x, device=G1) for x in csrcs]
    for x, y in zip(csrcs, cdsts):
        y.copy_(x, non_blocking=True)
    torch.cuda.synchronize(G1)                   # 传输在 PyTorch stream 上，做完才能交给 nvCOMP 解压
    with torch.cuda.device(G1):
        if args.no_out:
            return [dec.decode(nvcomp.as_array(x)) for x in cdsts]
        return [dec.decode(nvcomp.as_array(x), out=nvcomp.as_array(y)) for x, y in zip(cdsts, dsegs)]


def run_config(a, c, ref, dst, ssegs, dsegs):
    """在一个分片上测一种（算法, 分块）组合。
    编解码器在这里创建、函数结束即销毁：每个 Codec 会保留自己的临时缓冲区，
    如果像旧版那样一次性创建全部 Codec 并一直持有，显存会随测试的组合逐步累积"""
    r = {"td": {}, "te2e": {}}
    cur["stage"] = "创建编解码器"
    enc = nvcomp.Codec(algorithm=a, uncomp_chunk_size=c, device_id=args.src)
    decs = {b: nvcomp.Codec(algorithm=a, uncomp_chunk_size=c, device_id=args.dst,
                            decompress_backend=BK[b]) for b in backends}

    cur["stage"] = "B1 压缩"
    r["tc"], comps = bench(on(G0, lambda: [enc.encode(nvcomp.as_array(x)) for x in ssegs]))
    # 实测 encode 返回数组的长度是预分配容量，真实压缩大小是 buffer_size，只截取有效部分去统计和传输
    csrcs = [as_u8(x)[:x.buffer_size] for x in comps]
    cdsts = [torch.empty_like(x, device=G1) for x in csrcs]
    r["cbytes"] = sum(x.numel() for x in csrcs)

    cur["stage"] = "B2 传输"
    r["tx"] = bench(lambda: [y.copy_(x, non_blocking=True) for x, y in zip(csrcs, cdsts)])[0]

    for b, dec in decs.items():
        if args.no_out:
            decode = on(G1, lambda: [dec.decode(nvcomp.as_array(x)) for x in cdsts])
        else:
            decode = on(G1, lambda: [dec.decode(nvcomp.as_array(x), out=nvcomp.as_array(y))
                                     for x, y in zip(cdsts, dsegs)])
        cur["stage"] = f"B3 解压[{b}]"
        r["td"][b] = bench(decode)[0]
        cur["stage"] = f"校验[{b}]"
        verify(decode, ref, dst)
    del comps, csrcs, cdsts                      # 端到端测试会自己重新压缩，先释放这一份

    if not args.no_e2e:
        for b, dec in decs.items():
            cur["stage"] = f"B 端到端[{b}]"
            r["te2e"][b] = bench(lambda: b_path(enc, dec, ssegs, dsegs))[0]
    return r


shards = sorted(glob.glob(os.path.join(args.model_dir, "*.safetensors")))
if not shards:
    raise SystemExit(f"{args.model_dir} 下没有 .safetensors 文件")
if args.max_shards:
    shards = shards[::max(1, len(shards) // args.max_shards)][:args.max_shards]

raw_total, tA_total, acc = 0, 0.0, {}
skipped = set()   # 因显存不足跳过的（算法, 分块）组合，汇总时不统计
for i, f in enumerate(shards):
    ts = load_file(f)                                                    # 读盘不计时
    src = torch.cat([t.reshape(-1).view(torch.uint8) for t in ts.values()]).to(G0)
    del ts
    ref = src.to(G1)                                                     # 校验用的参考数据
    dst = torch.empty_like(ref)
    tA = bench(lambda: dst.copy_(src, non_blocking=True))[0]             # 方案 A：NVLink 直传
    raw_total += src.numel(); tA_total += tA
    print(f"[{i+1}/{len(shards)}] {os.path.basename(f)} {src.numel()/1e9:.2f} GB  A={tA*1e3:.2f} ms  | {mem()}")

    seg = args.seg_mb << 20                                              # 每段字节数
    offs = range(0, src.numel(), seg)
    ssegs = [src[o:o + seg] for o in offs]                               # 发送端原始数据分段（视图，不拷贝）
    dsegs = [dst[o:o + seg] for o in offs]                               # 接收端输出分段（视图，不拷贝）

    for a, c in itertools.product(algos, chunks):
        if (a, c) in skipped:
            continue
        oom = None
        try:
            r = run_config(a, c, ref, dst, ssegs, dsegs)
        except RuntimeError as ex:                     # torch.OutOfMemoryError 也是 RuntimeError 的子类
            if "out of memory" not in str(ex).lower():
                raise                                  # 不是显存不足的错误照常报出来
            oom = str(ex).splitlines()[0][:150]
        gc.collect(); torch.cuda.empty_cache()         # 每个组合结束都回收：销毁的 Codec + PyTorch 缓存
        if oom:
            skipped.add((a, c))
            print(f"    {a:7s} chunk={c:>7d} 在「{cur['stage']}」显存不足，跳过这个组合 | 回收后 {mem()}\n"
                  f"        原始错误: {oom}")
            continue
        s = acc.setdefault((a, c), [0, 0.0, 0.0]); s[0] += r["cbytes"]; s[1] += r["tc"]; s[2] += r["tx"]
        line = (f"    {a:7s} chunk={c:>7d}\tratio={src.numel()/r['cbytes']:.3f}\t"
                f"c={r['tc']*1e3:.1f}\tx={r['tx']*1e3:.2f}")
        for b in backends:
            t = acc.setdefault((a, c, b), [0.0, 0.0])
            t[0] += r["td"][b]
            line += f"\td[{b}]={r['td'][b]*1e3:.1f}"
            if not args.no_e2e:
                t[1] += r["te2e"][b]
                line += f" e2e[{b}]={r['te2e'][b]*1e3:.1f}"
        print(line + f" ms | 回收后 {mem()}")
    del src, ref, dst, ssegs, dsegs
    gc.collect(); torch.cuda.empty_cache()

GB = raw_total / 1e9
print(f"\n总量 {GB:.1f} GB | A 直传 {tA_total:.3f}s ({GB/tA_total:.0f} GB/s)")
new = not os.path.exists(args.csv)
with open(args.csv, "a", newline="") as fp:
    w = csv.writer(fp)
    if new:
        w.writerow(["algo", "chunk", "backend", "raw_GB", "comp_GB", "ratio", "A_s",
                    "comp_s", "xfer_s", "decomp_s", "B_sum_est_s", "B_pipe_ideal_s", "B_e2e_s",
                    "speedup_sum_est", "speedup_pipe_ideal", "speedup_e2e",
                    "comp_GBps", "xfer_GBps", "decomp_GBps"])
    for a, c in itertools.product(algos, chunks):
        if (a, c) in skipped:
            print(f"\n{a} chunk={c}: 显存不足，已跳过，不计入汇总")
            continue
        cb, tc, tx = acc[(a, c)]
        CGB = cb / 1e9                                   # 实际传输的压缩数据量
        for b in backends:
            td, te2e = acc[(a, c, b)]
            est, pipe = tc + tx + td, max(tc, tx, td)
            e2e_txt = "未测" if args.no_e2e else f"{te2e:.3f}s"
            sp_e2e = "" if args.no_e2e else f"{tA_total/te2e:.3f}"
            print(f"\n{a} chunk={c} backend={b}  压缩率={raw_total/cb:.3f}（{GB:.1f} GB -> {CGB:.1f} GB）\n"
                  f"    B1 SM压缩  {tc:.3f}s  ({GB/tc:.0f} GB/s，按原始数据量)\n"
                  f"    B2 NVLink  {tx:.3f}s  ({CGB/tx:.0f} GB/s，按压缩后数据量)\n"
                  f"    B3 解压    {td:.3f}s  ({GB/td:.0f} GB/s，按原始数据量)\n"
                  f"    B 三段相加(估算) {est:.3f}s | B 理想流水下限(估算) {pipe:.3f}s | B 端到端实测 {e2e_txt}\n"
                  f"    A 直传 {tA_total:.3f}s | 加速比=A/B：三段相加 {tA_total/est:.2f}，理想流水 {tA_total/pipe:.2f}"
                  + ("" if args.no_e2e else f"，端到端实测 {tA_total/te2e:.2f}"))
            w.writerow([a, c, b, f"{GB:.3f}", f"{CGB:.3f}", f"{raw_total/cb:.4f}", f"{tA_total:.4f}",
                        f"{tc:.4f}", f"{tx:.4f}", f"{td:.4f}", f"{est:.4f}", f"{pipe:.4f}",
                        "" if args.no_e2e else f"{te2e:.4f}",
                        f"{tA_total/est:.3f}", f"{tA_total/pipe:.3f}", sp_e2e,
                        f"{GB/tc:.1f}", f"{CGB/tx:.1f}", f"{GB/td:.1f}"])
