#!/usr/bin/env python3
# 检查 nvCOMP 压缩输出的真实大小，并验证"只保留前 buffer_size 字节"仍能被 DE 正确解压。
# 同时给出真实权重的压缩率，以及 CPU zlib 的参考压缩率。
#
# 用法：
#   export MODEL=/data/models/xxx
#   python check_comp_size.py                 # 默认：第 2 个分片的前 256MB，GPU0
#   python check_comp_size.py --mb 512 --shard-index 10 --gpu 1
import argparse, glob, os, time, zlib
import torch
from safetensors.torch import load_file
from nvidia import nvcomp

ap = argparse.ArgumentParser()
ap.add_argument("--model-dir", default=os.environ.get("MODEL"), help="模型目录，默认读环境变量 $MODEL")
ap.add_argument("--shard-index", type=int, default=1, help="用第几个分片（从 0 开始）")
ap.add_argument("--mb", type=int, default=256, help="取多少 MB 真实权重来测")
ap.add_argument("--gpu", type=int, default=0)
ap.add_argument("--no-zlib", action="store_true", help="跳过 CPU zlib 参考（level 6 较慢）")
args = ap.parse_args()
assert args.model_dir, "没有模型目录：先 export MODEL=模型目录，或者用 --model-dir 指定"

G = torch.device(f"cuda:{args.gpu}")
torch.cuda.set_device(G)          # nvcomp.as_array 按"当前 GPU"认定数据位置，必须和数据所在卡一致


def find_backend(key):            # 和 bench_de_xfer.py 里相同：自动查找解压后端枚举
    for name in dir(nvcomp):
        members = getattr(getattr(nvcomp, name), "__members__", None)
        if members and "backend" in name.lower():
            for k, v in members.items():
                if key in k.upper():
                    return v
    raise RuntimeError(f"找不到 {key} 后端枚举，用 help(nvcomp.Codec) 查看后手动填写")


HW = find_backend("HARDWARE")

shards = sorted(glob.glob(os.path.join(args.model_dir, "*.safetensors")))
assert shards, f"{args.model_dir} 下没有 .safetensors 文件"
f = shards[min(args.shard_index, len(shards) - 1)]
ts = load_file(f)
print("nvcomp", nvcomp.__version__, "| 分片:", os.path.basename(f),
      "| 数据类型:", sorted({str(t.dtype) for t in ts.values()}))
n = args.mb << 20
real = torch.cat([t.reshape(-1).view(torch.uint8) for t in ts.values()])[:n].to(G)
del ts
zeros = torch.zeros(real.numel(), dtype=torch.uint8, device=G)


def check(label, data, algo, kw):
    try:
        comp = nvcomp.Codec(algorithm=algo, device_id=args.gpu, **kw).encode(nvcomp.as_array(data))
        torch.cuda.synchronize(G)
        t = torch.from_dlpack(comp).view(torch.uint8).reshape(-1)
        size = comp.buffer_size
        # 只保留前 buffer_size 字节并复制到新缓冲区，再用 DE 解压：能正确还原，说明截取方式是对的
        packed = t[:size].clone()
        out = torch.empty_like(data)
        dec = nvcomp.Codec(algorithm=algo, device_id=args.gpu, decompress_backend=HW)
        dec.decode(nvcomp.as_array(packed), out=nvcomp.as_array(out))
        torch.cuda.synchronize(G)
        ok = "正确" if torch.equal(out, data) else "错误！"
        print(f"{label:4s} {algo:7s} {str(kw):22s} 数组字节={t.numel():>11d}  buffer_size={size:>11d}  "
              f"capacity={getattr(comp, 'capacity', 'NA')}  真实压缩率={data.numel()/size:.4f}  截取后DE解压{ok}")
    except Exception as e:
        print(f"{label:4s} {algo:7s} {str(kw):22s} 出错: {e}")


check("全零", zeros, "LZ4", {})                     # 对照：全零数据的真实压缩率应该非常高
for algo, kw in [("LZ4", {}), ("Snappy", {}),
                 ("Deflate", {"algorithm_type": 0}), ("Deflate", {"algorithm_type": 1}),
                 ("Deflate", {"algorithm_type": 2}), ("Gzip", {})]:
    check("权重", real, algo, kw)

if not args.no_zlib:                                # CPU zlib 作为"这份数据大概能压到多少"的参考
    b = real.cpu().numpy().tobytes()
    for lv in (1, 6):
        t0 = time.perf_counter()
        r = len(b) / len(zlib.compress(b, lv))
        print(f"CPU zlib level {lv}: 压缩率={r:.4f}（耗时 {time.perf_counter()-t0:.1f}s）")
