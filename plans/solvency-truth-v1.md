# solvency-truth-v1 — Bank-grade baseline matrix
Generated: 2026-06-12T22:49:21

## Setup
- Box: AMD EPYC 7282 Zen 2, 16C/32T, single NUMA
- Binaries: gzippy-isal (parallel-sm+isal), gzippy-native (parallel-sm+pure), rapidgzip 0.16.0
- Freeze: governor=performance, boost=0 (AMD)
- Ratio convention: rg_min/gz_min >= 0.99 = PASS
- N=7 interleaved (T1, T8); N=5 interleaved (T16)
- Sinks: /dev/shm tmpfs (32GiB)
- SHA-verified every run (output == raw pin)

## Routing assertions
- isal T8 silesia → ParallelSM: `True`
- isal T1 silesia → ISA-L path: `True`

## T1 results
| corpus | gz_sz | isal_s | native_s | rg_s | isal/rg | native/rg | isal_v | nat_v | isal_cpu | nat_cpu | rg_cpu | isal_RSS | nat_RSS | rg_RSS |
|--------|-------|--------|----------|------|---------|-----------|--------|-------|----------|---------|--------|----------|---------|--------|
| silesia                 | 64.5M | 0.504  | 0.711    | 0.598 | 1.187   | 0.841     | PASS   | LOSS  | 0.49s    | 0.70s   | 0.59s  | 69M      | 163M    | 56M    |
| access.log (*)          | 2.4M  | 0.040  | 0.073    | 0.062 | 1.536   | 0.853     | PASS   | LOSS  | 0.03s    | 0.06s   | 0.05s  | 7M       | 37M     | 28M    |
| aozora.txt (*)          | 3.8M  | 0.033  | 0.047    | 0.044 | 1.338   | 0.944     | PASS   | LOSS  | 0.02s    | 0.04s   | 0.04s  | 8M       | 20M     | 17M    |
| armexe.elf (*)          | 567K  | 0.009  | 0.011    | 0.011 | 1.155   | 0.969     | PASS   | LOSS  | 0.00s    | 0.00s   | 0.00s  | 5M       | 6M      | 7M     |
| data.csv (*)            | 3.2M  | 0.041  | 0.069    | 0.066 | 1.585   | 0.948     | PASS   | LOSS  | 0.03s    | 0.06s   | 0.05s  | 7M       | 33M     | 31M    |
| data.json (*)           | 1.4M  | 0.026  | 0.040    | 0.039 | 1.499   | 0.964     | PASS   | LOSS  | 0.02s    | 0.03s   | 0.03s  | 6M       | 20M     | 19M    |
| data.parquet            | 13.5M | 0.088  | 0.118    | 0.102 | 1.159   | 0.860     | PASS   | LOSS  | 0.08s    | 0.11s   | 0.09s  | 18M      | 38M     | 21M    |
| data.sqlite             | 12.3M | 0.117  | 0.207    | 0.154 | 1.319   | 0.743     | PASS   | LOSS  | 0.11s    | 0.19s   | 0.14s  | 16M      | 78M     | 38M    |
| dickens                 | 4.3M  | 0.035  | 0.051    | 0.046 | 1.310   | 0.906     | PASS   | LOSS  | 0.03s    | 0.04s   | 0.03s  | 9M       | 21M     | 17M    |
| ecoli.fastq             | 4.2M  | 0.044  | 0.074    | 0.065 | 1.481   | 0.888     | PASS   | LOSS  | 0.04s    | 0.07s   | 0.05s  | 9M       | 34M     | 29M    |
| engine.wasm (*)         | 392K  | 0.009  | 0.009    | 0.009 | 1.041   | 0.961     | PASS   | LOSS  | 0.00s    | 0.00s   | 0.00s  | 5M       | 6M      | 6M     |
| markup.xml (*)          | 2.0M  | 0.021  | 0.031    | 0.030 | 1.391   | 0.965     | PASS   | LOSS  | 0.01s    | 0.02s   | 0.02s  | 6M       | 14M     | 14M    |
| minjs.min.js (*)        | 1.0M  | 0.014  | 0.019    | 0.018 | 1.264   | 0.969     | PASS   | LOSS  | 0.00s    | 0.01s   | 0.00s  | 5M       | 9M      | 9M     |
| monorepo.tar            | 9.4M  | 0.096  | 0.158    | 0.139 | 1.444   | 0.881     | PASS   | LOSS  | 0.08s    | 0.15s   | 0.13s  | 14M      | 72M     | 49M    |
| movie.mp4               | 12.3M | 0.055  | 0.064    | 0.064 | 1.172   | 0.999     | PASS   | PASS  | 0.04s    | 0.06s   | 0.05s  | 16M      | 29M     | 18M    |
| photo.jpg               | 6.2M  | 0.036  | 0.041    | 0.041 | 1.137   | 0.999     | PASS   | PASS  | 0.02s    | 0.03s   | 0.03s  | 10M      | 17M     | 11M    |
| symbols.dwarf (*)       | 362K  | 0.008  | 0.009    | 0.009 | 1.146   | 0.971     | PASS   | LOSS  | 0.00s    | 0.00s   | 0.00s  | 4M       | 6M      | 6M     |
| tool.bin                | 19.9M | 0.174  | 0.256    | 0.214 | 1.232   | 0.837     | PASS   | LOSS  | 0.16s    | 0.24s   | 0.20s  | 24M      | 74M     | 41M    |
| weights.safetensors     | 79.3M | 0.385  | 0.563    | 0.416 | 1.080   | 0.739     | PASS   | LOSS  | 0.37s    | 0.56s   | 0.41s  | 84M      | 103M    | 19M    |
| winexe.exe (*)          | 1.4M  | 0.018  | 0.023    | 0.021 | 1.148   | 0.910     | PASS   | LOSS  | 0.01s    | 0.01s   | 0.01s  | 6M       | 10M     | 9M     |

T1 isal PASS: 20, LOSS: 0
(*) < 4MiB compressed — single-chunk path at T>1, startup-dominated at T1

## T8 results
| corpus | gz_sz | isal_s | native_s | rg_s | isal/rg | native/rg | isal_v | nat_v | isal_cpu | nat_cpu | rg_cpu | isal_RSS | nat_RSS | rg_RSS |
|--------|-------|--------|----------|------|---------|-----------|--------|-------|----------|---------|--------|----------|---------|--------|
| silesia                 | 64.5M | 0.330  | 0.299    | 0.238 | 0.720   | 0.796     | LOSS   | LOSS  | 1.56s    | 1.45s   | 1.04s  | 320M     | 296M    | 228M   |
| access.log (*)          | 2.4M  | 0.065  | 0.063    | 0.057 | 0.871   | 0.891     | LOSS   | LOSS  | 0.16s    | 0.16s   | 0.12s  | 57M      | 54M     | 53M    |
| aozora.txt (*)          | 3.8M  | 0.036  | 0.035    | 0.034 | 0.941   | 0.968     | LOSS   | LOSS  | 0.12s    | 0.12s   | 0.10s  | 34M      | 33M     | 31M    |
| armexe.elf (*)          | 567K  | 0.010  | 0.011    | 0.011 | 1.032   | 0.983     | PASS   | LOSS  | 0.00s    | 0.00s   | 0.00s  | 7M       | 7M      | 8M     |
| data.csv (*)            | 3.2M  | 0.064  | 0.063    | 0.052 | 0.816   | 0.831     | LOSS   | LOSS  | 0.20s    | 0.18s   | 0.14s  | 60M      | 58M     | 56M    |
| data.json (*)           | 1.4M  | 0.048  | 0.046    | 0.038 | 0.783   | 0.808     | LOSS   | LOSS  | 0.08s    | 0.07s   | 0.06s  | 32M      | 30M     | 30M    |
| data.parquet            | 13.5M | 0.062  | 0.059    | 0.049 | 0.790   | 0.828     | LOSS   | LOSS  | 0.27s    | 0.26s   | 0.21s  | 60M      | 59M     | 43M    |
| data.sqlite             | 12.3M | 0.112  | 0.108    | 0.085 | 0.757   | 0.784     | LOSS   | LOSS  | 0.44s    | 0.41s   | 0.33s  | 102M     | 101M    | 84M    |
| dickens                 | 4.3M  | 0.037  | 0.035    | 0.034 | 0.920   | 0.960     | LOSS   | LOSS  | 0.14s    | 0.13s   | 0.12s  | 36M      | 34M     | 31M    |
| ecoli.fastq             | 4.2M  | 0.058  | 0.056    | 0.054 | 0.922   | 0.951     | LOSS   | LOSS  | 0.21s    | 0.21s   | 0.17s  | 62M      | 60M     | 57M    |
| engine.wasm (*)         | 392K  | 0.009  | 0.009    | 0.009 | 1.088   | 0.999     | PASS   | PASS  | 0.00s    | 0.00s   | 0.00s  | 6M       | 6M      | 7M     |
| markup.xml (*)          | 2.0M  | 0.079  | 0.072    | 0.030 | 0.373   | 0.409     | LOSS   | LOSS  | 0.14s    | 0.12s   | 0.06s  | 23M      | 21M     | 21M    |
| minjs.min.js (*)        | 1.0M  | 0.027  | 0.026    | 0.023 | 0.843   | 0.887     | LOSS   | LOSS  | 0.03s    | 0.03s   | 0.02s  | 13M      | 12M     | 12M    |
| monorepo.tar            | 9.4M  | 0.108  | 0.097    | 0.079 | 0.731   | 0.811     | LOSS   | LOSS  | 0.47s    | 0.44s   | 0.35s  | 113M     | 108M    | 93M    |
| movie.mp4               | 12.3M | 0.033  | 0.030    | 0.026 | 0.779   | 0.863     | LOSS   | LOSS  | 0.11s    | 0.09s   | 0.08s  | 39M      | 34M     | 25M    |
| photo.jpg               | 6.2M  | 0.025  | 0.021    | 0.020 | 0.790   | 0.941     | LOSS   | LOSS  | 0.08s    | 0.07s   | 0.06s  | 26M      | 23M     | 19M    |
| symbols.dwarf (*)       | 362K  | 0.009  | 0.010    | 0.010 | 1.057   | 1.010     | PASS   | PASS  | 0.00s    | 0.00s   | 0.00s  | 6M       | 6M      | 7M     |
| tool.bin                | 19.9M | 0.119  | 0.110    | 0.092 | 0.774   | 0.832     | LOSS   | LOSS  | 0.55s    | 0.52s   | 0.41s  | 122M     | 117M    | 90M    |
| weights.safetensors     | 79.3M | 0.134  | 0.148    | 0.109 | 0.815   | 0.739     | LOSS   | LOSS  | 0.69s    | 0.78s   | 0.49s  | 154M     | 148M    | 78M    |
| winexe.exe (*)          | 1.4M  | 0.024  | 0.023    | 0.021 | 0.896   | 0.925     | LOSS   | LOSS  | 0.03s    | 0.03s   | 0.03s  | 14M      | 13M     | 13M    |

T8 isal PASS: 3, LOSS: 17
(*) < 4MiB compressed — single-chunk path at T>1, startup-dominated at T1

## T16 results
| corpus | gz_sz | isal_s | native_s | rg_s | isal/rg | native/rg | isal_v | nat_v | isal_cpu | nat_cpu | rg_cpu | isal_RSS | nat_RSS | rg_RSS |
|--------|-------|--------|----------|------|---------|-----------|--------|-------|----------|---------|--------|----------|---------|--------|
| silesia                 | 64.5M | 0.298  | 0.290    | 0.268 | 0.898   | 0.924     | LOSS   | LOSS  | 2.15s    | 2.10s   | 1.68s  | 393M     | 389M    | 327M   |
| access.log (*)          | 2.4M  | 0.065  | 0.063    | 0.055 | 0.854   | 0.874     | LOSS   | LOSS  | 0.16s    | 0.15s   | 0.12s  | 57M      | 55M     | 53M    |
| aozora.txt (*)          | 3.8M  | 0.035  | 0.035    | 0.033 | 0.944   | 0.948     | LOSS   | LOSS  | 0.12s    | 0.12s   | 0.11s  | 34M      | 34M     | 30M    |
| armexe.elf (*)          | 567K  | 0.011  | 0.011    | 0.011 | 1.019   | 0.972     | PASS   | LOSS  | 0.00s    | 0.00s   | 0.00s  | 7M       | 7M      | 8M     |
| data.csv (*)            | 3.2M  | 0.064  | 0.063    | 0.052 | 0.816   | 0.831     | LOSS   | LOSS  | 0.19s    | 0.19s   | 0.15s  | 60M      | 58M     | 55M    |
| data.json (*)           | 1.4M  | 0.049  | 0.047    | 0.038 | 0.773   | 0.807     | LOSS   | LOSS  | 0.07s    | 0.07s   | 0.05s  | 32M      | 30M     | 30M    |
| data.parquet            | 13.5M | 0.048  | 0.048    | 0.044 | 0.932   | 0.918     | LOSS   | LOSS  | 0.31s    | 0.30s   | 0.25s  | 62M      | 64M     | 49M    |
| data.sqlite             | 12.3M | 0.082  | 0.083    | 0.079 | 0.966   | 0.959     | LOSS   | LOSS  | 0.49s    | 0.48s   | 0.42s  | 117M     | 110M    | 96M    |
| dickens                 | 4.3M  | 0.038  | 0.036    | 0.035 | 0.920   | 0.969     | LOSS   | LOSS  | 0.13s    | 0.13s   | 0.12s  | 35M      | 35M     | 32M    |
| ecoli.fastq             | 4.2M  | 0.057  | 0.057    | 0.053 | 0.921   | 0.933     | LOSS   | LOSS  | 0.20s    | 0.20s   | 0.17s  | 62M      | 60M     | 56M    |
| engine.wasm (*)         | 392K  | 0.009  | 0.009    | 0.009 | 1.066   | 0.984     | PASS   | LOSS  | 0.00s    | 0.00s   | 0.00s  | 6M       | 6M      | 7M     |
| markup.xml (*)          | 2.0M  | 0.080  | 0.072    | 0.030 | 0.378   | 0.419     | LOSS   | LOSS  | 0.14s    | 0.12s   | 0.06s  | 23M      | 22M     | 21M    |
| minjs.min.js (*)        | 1.0M  | 0.027  | 0.026    | 0.023 | 0.860   | 0.890     | LOSS   | LOSS  | 0.03s    | 0.02s   | 0.02s  | 13M      | 12M     | 12M    |
| monorepo.tar            | 9.4M  | 0.110  | 0.101    | 0.087 | 0.792   | 0.859     | LOSS   | LOSS  | 0.50s    | 0.54s   | 0.45s  | 123M     | 117M    | 109M   |
| movie.mp4               | 12.3M | 0.032  | 0.030    | 0.029 | 0.908   | 0.969     | LOSS   | LOSS  | 0.16s    | 0.12s   | 0.11s  | 47M      | 41M     | 33M    |
| photo.jpg               | 6.2M  | 0.023  | 0.020    | 0.020 | 0.900   | 1.001     | LOSS   | PASS  | 0.09s    | 0.07s   | 0.07s  | 30M      | 25M     | 24M    |
| symbols.dwarf (*)       | 362K  | 0.009  | 0.010    | 0.010 | 1.044   | 0.986     | PASS   | LOSS  | 0.00s    | 0.00s   | 0.00s  | 6M       | 6M      | 7M     |
| tool.bin                | 19.9M | 0.102  | 0.097    | 0.092 | 0.906   | 0.949     | LOSS   | LOSS  | 0.66s    | 0.60s   | 0.54s  | 135M     | 131M    | 109M   |
| weights.safetensors     | 79.3M | 0.124  | 0.123    | 0.109 | 0.882   | 0.886     | LOSS   | LOSS  | 0.78s    | 0.85s   | 0.55s  | 179M     | 170M    | 99M    |
| winexe.exe (*)          | 1.4M  | 0.024  | 0.024    | 0.022 | 0.908   | 0.913     | LOSS   | LOSS  | 0.03s    | 0.03s   | 0.03s  | 14M      | 13M     | 13M    |

T16 isal PASS: 3, LOSS: 17
(*) < 4MiB compressed — single-chunk path at T>1, startup-dominated at T1

## perf stat — silesia T8 (retired instructions)

### isal
- cycles:       4,348,176,712
- instructions: 7,708,998,707
- IPC:          1.773

### native
- cycles:       4,035,346,066
- instructions: 7,610,885,936
- IPC:          1.886

### rg
- cycles:       2,853,822,685
- instructions: 4,897,241,338
- IPC:          1.716

isal/rg instruction ratio:   1.574
native/rg instruction ratio: 1.554

## Manifest (all pins)
```
# GZ sha256 pins
7a34adc06068af0af7bc19ab0f1280fb76f241fed6cbee6aa680d35dd6af5784  silesia.gz
505a2a7a04a1b648fe4e7f4ec7348154e0bc60c5080ae9049deedae82b94b27c  access.log.gz
7d5031f3a10759c680e424e94482b4e6b14750302ec7e8a408ca2997653088a0  aozora.txt.gz
31d5cc206e4a0fdd0c73af95c385887c38f58a0e9fe450274758c0f04d2ecb2b  armexe.elf.gz
cf825f06f11ede6f5ebca6ca1c385751d7c73f1872aec3e7efa66555f3845460  data.csv.gz
a3f8ae4aabd631a18736606183823d5bccb2b25d023a67eb1946a90a5d981364  data.json.gz
e13daa9660b488c1e4e53ce83f7fa5a3b3c4433a0e87602692657bfc5c5f9880  data.parquet.gz
2981eea6fb096613ac2ca6c82d4d75f9b8484080d561a53f76bdb4b91b115c80  data.sqlite.gz
6ef9057e5d63b6c2443f2439d2c0032691c87fe91807775b3b24c5431f39ffce  dickens.gz
6d3bc784d4f0861dfb5c9a38571d8a8defa4efb4c1c13f5f60f4ff5851709c21  ecoli.fastq.gz
4b588fda6a7a321059dc03378456ec16335f8eef6809bec09c7e46b4c70dd3f8  engine.wasm.gz
26c15de8e472720178c44c1dd76b8c39b88fb71f53fd8dd2fef8630782ad5614  markup.xml.gz
40be0536ab8c06262963e1e0116fe588ebae41adf7dfd2ad3ef246e29607e9e5  minjs.min.js.gz
217e7c0d24444c957d9832d00a3a5da051c35482d6764ee979f04f6d4740b8bd  monorepo.tar.gz
cb63cd70aec4ced2808f99d631bcb34c9db141a618650d76bddb569524be4adf  movie.mp4.gz
136b6bb68828528f0da33981436c1b0e84e6a48d6652cb3b0def6435f4514751  photo.jpg.gz
bcc0c9bcb149a983d8105e57e20415e32b32d233aa758a1e7afa0904fa15bcc3  symbols.dwarf.gz
a17e724de1a42bb4a39ead91e852c6c08089e66b9a39d3aa08be8d71f7ceb9fd  tool.bin.gz
ec8d6d112d1901643fa6b38ba9bfa2ce20cf4c2dc8378b24ebd3779d4297ca88  weights.safetensors.gz
e8e33e31ad882ebe61454b8658003e1894e25c0d363a2a446225720589fb70ac  winexe.exe.gz

# RAW sha256 pins
028bd002c89c9a909ccdbc2af0a223de285348edb014ccc8e27d297f52cb410f  silesia (raw)
82994588b2ee67f6ad032aa90fa1ad04906f5671755b1462230436dc50c915df  access.log (raw)
c12106e76b22b238f43bcc62b26d6426192425e5f819d7406e2c2af7f8ef0237  aozora.txt (raw)
36b694487054adb4bd239fcff2d7cff6fdc96105bfcc6715dc5e7a60f8a21138  armexe.elf (raw)
17820a52b15fc5e81047ccef24c868f514b7ea75e61eba17bd2d66d5c2e1e913  data.csv (raw)
0ad6eb1190815f4eff7fc07420c50b424ce9aca3312202816c907894cb32cdae  data.json (raw)
9bb667a7fdcf9d3e6211368fd79c6628990c5d128ed1a30e285b1d55f99b6601  data.parquet (raw)
21a73d3d7af4f82d688054348eea72965bdc4128f23524b5c60698e1b67c2634  data.sqlite (raw)
7312e776a3d7e5e25b7415116d3777c934f3c9c47fc153356ca16608d6428bc0  dickens (raw)
96493a50bfbd57769b4f35764741605b70d96be01680052ae898850d23c190c0  ecoli.fastq (raw)
ae1cd941deaa3e6a4880e6f1287a5354cfab9e8dfbbb389158e94026f364da49  engine.wasm (raw)
e8861c1cc6bc0fddcc4bfc2a7d6dae265bcd400e8782be9371e6f4ca30731aec  markup.xml (raw)
7f4930eba8f8541dbec28dca5bd5f787f8eef1cde0369ac9657b70bed230b3e0  minjs.min.js (raw)
0dd50d07b0147211144cf696ed8c16418b8b34c0c064df82c7ac352cc640f509  monorepo.tar (raw)
66a42e9789020213c4b3ebd86a471aeba4fc0d31fde9cea4fc09bd3c65b67619  movie.mp4 (raw)
51049ed4e34514880bfbe39d34e2dd2c56e31f55919e7f4f0b34ef1698cb9f50  photo.jpg (raw)
9f3b57ebf4c2e5ad963c38e32a4fffa66fb7f4ab364901ad33ecc907323e4a08  symbols.dwarf (raw)
c50474410d63d60e6addd259d4310eac4bc67ed16248c69017c68cc0280ed5f9  tool.bin (raw)
53aa51172d142c89d9012cce15ae4d6cc0ca6895895114379cacb4fab128d9db  weights.safetensors (raw)
4c9d082ee20f0d9e44881ac4e92adf765efc314d82103c53d7f576bd78dc5761  winexe.exe (raw)
```
