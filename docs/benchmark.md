# GCMagicc v1 computational benchmark

Measured on the host `gus` (AMD EPYC 9555 64-Core Processor, 1 socket, 64 physical cores / 128
threads, 4 NUMA nodes, 1.5--4.41 GHz, 1511 GiB RAM, no GPU). All runs are CPU-only:
`CUDA_VISIBLE_DEVICES=''`.

## Headline result

One 100-year (1200-month), monthly, 1x1-degree, 10-variable GCMagicc rollout -- that is, one ensemble
member -- takes **90--96 s of wall-clock time on 8 pinned cores**, with a peak resident memory of
140 GB when generated in a single call.

Thread pinning for the headline figure:

```sh
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
       NUMEXPR_NUM_THREADS=8 VECLIB_MAXIMUM_THREADS=8
export OMP_PROC_BIND=close OMP_PLACES=cores CUDA_VISIBLE_DEVICES=''
taskset -c 0-7 python -c "...run.sample_from_combined_model(x, device='cpu', nside=64, rectangular=True, nlat=180)"
```

with `torch.set_num_threads(8)` and `torch.set_num_interop_threads(1)`.

## What the number covers

End-to-end CPU-only generation of one complete ensemble member: predictor-tensor assembly from a
released-format MAGICC predictor HDF5, loading the full nside 1-to-64 checkpoint cascade (13
`torch.load` calls, 5.2 GB), the multiresolution generative rollout itself, and the
HEALPix-to-latitude/longitude regrid to a 180x360 mesh.

## What it excludes

- Training cost.
- The upstream MAGICC run that produces the predictors, and the ERA5 splicing of workflow 616 -- a
  pre-existing predictor file was used.
- Writing output to disk: no NetCDF or Zarr encoding, no I/O, no 360-day calendar regrid. The
  in-memory result array is 6.2 GB in float64.
- Multi-member ensembles: this is the per-member cost.

## Notes that matter for reproduction

- **Memory scales with rollout length, not with checkpoint size.** Peak RSS is 140 GB for the
  production configuration (`dependence=True`, which internally prepends 144 warm-up months, giving
  1344 steps) and 125 GB with `dependence=False`. Emitting the rollout in ten-year chunks reduces
  peak RSS to 23 GB for about 20 % more wall-clock time, and is the recommended configuration on
  memory-limited hardware.
- **Roughly 40 % of the time is the regrid.** A HEALPix-native run (no interpolation to
  latitude/longitude) completes in 47.3 s.
- **Per-member throughput saturates by about 16 cores.** Ensembles should be produced by running
  members concurrently rather than by allocating more cores to a single member.
- **Host contention dominates the spread.** Across four replicates: 90.26, 96.02, 117.73 and
  181.68 s. The slow runs achieved only 3.2 of 8 pinned cores on a machine carrying a load average
  of 22--48. The two warm-cache replicates (90.26 and 96.02 s, median 93.14 s, spread ~6 %) are the
  representative figures.
- All 37 checkpoint files of the on-host snapshot were verified byte-for-byte against
  `data/checkpoint_manifest.json` before benchmarking.

## Per-run measurements

```
run_id,what_was_run,wall_clock_s,rollout_call_s,cores,mean_cores_used,cpu_model,peak_rss_gb,n_members,months,n_variables,output_shape,dependence,chunk_months,ckpt_load_s,regrid_s,net_compute_s,notes
d8_b,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=True (production setting), lat/lon output, 8 cores — replicate 2",90.26,87.237,8,4.51,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",139.72,1,1200,10,1200x10x180x360,True,0,0.44,37.703,49.094,warm page cache; PRODUCTION-CONFIG HEADLINE
d8_d,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=True (production setting), lat/lon output, 8 cores — replicate 4",96.02,93.139,8,4.45,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",139.72,1,1200,10,1200x10x180x360,True,0,0.437,38.37,54.332,warm page cache; PRODUCTION-CONFIG HEADLINE
d8_a,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=True, lat/lon output, 8 cores — replicate 1",117.73,114.798,8,3.9,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",139.72,1,1200,10,1200x10x180x360,True,0,0.44,37.797,76.561,"first dependence run, partially cold page cache"
d8_c,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=True, lat/lon output, 8 cores — replicate 3",181.68,178.347,8,3.18,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",139.72,1,1200,10,1200x10x180x360,True,0,1.049,37.255,140.042,OUTLIER: only 318% mean CPU of 8 pinned cores; host shared with other users' jobs (load avg 22-48) during this run
d16_a,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=True, lat/lon output, 16 cores",84.67,81.559,16,7.8,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",139.72,1,1200,10,1200x10x180x360,True,0,0.468,37.624,43.466,"core-count scaling, production setting"
r8_b,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=False, lat/lon output, 8 cores — replicate 2",84.71,81.714,8,4.24,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",125.37,1,1200,10,1200x10x180x360,False,0,0.452,37.353,43.909,warm page cache
r8_c,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=False, lat/lon output, 8 cores — replicate 3",82.9,80.042,8,4.25,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",125.41,1,1200,10,1200x10x180x360,False,0,0.464,36.858,42.721,warm page cache
r8_a,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=False, lat/lon output, 8 cores — replicate 1",136.81,133.931,8,2.93,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",125.38,1,1200,10,1200x10x180x360,False,0,0.432,82.643,50.856,cold page cache: regrid 82.6 s vs ~37 s warm; excluded from warm spread
r1_a,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=False, lat/lon output, 1 core",292.69,289.915,1,0.85,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",125.37,1,1200,10,1200x10x180x360,False,0,0.455,36.869,252.59,single-core reference
r16_a,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=False, lat/lon output, 16 cores",72.12,69.086,16,6.74,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",125.38,1,1200,10,1200x10x180x360,False,0,0.451,37.949,30.687,core-count scaling
r32_a,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=False, lat/lon output, 32 cores",77.24,73.937,32,14.53,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",125.39,1,1200,10,1200x10x180x360,False,0,0.435,31.825,41.677,core-count scaling; no gain beyond 16 cores
r8_hp,"100-yr (1200-month) 10-var GCMagicc rollout, dependence=False, HEALPix nside=64 native output (no lat/lon regrid), 8 cores",47.33,44.596,8,7.15,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",125.38,1,1200,10,1200x10x49152,False,0,0.45,0.0,44.146,isolates HEALPix->1deg regridding cost (~37 s of the lat/lon runs)
c8_120,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=False, emitted in 10-yr chunks, 8 cores — replicate 1",100.51,97.537,8,4.76,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",23.2,1,1200,10,1200x10x180x360,False,120,4.479,35.421,57.638,"memory-lean equivalent: identical 1200x10x180x360 output, peak RSS 23.2 GB"
c8_120b,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=False, emitted in 10-yr chunks, 8 cores — replicate 2",101.12,98.214,8,4.96,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",23.21,1,1200,10,1200x10x180x360,False,120,4.556,32.825,60.832,memory-lean equivalent replicate
c8_12,"100-yr (1200-month) 1deg 10-var GCMagicc rollout, dependence=False, emitted in 1-yr chunks, 8 cores",353.36,350.462,8,5.05,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",12.9,1,1200,10,1200x10x180x360,False,12,54.207,33.828,262.427,"lowest peak RSS 12.9 GB, but 5.2 GB of checkpoints reloaded per chunk (54 s of 350 s)"
smoke24,"2-yr (24-month) 1deg 10-var rollout, dependence=False, lat/lon output, 8 cores",5.78,3.22,8,3.72,"AMD EPYC 9555 64-Core Processor (1 socket, 64 physical cores / 128 threads, 4 NUMA nodes, base 1.5-4.41 GHz)",8.36,1,24,10,24x10x180x360,False,0,0.427,0.356,2.437,harness validation only — NOT a 100-year rollout
```
