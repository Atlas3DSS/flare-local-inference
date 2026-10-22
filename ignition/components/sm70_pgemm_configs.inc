// Served Sol-20 r4 instantiations for sm70_prefill.cu (astra-18): X(F, MT, NG, W, BK, PF, MINB) of pgemm4, a subset of
// the reviewed r4 sweep list (sol-20/kernel/sol20_configs_r4.inc; identity4b: 0 mismatches at capacities 128/256/512).
// The r1-r3 entry points stay compiled but have no instantiations; serving calls atlas_pgemm4 only.
#define SOL20_CONFIGS(X)
#define SOL20_CONFIGS2(X)
#define SOL20_CONFIGS4(X) \
  X(false,4,1,4,64,1,4) X(false,6,1,4,64,1,3) X(false,4,2,4,32,1,1) X(false,3,1,4,64,1,5) X(false,8,2,8,64,1,1) \
  X(true,8,2,8,64,1,1) X(true,4,2,4,32,1,1) X(true,8,2,8,64,2,1)
