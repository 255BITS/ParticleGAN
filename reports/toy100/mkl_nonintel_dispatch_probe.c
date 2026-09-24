/* Diagnostic only: force MKL's non-Intel vendor branch.
 * On this GenuineIntel host, ordinary dispatch already takes the Intel path,
 * so omitting LD_PRELOAD does not measure that branch.
 * This is not a benchmark fix or a supported deployment configuration.
 */
int mkl_serv_intel_cpu_true(void) { return 0; }
