/* Diagnostic only: test Intel MKL's vendor dispatch on AVX2-capable AMD.
 * This is not a benchmark fix or a supported deployment configuration.
 */
int mkl_serv_intel_cpu_true(void) { return 1; }
