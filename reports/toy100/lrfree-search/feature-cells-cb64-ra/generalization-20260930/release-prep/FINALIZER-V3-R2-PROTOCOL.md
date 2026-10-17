# Corrected metadata-only finalizer preparation

This fresh helper preserves sealed v3 and its first invalid snapshot. The first
validator expected cancelled status CANCELLED; the actual retained RA15 suite
receipt says CANCELLED_BEFORE_TEST_EXECUTION, with no numerical start or log.
The comparison now uses that exact actual status. Original gate validators,
source routes, package/config, replay, suite counts and quality criteria are
unchanged. The helper revision/preparation filename change makes the source
correction explicit. Append inventory also retains earlier closed release-prep
snapshots, including the original metadata error. The new output itself is
archived separately after close.

Run in a fresh child directory under release-prep. Qualification applies to
RA16 on PR155 f459cb6d6aaaabeb1af076ec53ad7a963618de90. The advanced cabe208
upstream remains pending its own source/execution bridge and latest suite.
