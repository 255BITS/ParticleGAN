# Principal: binary_cluster_pairs_d24

Track: geometry (discriminator / lengthscale)

Claim: published absolute batchfeat kernel lengths + init_std=0.5 fail on a
hierarchical two-pair GMM (fine intra-gap 4.0, coarse inter-cluster 24), while
the same shared_c6 GAN formulation still solves it with a lengthscale-free
SimpleMLP control at matched prior init (init_std=6.0).

Solvability gate: in-family control PASS (confirmed).
