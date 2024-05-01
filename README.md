# Type-based unsourced multiple access

This repository contains the Matlab numerical routines of the paper:

[1] K.-H. Ngo, D. P. Krishnan, K. Okumus, G. Durisi, and E. G. Strom, "Type-based unsourced multiple access," arXiv preprint [arXiv:2404.19552](https://arxiv.org/pdf/2404.19552), Apr. 2024. 

Please cite the aforementioned papers if you use this code.

## Content of the repository

This repository contains the following files:

1. `TUMA_GMAC.m`: simulation of the multi-target position tracking problem via type-based unsourced multiple access (TUMA).
2. `AMP.m`: the approximate message passing (AMP) decoder described in Section III-A of [1].
3. `EP.m`: the expectation propagation (EP) decoder described in Section III-B of [1].
4. `scalarAMP.m`: the scalar AMP decoder described in Section III-C of [1].

The repository also contains the functions `emd.m`, `gdf.m`, `gdm.m` used to evaluate the Wasserstein distance between discrete measures, developed by Ulas Yilmaz: https://se.mathworks.com/matlabcentral/fileexchange/22962-the-earth-mover-s-distance
