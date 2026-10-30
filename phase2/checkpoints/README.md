# Final Phase 2 checkpoints (seven files, 489 MB)

Attached to the GitHub release **`phase2-final`** of this repository (several exceed GitHub's 100 MB file limit).
Download all seven into `phase2/inference/checkpoints/` and verify:

| File | Network | Size | SHA-256 |
|---|---|---|---|
| `locoformer_d32.pth` | TF-Locoformer member 1 | 60 MB | `1a3d066c39c4f775573e0109f55c702f8755da81c07bbb3ace31fdcf265adaea` |
| `locoformer_orig.pth` | TF-Locoformer member 2 | 60 MB | `186c07d328fb0e06623d2d1f6deeb6e9a9247da3bd8e4704d717c3653eaa69ff` |
| `locoformer_spec.pth` | TF-Locoformer member 3 | 60 MB | `81818146b2f2b9825844c49d4d7d381c440dc262607f5e52011b22b0425583f9` |
| `locoformer_extra.pth` | TF-Locoformer member 4 | 60 MB | `f2556d47dffac540f6375c3a6d0bc01e7a4c29aa724586cef51eddcf408fe57c` |
| `predictor_d32_s800.pt` | TF-GridNet (filename historical) | 88 MB | `703057acdaa470b96a3d0f2dd377dd333125acf51d3e5ed95662021bd5212953` |
| `bsrnn_extended_s1600.ckpt` | URGENT BSRNN | 151 MB | `af4366a83e17cd90e2b31493914fb51cc91b258616c7999b0c1f30f75a8dc607` |
| `t2_mpdns_publicnoise_hilr_1008.pt` | MP-SENet | 9 MB | `78228aaaa9ad581adf55360192ee119d1c704ea144b6b85af70a789437261b1e` |

```bash
cd phase2/inference/checkpoints && sha256sum -c ../../checkpoints/SHA256SUMS
```
