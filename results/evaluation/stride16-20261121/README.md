# COPE released-checkpoint evaluation

Ran on 2026-11-21 on one NVIDIA RTX 4090 with Python 3.10.22,
PyTorch 2.5.1+cu121 and the COPE stride-16 configs. PSS was enabled with
K1=200 and K2=5; image size was 256x128. These are evaluations of released
checkpoints, not models trained in this workspace.

| Evaluation set | Checkpoint | Rank-1 | mAP | Rank-5 | Rank-10 | Inference |
|---|---|---:|---:|---:|---:|---:|
| Market-1501 | cope_market.pth | 96.3 | 93.0 | 98.6 | 99.1 | 32.35 s |
| P-DukeMTMC-reID | cope_pduke.pth | 93.2 | 88.4 | 94.8 | 95.5 | 17.74 s |
| Occluded-ReID | cope_occreid_preid.pth | 88.7 | 88.2 | 93.9 | 96.6 | 3.87 s |
| Partial-REID | cope_occreid_preid.pth | 88.0 | 88.3 | 90.3 | 94.0 | 1.76 s |

The Occluded-ReID original images were flattened and given synthetic camera
IDs 1 (occluded queries) and 2 (whole-body gallery), following the split and
camera treatment of BPBreID. These IDs are not original camera metadata.
Partial-REID uses `occluded_body_images` as query, as the COPE loader does.
The logs include full configurations, counts and commands. They contain local
server paths but no image data or model weights. Inference times are hardware
specific and exclude model construction and dataset initialization.

The earlier Occluded-Duke run is recorded in `../occ_duke_eval.log`. MSMT17
was not available at the time of these runs. Paper table values have not yet
been verified against the original PDF, so no equivalence is asserted here.
