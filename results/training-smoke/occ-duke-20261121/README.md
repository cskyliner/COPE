# Occluded-Duke training smoke test

On 2026-11-21, tested the original stride-16 configuration on a 24 GiB RTX
4090 using the prepared dataset and OpenAI CLIP initialization. This is a
short diagnostic, not a trained model or an epoch-level reproduction.

- Batch size 64, four images per identity; 20 training batches per stage.
- First stage: 18 actual optimizer updates of 20 batches; first two skipped
  while AMP adjusted its initial scale. Peak allocated 11,410 MiB.
- Second stage: full 15,618-image memory-bank pass, then 20 training batches;
  13 actual optimizer updates as AMP reduced its initial scale. Peak allocated
  17,606 MiB and reserved 18,706 MiB; after warmup, observed updates took about
  0.34 seconds each.
- The second-stage memory bank contained 702 identities; no validation or full
  epoch was run. The short first-stage checkpoint is diagnostic only.

`smoke20.py` imports the unchanged training loop, limits the number of training
batches, and runs one shortened pass in each stage. It fixes local server paths,
disables periodic evaluation, and uses the original 60/100-epoch scheduler
settings for the respective optimizer schedules. `smoke20.log` is the captured
output. Timing excludes eventual full evaluation and is not a full-run estimate.

## Training data checks

Every training image had a corresponding mask for Occluded-Duke (15,618),
Market-1501 (12,936), and P-DukeMTMC-reID (12,927). A real training batch from
each dataset contained 64 images of shape 3x256x128, 64 masks of shape
5x256x128, and 16 identities with 4 images each; sampled image and mask
tensors contained finite values. The Market and P-Duke masks were verified
against the locally downloaded archives before extraction.
