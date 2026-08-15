# Methodology

## Why these choices

### Per-scene normalization instead of fixed statistics

The first version of this pipeline baked training-AOI band statistics into the
code as constants. That works exactly once. Two Sentinel-2 scenes over
different cities — or the same city in a different season — differ in
atmospheric path radiance and surface reflectance distribution, so applying one
scene's mean and standard deviation to another shifts every band off the
distribution the model learned.

Statistics are now computed at load time from each scene's own non-zero pixels,
in both training and inference. Zero is treated as no-data, which is what
Sentinel-2 L2A writes outside the swath and under the cloud mask.

The cost: normalization is no longer a fixed function, so a patch's normalized
values depend on the scene it came from. For classification this is fine —
relative spectral relationships are preserved — but it does mean absolute
reflectance thresholds cannot be read off the normalized array.

### Frozen encoder

SSL4EO-S12 pretraining saw on the order of a million Sentinel-2 patches. The
reference annotation set here is a few hundred. Fine-tuning end to end on that
ratio does not adapt the encoder, it damages it: the general multispectral
features get overwritten by whatever is idiosyncratic about one city.

So only the decoder and segmentation head train. After the decoder converges,
a small number of low-LR batches run with the encoder unfrozen — enough for
batch-norm statistics to settle on local reflectance, not enough to unpick the
representations. `encoder_warmup_batches: 0` disables this.

### Non-overlapping validation

The training grid uses `stride=128` with `patch_size=256`, so adjacent patches
share half their pixels. Splitting that index list randomly means a held-out
patch shares pixels with several training patches, and validation loss then
partly measures memorization.

`build_disjoint_split()` draws validation patches from a non-overlapping grid
(`stride == patch_size`), then drops every training patch whose window
intersects a validation window. This costs training data — a ring of patches
around each validation tile is discarded — which is the correct trade.

### Weighted sampling

Class frequencies in the Aligarh AOI are heavily skewed toward agriculture and
background. Water and barren occupy a small fraction of pixels, and with
uniform sampling they contribute almost nothing to the gradient. Patches are
weighted by the inverse frequency of their dominant class.

This is coarse — it weights by dominant class, not by pixel composition — but
patch-level weighting is what the sampler API supports, and pixel-level class
weights in the loss would interact awkwardly with Dice.

### Logit averaging at inference

Overlapping windows must be combined somehow. Averaging argmax labels produces
visible discontinuities at tile boundaries wherever two windows disagree.
Averaging logits before argmax blends the decision smoothly, and costs only the
memory for a `(num_classes, H, W)` float32 accumulator — which for a large
scene is the pipeline's real memory ceiling.

A 5-pixel median filter afterwards removes isolated single-pixel predictions
that survive the averaging.

---

## Explainability

Three methods, because each fails differently:

**Gradient magnitude.** Fast, one backward pass per class. Susceptible to
gradient saturation — a band the model relies on completely can show near-zero
gradient if the activation is already saturated. The class score is the *mean*
logit over the patch; using the max makes the result depend on one lucky pixel.
Gradients are *not* multiplied by input magnitude, which would simply re-rank
bands by reflectance scale and hand SWIR an automatic win.

**Occlusion.** Zero each band, measure the drop in each class's mean logit.
Slowest (one forward pass per band) but the most directly interpretable, and it
makes no linearity assumption. Its blind spot is correlated bands: zeroing one
of two redundant bands shows little drop even though the pair matters.

**SHAP `GradientExplainer`.** Shapley-axiom grounded, so attributions are
consistent and additive in a way the other two are not. Requires the `shap`
package and more compute; the pipeline degrades gracefully to the gradient
result if it is unavailable.

All three L1-normalize per class, so each class's bars sum to 1 and classes
compare to one another rather than to their own logit magnitudes.

**Sanity check.** NIR (B8) and Green (B3) should dominate for forest and
agriculture — that is the vegetation red-edge response, and it is not a
subtle effect. If they do not, suspect the band ordering in the input GeoTIFF
before you suspect the model.

### SegGradCAM

Grad-CAM adapted to segmentation by targeting a class over a spatial mask
rather than a single output logit. The target layer is the last convolutional
stage of the UNet++ decoder: still spatially arranged, already
class-discriminative. The segmentation head is too shallow to carry semantics;
encoder layers are too generic and too coarse.

---

## Known limitations

**Single-AOI training.** The reference masks cover one city. Cross-AOI
generalization is asserted by the architecture (pretrained encoder, per-scene
normalization) rather than demonstrated. Validating on a second annotated city
is the obvious next step and has not been done.

**Masks are paired by filename order.** `load_masks()` matches masks to patch
coordinates by sorted filename, assuming both were generated by
`make_patches.py` with identical parameters over identically-sized rasters.
There is no georeferencing check. Mismatched counts abort fine-tuning; a
mismatched *order* would silently train on misaligned labels.

**The validation set is small.** `train.py` scores held-out patches only, but
with `val_fraction: 0.1` on a non-overlapping grid that is a handful of tiles.
Per-class IoU for a rare class can hinge on a single patch, so treat individual
class scores as indicative rather than precise. Raise `val_fraction` for a
steadier estimate, at the cost of training data.

**Road merged into building.** At 10 m resolution a two-lane road is well under
one pixel wide and rarely forms a connected component. Keeping it as a class
produced near-zero IoU and dragged the building class down with it. The merge
is a modelling decision, not a data-quality fix, and it means the maps cannot
distinguish transport infrastructure from built-up area.

**Median composite hides temporal signal.** The GEE export takes an annual
median to suppress cloud. Seasonal crop cycles — genuinely informative for
distinguishing agriculture from barren — are averaged away. A multi-season
stack would help and would also multiply the input bands.

**Background is a real class, not a mask.** Class 0 covers both no-data and
genuinely unclassifiable pixels, so it absorbs errors that would otherwise be
visible. Reported background IoU should not be read as meaningful.
