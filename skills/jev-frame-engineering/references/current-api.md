# Current API guidance

Reviewed against the live TypeSafe System One docs on 2026-10-15. Start future reviews with
the [documentation index](https://docs.typesafe.ai/llms.txt); use normal page URLs if the indexed
Markdown URLs fail. Prefer the feature-specific pages when summaries omit an experimental feature.

## Model selection

The [Models page](https://docs.typesafe.ai/models) lists Jev 1.14, released October 14, 2026.
At this review, `jev-preview` points to `jev-1.14.0`, while `jev-latest` remains on `jev-1.13.0`.
The helpers retain the older pin because the published fixture measurements used it. Compare
1.14 explicitly with `--model jev-1.14.0` on the same labelled data before migrating thresholds;
record the resolved response model. No new live model evaluation accompanies this docs review.

For context budgets see [batching](batching.md); for prices see [question economics](question-economics.md).
The [Jev 1.13 limitations](https://docs.typesafe.ai/model-jaggedness/jev-1.13) describe that version;
neither assume all those defects persist in 1.14 nor assume the new release fixes them.

## Experimental images

The [Images page](https://docs.typesafe.ai/images) requires
`TypeSafe-Danger-Zone-Images: true` on each image or Box request and labels these features
experimental, below production readiness. Images work on Jev 1.13 and 1.14. A top-level `assets`
map supplies image references used in state, instructions, or criteria. Prefer the official
SDK's image handling; the local helpers do not forward assets or experimental headers.

Use distinct references to avoid accidental substitution. Each occurrence expands the image
again, so place shared images in state. Prepare upright PNG/JPEG pixels within the current
size limits; EXIF orientation is not applied. Check the live limits before building an image
workflow, and measure its own accuracy and usage rather than inheriting text-fixture results.

## Experimental Box questions

[Box](https://docs.typesafe.ai/primitives/box) requires Jev 1.14 and the images header. It takes
`type: "box"` and `instructions`, without criteria. The answer contains normalized
`top`, `bottom`, `left`, `right` edges plus confidence. Code maps edges to pixels and checks
geometry before cropping.

Box always returns coordinates, including when the target is absent. Pair it with a presence
Noul and ignore the box when presence is below the application's validated threshold. Name one
target and image per question. Evaluate localization against labelled boxes and absent-target
cases; the text helpers cannot score Box answers. No recommended Box confidence threshold or
formula is documented; do not reuse Choice/Score formulas for it.
