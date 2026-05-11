# Single-Symbol Localization in Circuit Diagrams
## Using Classical Computer Vision and Few-Shot Learning Techniques

### Product Requirements Document + Technical Design Document

**Version:** 1.0  
**Classification:** Research-Grade Engineering Design Document  
**Domain:** Computer Vision · Few-Shot Learning · Industrial Document Analysis  
**Status:** Proposed Architecture

---

> **Motivating Example (Analyzed in This Document):**  
> Target symbol = Mutual Reactor (MR) — two stacked inductor humps with a `+` terminal marker  
> Source diagram = Single-line power distribution schematic (IEC/IEEE-style)  
> Challenge = Detect all ~7 MR instances among visually similar inductors, text, wire intersections, and other symbols on a noisy scan

---

## Table of Contents

1. Executive Summary  
2. Problem Characteristics  
3. System-Level Design Philosophy  
4. Full End-to-End Pipeline Overview  
5. Image Preprocessing  
6. Candidate Region Proposal  
7. Synthetic Template Augmentation  
8. Similarity Matching Methodologies  
9. PCA-Based One-Shot Learning Refinement  
10. Graph-Based Structural Verification  
11. Siamese Network Extension (Research Extension)  
12. Non-Maximum Suppression  
13. Evaluation Methodology  
14. Baseline Systems  
15. Final Recommended Pipeline  
16. Implementation Plan  

---

---

# 1. Executive Summary

## 1.1 Precise Problem Statement

We are given exactly two inputs:

1. **A large circuit diagram image** `I ∈ ℝ^{H×W}` — a single-line power distribution schematic, likely a scan or export from CAD software (EPLAN, AutoCAD Electrical, etc.). The image contains dozens of heterogeneous symbols: inductors, grounds, motors, relays, junction dots, text annotations, and wire segments.

2. **A single target symbol template** `T ∈ ℝ^{h×w}` — in our case, the Mutual Reactor (MR) symbol: two stacked inductor coils with a `+` terminal marker above the upper coil.

The task is **exhaustive spatial localization**: find the 2D bounding box `{(x_i, y_i, w_i, h_i)}` of every occurrence of the target symbol in `I`, under the following degradation conditions:

- Raster scan artifacts (JPEG/PNG compression noise, uneven exposure)
- Slight rotational variance (±5°) from scan skew or manual drafting
- Non-uniform line thickness across instances
- Symbol clutter: wires passing through or adjacent to symbols
- Scale variation (symbols at slightly different sizes)
- Text labels immediately below each symbol

## 1.2 Why This Is a Template Localization / Few-Shot Matching Problem

This is **not** a conventional object detection problem. Standard deep learning detection pipelines (Faster R-CNN, YOLO, DETR) require:
- Thousands of labeled bounding boxes across diverse images
- GPU-accelerated training runs (hours to days)
- Domain-specific augmentation strategies that require human knowledge of failure modes

We have **zero** labeled training examples — only one template image. The problem is more precisely characterized as:

| Problem Type | Definition | Our Case |
|---|---|---|
| Template Matching | Find one image patch within another | ✅ Closest match |
| Few-Shot Detection | Detect using 1–5 reference shots | ✅ One-shot variant |
| Structured Pattern Search | Exploit known topology of target | ✅ Circuit symbols are topologically rigid |
| Exemplar-Based Retrieval | Query image → retrieve spatial locations | ✅ Spatial retrieval |

The problem has **three defining properties** that distinguish it from general object detection:

**Property 1 — Geometric Rigidity:** Circuit symbols are standardized (IEC 60617, IEEE Std 315). Their topology — number of loops, junction counts, terminal positions — is invariant across instances. This is exploitable.

**Property 2 — Edge Dominance:** Circuit diagrams carry almost zero texture information. Structure is encoded entirely in line topology. This means edge-based descriptors are maximally discriminative.

**Property 3 — Low Appearance Entropy:** The appearance manifold of a single symbol class (e.g., MR) is tight. Intra-class variance is low (mostly noise and minor scale/thickness changes). This justifies PCA-based subspace models built from synthetic augmentations.

## 1.3 Why Standard Object Detection Is Unsuitable

| Issue | Detail |
|---|---|
| **No training data** | YOLO/DETR require ≥500–1000 annotated examples minimum for reliable learning |
| **Domain gap** | Pretrained ImageNet features are texture-biased; circuit symbols are pure line art |
| **Overkill computation** | Running a full detection model for a single-query one-shot problem wastes compute |
| **Interpretability** | Engineering applications require explainable decisions; neural detectors are black boxes |
| **Deployment constraints** | Industrial tools may not have GPU infrastructure |

## 1.4 Why Classical CV Is Highly Relevant Here

Classical computer vision methods were historically developed for **structured, low-texture, high-precision industrial vision tasks** — precisely the regime circuit diagrams occupy. Key advantages:

- **Deterministic behavior:** Given the same preprocessing, results are reproducible
- **Mathematical interpretability:** Every decision maps to a computable geometric or statistical quantity
- **No training phase:** The pipeline is ready to use on any new symbol immediately
- **Robustness by design:** Methods like Chamfer matching were explicitly designed for noisy edge images

---

# 2. Problem Characteristics

## 2.1 Geometric Nature of Circuit Symbols

Circuit symbols are **topological objects**, not photometric ones. The MR symbol in our case study is defined by:

```
      +           ← Terminal marker (small cross or dot)
    ╔═══╗
   (  ○  )        ← Upper coil loop (inductor hump 1)
   (  ○  )        ← Lower coil loop (inductor hump 2)
    ╚═══╝
```

Formally, the symbol can be described as a **planar graph** `G = (V, E)` where:
- V = {terminal junction, loop apex 1, loop nadir 1, loop apex 2, loop nadir 2, wire endpoints}
- E = directed curved line segments connecting V

This graph structure is **invariant** to:
- Mild scale changes (aspect ratio preserved)
- Small rotations (topology unchanged)
- Noise (local perturbations do not alter connectivity)

This topological rigidity is the primary asset of our approach.

## 2.2 Sparsity of Circuit Diagrams

A circuit diagram image is **extremely sparse** in the information-theoretic sense. For a 3000×800 pixel scan:
- Foreground (line/symbol) pixels: typically 3–8% of total pixels
- Background (white space): 92–97%

This has algorithmic implications:
- Sliding window approaches evaluate mostly empty regions — wasted computation
- Connected component analysis will yield a tractable number of components (dozens to hundreds, not millions)
- Binarization is effective and lossless for line art

## 2.3 Edge and Topology Dominance Over Texture

In natural images, texture (frequency content) is as discriminative as edges. In circuit diagrams:

- All discriminative information lives in **edge topology** (line curvature, junctions, loop counts)
- No textures exist — all regions are uniform white or uniform black
- This means pixel-intensity methods (raw cross-correlation on gray images) are **suboptimal**
- Edge-based methods (Chamfer matching, distance transforms on edge maps) will **outperform** intensity methods

This is not merely empirical intuition — it follows from the information content analysis: the mutual information `I(T_edges; I_edges)` at correct alignment is far higher relative to background than `I(T_gray; I_gray)`.

## 2.4 Challenge Analysis

### 2.4.1 Wire Clutter
Circuit diagrams contain a dominant horizontal bus wire (in our example image) that **passes through or directly adjacent to** all MR symbols. This means:
- A candidate region containing an MR will always contain wire segments
- The template must not be confused by partial wire matches
- Matching must focus on the **enclosed topology** of the symbol, not surrounding wires

### 2.4.2 Symbol Overlap and Proximity
In the example diagram, MR symbols appear at regular intervals along a bus, with spacing of approximately 1 symbol-width. This creates:
- Risk of merged connected components (two adjacent symbols joining into one region)
- Risk of NMS over-suppression (IoU between adjacent detections)

### 2.4.3 Rotational Variance
Scanned documents may have ±2–5° skew. Additionally, some circuit drawing conventions rotate symbols:
- 0° (nominal, vertical coil axis)
- 90° (coil axis horizontal, for horizontal bus connections)

The template augmentation pipeline must generate sufficient rotational coverage.

### 2.4.4 Varying Scan Quality
The provided example image exhibits:
- Slight JPEG compression artifacts (block artifacts at 8×8 boundaries)
- Uneven ink density (some MR symbols appear lighter than others)
- Background not perfectly white (slight grayish tone ~240/255)

Adaptive thresholding handles uneven illumination; median filtering handles JPEG noise.

### 2.4.5 Scale Variation
Even within a single diagram, symbols may appear at slightly different scales due to:
- Inconsistent CAD symbol libraries used by the drafter
- Non-uniform scanning/zooming during capture
- Multi-page diagrams assembled at different DPI

Empirically, scale variation within a single diagram is typically ±10–15%.

### 2.4.6 Partial Occlusion
Text labels (e.g., "MR", "Z2", "N2", "A1", "Y1", "Z1") appear directly below each symbol. If the diagram has dense labeling, the label may partially occlude the bottom wire of the symbol. Matching must be robust to this.

### 2.4.7 Nearby Text
The alphanumeric annotations in the diagram use a similar line weight to the symbols. This creates false positive risk: partial letter shapes (e.g., "M" from "MR") may resemble inductor loops at certain scales.

---

# 3. System-Level Design Philosophy

## 3.1 Why a Hybrid Classical CV + Similarity Matching Pipeline Is Optimal

The design space of approaches for this problem can be characterized by three axes:

```
           High Data
               │
               │  Deep Learning (YOLO, DETR, Faster R-CNN)
               │  Siamese Networks (with large support sets)
               │
   Classical ──┼────────────────────────── Neural
   CV          │
               │  ← WE ARE HERE
               │  Hybrid: Classical preprocessing +
               │  Similarity matching + PCA refinement
               │
               │  Raw template matching (baseline)
           Low Data
```

The hybrid approach is optimal because:

1. **Preprocessing reduces noise without requiring learning** — Binarization, morphological operations, and skeletonization are parameter-driven, not data-driven
2. **Region proposal is purely geometric** — Connected component analysis requires no learned features
3. **Similarity matching exploits known structure** — Chamfer distance and NCC exploit the edge-dominant nature of circuit diagrams
4. **PCA is regularized by augmentation** — A small synthetic augmentation set (50–200 variants) gives PCA enough variance to model the true appearance distribution

## 3.2 Why Fully Supervised Deep Learning Is Unrealistic

Beyond data scarcity, there are systemic issues:

**Problem 1 — Domain Specificity:** There is no large-scale publicly available dataset of annotated electrical single-line diagrams. The closest (COCO, OpenImages) contains zero circuit symbols.

**Problem 2 — Symbol Vocabulary Explosion:** A real engineering system must localize dozens of symbol types (resistors, capacitors, transformers, relays, circuit breakers, etc.). Building a supervised dataset for each would require enormous expert annotation effort.

**Problem 3 — Symbol Standard Variance:** IEC, IEEE, JIS, and AS/NZS standards all have different graphical representations for the same component. A deep learning model trained on one standard may fail silently on another.

**Problem 4 — Update Frequency:** Engineering teams regularly introduce new custom symbols. The pipeline must accommodate new queries at zero marginal cost.

## 3.3 The Role of Region Proposal

Region proposal is the architectural decision that most dramatically reduces computational cost. Without it:

**Naive sliding window cost:**
For image `I ∈ ℝ^{3000×800}` and template `T ∈ ℝ^{60×40}`:
- Number of positions: `(3000-60) × (800-40) = 2,236,800` positions
- At `S = 5` scales and `R = 12` rotations: `2,236,800 × 60 = 134,208,000` evaluations
- Each evaluation involves a patch comparison of cost O(h×w) = O(2400)
- Total: **~3.2 × 10^11 operations** — prohibitive

With region proposal (connected components):
- Typical circuit diagram: 50–300 candidate regions
- After area/aspect-ratio filtering: 20–80 candidates
- Per-candidate matching at all scales/rotations: `80 × 60 = 4,800` evaluations
- **Speedup: 4–6 orders of magnitude**

## 3.4 Precision vs Recall Tradeoffs

The system operates under an asymmetric cost model:

| Error Type | Consequence | Cost |
|---|---|---|
| False Negative (missed detection) | Symbol not found — dangerous in engineering validation | HIGH |
| False Positive (spurious detection) | Extra annotation for human to review | LOW |

**Design implication:** Set thresholds to favor **high recall over high precision**. Use NMS aggressively to prune duplicates, but use matching thresholds conservatively to avoid misses.

---

# 4. Full End-to-End Pipeline Overview

## 4.1 Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                    SYSTEM INPUT                                      │
│                                                                      │
│   Circuit Diagram I ∈ ℝ^{H×W×C}    Template T ∈ ℝ^{h×w×C}         │
└────────────────────┬──────────────────────────┬─────────────────────┘
                     │                          │
                     ▼                          ▼
┌────────────────────────────────────────────────────────────────────┐
│                 STAGE 1: PREPROCESSING                              │
│                                                                     │
│  I → Grayscale → Denoise → Binarize → Morphology → I_bin           │
│  T → Grayscale → Denoise → Binarize → Morphology → T_bin           │
└────────────────────┬──────────────────────────┬────────────────────┘
                     │                          │
                     ▼                          ▼
          ┌──────────────────┐      ┌───────────────────────────┐
          │ STAGE 2:         │      │ STAGE 3:                  │
          │ REGION PROPOSAL  │      │ TEMPLATE AUGMENTATION     │
          │                  │      │                           │
          │ Connected Comp.  │      │ Rotate × Scale ×          │
          │ → Filter by      │      │ Thickness jitter          │
          │   area/aspect    │      │ → T_aug = {T_1...T_N}     │
          │ → Candidates R   │      │ → Build PCA model         │
          └────────┬─────────┘      └───────────────┬───────────┘
                   │                                │
                   ▼                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 STAGE 4: MULTI-SCALE SIMILARITY MATCHING            │
│                                                                      │
│  For each candidate r ∈ R:                                          │
│    For each scale s ∈ S:                                            │
│      Compute: NCC(r, T_s), Chamfer(r, T_s), Hausdorff(r, T_s)      │
│      → Score vector v_r                                             │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 STAGE 5: PCA SUBSPACE REFINEMENT                    │
│                                                                      │
│  For top-K candidates from Stage 4:                                 │
│    Project candidate patch onto PCA eigenspace                      │
│    Compute reconstruction error e_r                                 │
│    Rerank by combined score: α·sim + β·(1-e_r)                      │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 STAGE 6: NON-MAXIMUM SUPPRESSION                    │
│                                                                      │
│  Sort detections by confidence score                                │
│  Greedily suppress detections with IoU > θ_nms with higher-score   │
│  neighbor → Final detection set D                                   │
└────────────────────────────┬────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                 OUTPUT                                               │
│                                                                      │
│  D = {(x_i, y_i, w_i, h_i, score_i)} for each detected symbol      │
│  Visualization: Bounding boxes overlaid on original I               │
└─────────────────────────────────────────────────────────────────────┘
```

## 4.2 Stage-by-Stage Summary

| Stage | Input | Output | Key Operation | Failure Mode |
|---|---|---|---|---|
| Preprocessing | Raw images | Binary edge maps | Adaptive threshold + morph | Over-smoothing destroys loops |
| Region Proposal | Binary diagram | Candidate bounding boxes | Connected component analysis | Wire connectivity merges symbols |
| Template Augmentation | Single template | N augmented variants + PCA model | Rotation/scale transforms | Unrealistic augmentations reduce PCA quality |
| Similarity Matching | Candidates + templates | Similarity scores per candidate | NCC + Chamfer | Noisy edge maps inflate Chamfer scores |
| PCA Refinement | Top-K candidates | Refined scores | Subspace projection | Low sample count degrades eigenspace |
| NMS | Scored detections | Deduplicated detections | IoU-based suppression | Aggressive NMS removes valid nearby detections |

---

# 5. Image Preprocessing

## 5.1 Design Rationale

Preprocessing serves three purposes specific to circuit diagrams:
1. **Noise removal** — Eliminate scanner artifacts and JPEG ringing without destroying thin lines
2. **Binarization** — Convert to binary {0,1} image since all information is in foreground/background distinction
3. **Normalization** — Equalize line thickness across the image to reduce intra-class variance

The key constraint is **topological preservation**: preprocessing must not break wires, fill loop interiors, or merge nearby symbols.

## 5.2 Grayscale Conversion

For color inputs (RGB), convert to grayscale using luminance weighting:

```
I_gray = 0.2126·R + 0.7152·G + 0.0722·B
```

The weights correspond to human photometric sensitivity. For circuit diagrams (which are always black-on-white or dark-on-light), simpler averaging also works:

```
I_gray = (R + G + B) / 3
```

For our example image (which appears to be already grayscale), this step is a no-op.

**Why not use color channels separately?** Circuit diagrams carry zero color-discriminative information. Using color would add noise without signal.

## 5.3 Noise Filtering

### 5.3.1 Gaussian Filtering

The Gaussian kernel is defined as:

```
G(x, y; σ) = (1 / (2πσ²)) · exp(-(x² + y²) / (2σ²))
```

Applied as convolution: `I_smooth = I_gray * G(σ)`

**Parameter choice:** σ = 0.5–1.0 for high-resolution scans (300 DPI), σ = 0.3–0.7 for 72–150 DPI images.

**Purpose:** Attenuate Gaussian white noise and JPEG DCT artifacts before binarization.

**Tradeoff:** Large σ blurs thin lines and merges nearby parallel lines. For circuit diagrams with thin wire lines (1–2 pixels at 96 DPI), σ > 1.5 is destructive.

### 5.3.2 Median Filtering

Median filtering replaces each pixel with the median of its `k×k` neighborhood:

```
I_median(x,y) = median{I(x+i, y+j) : i,j ∈ [-k/2, k/2]}
```

**Purpose:** Removes salt-and-pepper noise (impulse noise from scanner) while preserving edge sharpness. Unlike Gaussian filtering, median filtering does not blur edges because it is a rank-order filter that ignores outlier values.

**Parameter choice:** k=3 is standard. k=5 is appropriate only when noise density is very high.

**When to use over Gaussian:** When the image has discrete isolated noise pixels (common in scans) rather than distributed Gaussian noise. For the provided example with JPEG artifacts, a light Gaussian pre-filter followed by median is optimal.

## 5.4 Binarization

### 5.4.1 Otsu's Thresholding

Otsu's method finds the optimal global threshold `τ*` by maximizing the between-class variance:

```
σ²_B(τ) = ω_0(τ) · ω_1(τ) · [μ_0(τ) - μ_1(τ)]²
```

where:
- `ω_0(τ)` = fraction of pixels below threshold τ (background class weight)
- `ω_1(τ)` = fraction of pixels above threshold τ (foreground class weight)
- `μ_0(τ)`, `μ_1(τ)` = class means

`τ* = argmax_τ σ²_B(τ)`

**Implementation:** Compute over histogram `H[0..255]`. Cumulative sum gives `ω_0`, `ω_1`, `μ_0`, `μ_1` in O(256) time.

**Limitation for circuit diagrams:** Otsu assumes bimodal histogram. If the image has significant background variation (uneven illumination from scanning), the single global threshold fails — part of the image is correctly binarized while another part is under/over-thresholded.

**When Otsu succeeds:** High-quality clean scans with uniform background — our example image is close to this, as it has a relatively uniform light gray background.

### 5.4.2 Adaptive Thresholding

For non-uniform illumination, adaptive thresholding computes a local threshold for each pixel based on a local neighborhood:

```
τ(x,y) = μ(x,y; w) - C
```

where `μ(x,y; w)` is the mean (or Gaussian-weighted mean) over a `w×w` window centered at (x,y), and `C` is a small constant (typically 5–15) to bias toward thresholding slightly below local mean.

**Gaussian adaptive variant:**
```
τ(x,y) = G(x,y; σ_local) * I_gray - C
```

**Parameter choice for circuit diagrams:**
- Window size `w`: Should be 3–5× the width of the thickest expected line. For the MR symbol at typical resolution: w = 15–25 pixels.
- Constant `C`: 7–12 for standard scan quality.

**Why adaptive is preferred:** Even within our example image, the left half and right half may have slightly different exposure from the scanner. Adaptive thresholding handles this transparently.

**Failure mode:** If `w` is too small (smaller than symbol feature size), the local mean is pulled toward the feature itself and the threshold follows the feature — resulting in inconsistent binarization.

## 5.5 Morphological Operations

After binarization, morphological operations refine the binary image structure.

### 5.5.1 Mathematical Morphology Foundations

Let `A` be the binary foreground set and `B` be the structuring element (SE).

**Erosion:**
```
A ⊖ B = {z : B_z ⊆ A}
```
A pixel is retained only if the entire SE, centered at that pixel, is contained in A. This shrinks foreground objects and removes thin protrusions.

**Dilation:**
```
A ⊕ B = {z : B̂_z ∩ A ≠ ∅}
```
A pixel is included if any part of the SE overlaps with A. This grows foreground objects and fills small gaps.

**Opening:** `A ∘ B = (A ⊖ B) ⊕ B`  
Removes small isolated noise objects (smaller than SE) without significantly changing the shape of larger objects.

**Closing:** `A • B = (A ⊕ B) ⊖ B`  
Fills small gaps and holes in foreground objects.

### 5.5.2 Application Strategy for Circuit Diagrams

The morphological processing strategy must balance two competing objectives:

**Objective A (Noise removal):** Remove isolated noise pixels, scanner dust, text artifacts that could generate false candidates.

**Objective B (Topology preservation):** Preserve the precise loop structure of inductor coils — the two humps that define the MR symbol topology.

**Strategy:**

```
1. Light opening with 1×1 or 2×2 SE → Remove noise pixels
2. SKIP closing (it fills loop interiors — destroys coil topology)
3. Dilation with 1×1 SE → Reconnect broken thin lines from binarization
```

**Critical constraint:** Do NOT apply closing to circuit diagram images. Closing fills the interior of loops — which is the most distinctive feature of inductor and transformer symbols. A closed loop becomes a filled blob, destroying all topological information.

## 5.6 Skeletonization / Thinning

Skeletonization reduces binary foreground objects to their medial axis — a 1-pixel-wide skeleton that preserves topology.

**Zhang-Suen thinning algorithm:** Iteratively removes boundary pixels that satisfy both:
1. `2 ≤ N(p) ≤ 6` (connectivity constraint: not isolated, not articulation point)
2. `A(p) = 1` (no 180° rotation of 01 pattern — prevents cutting)
3. `p2 · p4 · p6 = 0` or `p4 · p6 · p8 = 0` (sub-iteration conditions)

This is iterated until no more pixels can be removed.

**Mathematical guarantee:** Zhang-Suen preserves 8-connectivity of the foreground and 4-connectivity of the background. The skeleton is homotopy-equivalent to the original — meaning it has the same loop count and connectedness.

**Why this matters for MR matching:** After skeletonization:
- All MR instances are reduced to identical 1px-thick loop structures
- Line thickness variance is eliminated
- Matching becomes purely topological, not photometric

**Failure mode:** If binarization introduced breaks in lines, skeletonization may disconnect the skeleton at those breaks. A light dilation before skeletonization mitigates this.

**Computational cost:** O(H·W·K) where K is the number of iterations (typically 20–50 for realistic symbols).

---

# 6. Candidate Region Proposal

## 6.1 The Exhaustive Window Problem

As computed in Section 3.3, exhaustive sliding window matching is computationally prohibitive. The fundamental insight motivating region proposal is:

> **Circuit diagrams are sparse.** Only 3–8% of pixels are foreground. Candidates for symbol matching exist only in foreground-dense regions. Background windows can be rejected with near-zero cost.

The region proposal stage exploits this sparsity to reduce the candidate set by 3–5 orders of magnitude before applying expensive matching operations.

## 6.2 Connected Component Analysis

### 6.2.1 Definition

In a binary image `I_bin`, a connected component is a maximal set of foreground pixels such that every pixel in the set is 8-connected (or 4-connected) to at least one other pixel in the set.

**8-connectivity:** Pixel (x,y) is connected to all 8 neighbors: `(x±1, y), (x, y±1), (x±1, y±1)`

**4-connectivity:** Only horizontal and vertical neighbors: `(x±1, y), (x, y±1)`

**Which to use for circuit diagrams:** 8-connectivity is preferred because diagonal line segments (in angled circuit elements) should form connected components.

### 6.2.2 Algorithm: Two-Pass Labeling (Rosenfeld-Pfaltz)

**Pass 1:** Scan left-to-right, top-to-bottom. Assign a provisional label to each foreground pixel. If a pixel has labeled neighbors, assign the minimum label and record equivalences.

**Pass 2:** Resolve all label equivalences using Union-Find. Replace all provisional labels with their canonical root label.

**Complexity:** O(H·W·α(n)) where α is the inverse Ackermann function (effectively constant). In practice this is O(H·W).

### 6.2.3 Connected Component Extraction Results

For the example circuit diagram:
- After binarization and light morphology, we expect approximately:
  - 6–10 wire segment components (merged by bus wire)
  - 7–9 MR symbol components (possibly merged with bus wire)
  - 2–3 ground symbol components
  - 2 motor symbol components
  - Multiple text character components
  - 1 G/B box component

**Critical complication — Bus Wire Merging:** In single-line diagrams, all symbols are connected to the main bus wire. If the bus wire and the symbols are represented as one connected foreground region, the entire diagram is one component. This is the most significant failure mode for connected component-based region proposal.

**Solution — Wire Removal via Morphology:**

```
1. Apply horizontal morphological erosion: I ⊖ H where H = 1×K horizontal SE (K ≈ 30px)
   → This removes horizontal line segments longer than K pixels (the bus wire)
   → Symbol components (which have loops, not long horizontals) are preserved

2. Subtract: I_no_wire = I_bin XOR (I_bin ⊖ H ⊕ H)
   → This isolates the symbols from the wire network
```

This morphological opening with a long horizontal SE is a form of **background subtraction specific to horizontal line structure** — highly effective for single-line electrical diagrams.

## 6.3 Bounding Box Extraction and Filtering

For each connected component `C_i`, compute the axis-aligned bounding box:

```
x_min_i = min{x : (x,y) ∈ C_i}
x_max_i = max{x : (x,y) ∈ C_i}
y_min_i = min{y : (x,y) ∈ C_i}
y_max_i = max{y : (x,y) ∈ C_i}

BBox_i = (x_min_i, y_min_i, x_max_i - x_min_i, y_max_i - y_min_i)
```

### 6.3.1 Area Filtering

Remove components whose bounding box area falls outside the expected range:

```
A_min ≤ |BBox_i| ≤ A_max
```

For MR symbol at 96 DPI with typical drawing scale:
- Estimated symbol size: ~60×80 pixels
- Expected area range: `[30×40, 120×160]` = `[1200, 19200]` pixels²
- With ±25% scale variance: `[900, 24000]` pixels²

This filter removes:
- Single isolated noise pixels (A << A_min)
- Large merged structures like the bus wire or multi-symbol components (A >> A_max)

### 6.3.2 Aspect Ratio Filtering

For MR symbol: width ≈ 60px, height ≈ 80px → aspect ratio ≈ 0.75

Filter:
```
AR_min ≤ w_i / h_i ≤ AR_max
```

For our case: `0.4 ≤ AR ≤ 1.8` (generous to allow for rotations and scale variance)

This rejects:
- Pure horizontal wire segments (AR >> 2.0)
- Pure vertical wire segments (AR << 0.3)
- Square-ish noise blobs at wrong aspect ratio

### 6.3.3 Edge Density Filtering

Circuit symbols are edge-dense relative to their bounding box area. Compute:

```
ρ_i = |C_i| / |BBox_i|
```

where `|C_i|` = number of foreground pixels in component, `|BBox_i|` = area of bounding box.

For MR symbol: the two coil loops fill roughly 30–50% of their bounding box → `ρ_MR ≈ 0.3–0.5`

For wire segments: foreground pixels / bbox area ≈ 1/width → very low density → `ρ_wire < 0.1`

Filter: `ρ_min ≤ ρ_i ≤ ρ_max` where `ρ_min = 0.15`, `ρ_max = 0.8`

This selects for loop-containing structures and rejects linear wire segments.

### 6.3.4 Circularity / Compactness

For components that might be roughly round (like motor circles):

```
circularity = 4π · |C_i| / perimeter_i²
```

Circular shapes have circularity ≈ 1.0; elongated or complex shapes have lower values.

For MR symbol: two loops create a complex contour → circularity < 0.3 (useful for rejecting round symbols like motor M circles)

### 6.3.5 Composite Filter Logic

```python
def filter_candidates(components, template_size):
    h_t, w_t = template_size
    candidates = []
    for comp in components:
        area = comp.bbox_area()
        ar = comp.width / comp.height
        density = comp.pixel_count / comp.bbox_area()
        
        if (0.5 * w_t * h_t) < area < (4.0 * w_t * h_t):
            if 0.3 < ar < 2.5:
                if 0.1 < density < 0.85:
                    candidates.append(comp)
    return candidates
```

## 6.4 Contour Hierarchy Analysis

OpenCV's `findContours` function returns a **contour hierarchy** — a parent-child tree structure where:
- Level 0: Outer contours (entire symbol boundary)
- Level 1: Inner contours (holes = loop interiors)

For the MR symbol: the two coil loops create **exactly 2 inner contours** (holes in the binary image). This is a powerful discriminative signal:

```
For a valid MR candidate: hierarchy_depth ≥ 1, inner_contour_count == 2
```

Using hierarchy to filter:
- Zero inner contours: straight wire, text, junction dot → REJECT
- 1 inner contour: single-loop inductor (not MR) → REJECT (or use as negative example)
- 2 inner contours: double-loop inductor (MR candidate) → ACCEPT
- 3+ inner contours: transformer, complex symbol → REJECT

This is an extremely powerful **topology-based filter** that directly exploits the mathematical structure of the circuit symbol.

---

# 7. Synthetic Template Augmentation

## 7.1 Motivation: The One-Shot Problem

We have exactly one template image `T`. This creates a fundamental problem for any statistical method:

- **PCA requires multiple samples** to estimate a meaningful covariance matrix
- **Any parametric distribution model** requires more than one sample
- **Robustness estimation** requires variation in the training examples

Synthetic augmentation solves this by generating a **pseudo-dataset** `{T_1, T_2, ..., T_N}` that approximates the distribution of appearances the symbol would have across different instances.

The augmentation must be **physically motivated** — we augment only along axes of variation that genuinely occur in practice:
- Scanner skew → small rotations
- Different scale conventions → scale changes
- Ink spread / scan resolution → thickness changes
- Drawing imprecision → small elastic deformations

We must NOT augment along axes that would create physically impossible instances:
- Large rotations (90°+) only if the symbol genuinely appears rotated in the target diagram
- Mirror flips only if the symbol is symmetric (MR symbol IS left-right symmetric, so horizontal flip is valid)
- Color changes (irrelevant for binary images)

## 7.2 Rotation Augmentation

For rotation by angle θ, the transformation matrix is:

```
R(θ) = [cos θ  -sin θ] 
       [sin θ   cos θ]
```

Applied to image coordinates (with center-based rotation):

```
[x']   [cos θ  -sin θ] · [x - cx]   [cx]
[y'] = [sin θ   cos θ]   [y - cy] + [cy]
```

**Parameter range:** θ ∈ {-10°, -7°, -5°, -3°, -2°, -1°, 0°, 1°, 2°, 3°, 5°, 7°, 10°}

**Rationale:** Scanner skew is typically ≤ 5°. Larger rotations (90°, 270°) should be included only if the diagram contains horizontal-axis MR symbols. In our example diagram, all MR symbols appear upright, so θ_max = 10° is sufficient.

**Interpolation:** Use bilinear interpolation for subpixel accuracy. After rotation, re-binarize with the original threshold.

**Post-rotation processing:** Apply skeletonization again to normalize thickness artifacts introduced by rotation interpolation.

## 7.3 Scale Augmentation

Scale by factor s using the transformation:

```
S(s) = [s  0]
       [0  s]
```

**Parameter range:** s ∈ {0.75, 0.80, 0.85, 0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25}

All scaled templates are resized to a canonical size `(h_0, w_0)` (the original template size) before being stored in the augmentation pool. This normalization is critical — PCA requires all input vectors to have the same dimension.

**Resize interpolation:** For downscaling, use INTER_AREA (anti-aliasing). For upscaling, use INTER_CUBIC. Post-resize binarize.

**Non-uniform scaling:** Also include mild anisotropic scaling `S(s_x, s_y)` with s_x ≠ s_y to model aspect ratio variation:

```
s_x ∈ {0.9, 1.0, 1.1},  s_y ∈ {0.9, 1.0, 1.1}
```

Excluding the s_x = s_y cases (already covered), this adds 8 more variants.

## 7.4 Translation Jitter

Apply random sub-pixel translations `(t_x, t_y)` where `t_x, t_y ∈ [-3, 3]` pixels. This models the localization imprecision of the region proposal stage and makes the PCA model robust to small bounding box alignment errors.

## 7.5 Line Thickness Variation

Real circuit diagrams can have line thicknesses varying from 0.5px to 3px depending on:
- Printer/plotter resolution
- CAD pen-width settings
- Scanner threshold

**Thinning:** Apply erosion with a small (2×2) SE to the binary template → simulates thin-line drawing

**Thickening:** Apply dilation with a (2×2) or (3×3) SE → simulates bold-line or over-inked scan

```
T_thin = T_bin ⊖ SE_2x2
T_thick = T_bin ⊕ SE_3x3
```

Include both variants in the augmentation pool.

## 7.6 Elastic Deformation (Optional)

For maximal robustness, apply smooth elastic deformations using a random displacement field:

```
Δx(x,y) = α · G(x,y; σ) * rand_x(x,y)
Δy(x,y) = α · G(x,y; σ) * rand_y(x,y)
```

where `rand_x, rand_y ~ U(-1,1)` and α controls deformation magnitude, σ controls smoothness.

**Constraint:** σ must be large enough that deformations are smooth (not creating unrealistic jagged edges). Recommended: σ = 4–8 pixels, α = 2–5 pixels displacement.

**Use with caution:** Aggressive elastic deformation can break the topology of inductor loops — the key distinguishing feature.

## 7.7 Augmentation Pool Summary

| Augmentation | Variants | Count |
|---|---|---|
| Rotation | -10° to +10° at 1–2° steps | 13 |
| Uniform scale | 0.75× to 1.25× at 0.05 steps | 11 |
| Anisotropic scale | 8 combinations | 8 |
| Thickness (thin/thick) | 2 | 2 |
| Translation jitter | 5 random | 5 |
| Elastic deformation | 10 random | 10 |
| Original | 1 | 1 |
| **Total** | | **~50** |

Combined with the cross-product of rotation × scale (the most important factors): `13 × 11 = 143` primary augmentations.

**Final pool size:** ~50–200 augmented templates. This is sufficient for stable PCA estimation (in practice, PCA is stable with N >> d where d is the intrinsic dimensionality of the appearance space, which for a symbol with limited variation is ~3–10).

---

# 8. Similarity Matching Methodologies

## 8.1 Normalized Cross Correlation (NCC)

### 8.1.1 Formulation

Given a candidate region patch `P ∈ ℝ^{h×w}` and a template `T ∈ ℝ^{h×w}` (same size after resize), NCC is defined as:

```
NCC(P, T) = Σ_{x,y} [(P(x,y) - μ_P) · (T(x,y) - μ_T)] 
            ─────────────────────────────────────────────
            √[Σ_{x,y}(P(x,y) - μ_P)²] · √[Σ_{x,y}(T(x,y) - μ_T)²]
```

where `μ_P = (1/hw) Σ P(x,y)` and `μ_T = (1/hw) Σ T(x,y)`.

**Range:** NCC ∈ [-1, +1]. NCC = 1 indicates perfect correlation, NCC = -1 indicates perfect anti-correlation, NCC = 0 indicates no linear relationship.

**Equivalence to cosine similarity:** NCC is the cosine similarity between the mean-subtracted (zero-centered) vectorizations of P and T.

### 8.1.2 Illumination Invariance

The mean subtraction in NCC makes it invariant to global additive intensity offset (illumination level). The normalization by standard deviations makes it invariant to global multiplicative scaling (contrast).

Formally: `NCC(P + c, T) = NCC(P, T)` for any constant c (additive invariance)  
And: `NCC(αP, T) = NCC(P, T)` for any scalar α > 0 (multiplicative invariance)

This is highly desirable for circuit diagrams where scan quality varies.

### 8.1.3 FFT Acceleration

Computing NCC for all positions in a sliding window scan can be accelerated using the Fast Fourier Transform via the convolution theorem:

```
For position (u,v): NCC(u,v) = Σ_{x,y} P(x+u,y+v) · T(x,y) / (σ_P · σ_T)

Using FFT: NCC_map = IFFT[FFT(I) · conj(FFT(T))] / normalization
```

This reduces the per-position cost from O(hw) to O(1) amortized (with O(HW log(HW)) preprocessing), making full-image NCC computation feasible.

**However:** For our region-proposal-based pipeline, we don't compute NCC at all image positions — only at ~80 candidate positions. FFT acceleration is unnecessary here; direct NCC computation per candidate is O(hw) = O(60×80) = O(4800) operations, entirely tractable.

### 8.1.4 Limitations for Circuit Diagrams

**Problem 1 — Binary image sparsity:** For binary images, the mean μ_P and μ_T depend heavily on the foreground density. Two patches with similar topology but different amounts of surrounding wire will have different means, potentially reducing NCC despite structural similarity.

**Problem 2 — Strict spatial correspondence:** NCC requires pixel-by-pixel correspondence. Small misalignments (1–2 pixels) significantly reduce NCC scores even when symbols are visually identical.

**Problem 3 — Not edge-based:** NCC operates on raw pixel intensities. For circuit diagrams, this means white background pixels dominate (95% of image is background), and the metric is pulled by background matching rather than foreground structure matching.

**Solution:** Apply NCC to edge maps (output of Canny or Sobel) rather than raw binary images. Edge NCC is far more discriminative for line-art images.

### 8.1.5 Complexity

Per candidate: O(h·w)  
For N candidates: O(N·h·w)  
With our expected N = 80, h = 80, w = 60: ~384,000 operations — negligible.

---

## 8.2 Chamfer Matching

### 8.2.1 Motivation

Chamfer matching is specifically designed for matching **edge images** under noise and geometric distortion. It was introduced by Barrow et al. (1977) and formalized by Borgefors (1988) for industrial shape matching. It is arguably the most appropriate method for circuit diagram symbol matching.

### 8.2.2 Distance Transform

The Distance Transform `DT(I_edge)` of a binary edge image assigns to each pixel (x,y) the distance to the nearest edge pixel:

```
DT(x,y) = min_{(x',y') ∈ E} d((x,y), (x',y'))
```

where E is the set of edge pixels and d is Euclidean distance.

**Chamfer distance approximation:** True Euclidean DT is O(HW) but requires a two-pass algorithm. The Chamfer approximation uses local distance weights:

```
3-4 Chamfer: neighbor weights {1,1} → diagonal {1,0} horizontal/vertical
             DT_approx ≈ 1/3 · DT_euclidean  (overestimates by ~5%)
```

**Modern implementation:** `cv2.distanceTransform(img, cv2.DIST_L2, 5)` computes exact Euclidean DT in O(HW).

### 8.2.3 Chamfer Score

Given:
- Template edge image `E_T` (binary, 1 at edge pixels)
- Candidate patch edge image `E_P` (binary, 1 at edge pixels)
- Distance transform of the candidate patch: `DT_P`

The Chamfer score is:

```
d_chamfer(T, P) = (1/|E_T|) · Σ_{(x,y) ∈ E_T} DT_P(x,y)
```

**Interpretation:** For each edge pixel in the template, find the nearest edge pixel in the candidate patch. Average these distances. If the shapes are perfectly aligned, all template edge pixels land exactly on patch edge pixels → score = 0.

**Key property — Robustness to deformation:** Because Chamfer uses the *nearest* edge pixel rather than the corresponding pixel, it tolerates:
- Small translations (up to DT gradient length)
- Small deformations (elastic distortion up to ~3–5 pixels)
- Missing edges (broken lines contribute non-zero but bounded cost)

This makes Chamfer significantly more robust than NCC for noisy scan images.

### 8.2.4 Bidirectional Chamfer

One-directional Chamfer (template → patch) is asymmetric: it can be fooled if the candidate patch has many spurious edge pixels (since only template edges are evaluated against the patch DT).

**Solution — Bidirectional Chamfer:**

```
d_bidir(T, P) = (1/2) · [d_chamfer(T,P) + d_chamfer(P,T)]
             = (1/2) · [(1/|E_T|) Σ DT_P(t) + (1/|E_P|) Σ DT_T(p)]
```

This penalizes both:
- Template edges without nearby candidate edges (forward term)
- Candidate edges without nearby template edges (backward term) — penalizes clutter

### 8.2.5 Why Chamfer Is Optimal for Circuit Diagrams

Circuit diagrams have three properties that make Chamfer ideally suited:

1. **All information is in edges:** Circuit symbols ARE edge images. There is no interior texture. Chamfer operates natively on this representation.

2. **Topological deformation occurs in practice:** Scanner noise introduces small breaks and jitters in lines. Chamfer's nearest-edge matching handles this gracefully.

3. **Clutter rejection via backward term:** The bus wire creates spurious edges near the symbol. The backward Chamfer term assigns high cost to these unmatched wire edges, naturally penalizing regions that contain excessive clutter.

### 8.2.6 Complexity

Computing DT: O(H·W) per image  
For N candidates: O(N·h·w) for distance transform + score computation  
Expected: O(80 × 4800) = O(384,000) — negligible

---

## 8.3 Hausdorff Distance

### 8.3.1 Definition

The Hausdorff distance between two sets A and B is:

```
d_H(A,B) = max{h(A,B), h(B,A)}
```

where the directed Hausdorff distance is:

```
h(A,B) = max_{a ∈ A} min_{b ∈ B} d(a,b)
```

**Interpretation:** The Hausdorff distance is the largest minimum distance — the worst-case nearest-neighbor distance.

### 8.3.2 Comparison to Chamfer

| Property | Chamfer (avg) | Hausdorff (max) |
|---|---|---|
| Outlier sensitivity | Low (averages) | HIGH (takes max) |
| Robustness to broken edges | Good | Poor |
| Computation | O(h·w) | O(h·w) |
| Use case | Noisy, partial matches | Clean, complete shapes |

**Verdict for circuit diagrams:** Hausdorff is too sensitive to outliers. A single broken edge pixel in a scan creates a high Hausdorff distance even if the global shape matches well. **Chamfer is preferred.**

**Modified Hausdorff (Huttenlocher 1993):** Uses the average of a fraction of the largest distances instead of the maximum:

```
d_MH(A,B) = (1/|A|) Σ_{a ∈ A} min_{b ∈ B} d(a,b) = d_chamfer(A,B)
```

The modified Hausdorff IS the Chamfer distance. This unifies the two formulations.

---

## 8.4 Structural Shape Matching

### 8.4.1 Hu Moments

Moment-based shape descriptors are invariant to translation, scale, and rotation. The (p+q)-th order central moment is:

```
μ_pq = Σ_x Σ_y (x - x̄)^p (y - ȳ)^q · I(x,y)
```

Normalized moments:
```
η_pq = μ_pq / μ_00^(1 + (p+q)/2)
```

Hu's 7 invariant moments `{φ_1, ..., φ_7}` are nonlinear combinations of η_pq that are invariant to rotation, scale, and translation.

**Limitation for circuit diagrams:** Hu moments capture global shape statistics. Two very different shapes can have similar Hu moments (the moments are not injective). For MR vs. single-inductor discrimination, Hu moments alone may be insufficient.

**When useful:** As a pre-filter. If `|φ_i(candidate) - φ_i(template)|` exceeds a threshold for multiple i, reject the candidate early.

### 8.4.2 Fourier Descriptors

The boundary of a shape can be parameterized as a complex function `z(t) = x(t) + iy(t)`. Its DFT coefficients `Z(k) = Σ z(t) e^{-2πikt/N}` are the Fourier descriptors.

**Invariances:**
- Translation: |Z(k)| = same (starting point removed by |·|)
- Scale: Normalize by |Z(1)|
- Rotation: |Z(k)|/|Z(1)| is rotation-invariant
- Starting point: |Z(k)| is starting-point invariant

**Advantage:** Fourier descriptors capture shape at multiple frequency levels. Low-frequency coefficients capture global shape; high-frequency coefficients capture fine details.

**For MR symbol:** The distinctive two-hump boundary will create specific Fourier descriptor patterns at intermediate frequencies.

**Computational cost:** O(N log N) per contour, where N is contour length.

### 8.4.3 Shape Context

Shape context (Belongie et al. 2002) is a more discriminative descriptor:

For each boundary point `p_i`, compute a histogram of the log-polar distribution of all other boundary points:

```
h_i(k) = #{p_j ≠ p_i : (p_j - p_i) ∈ bin_k}
```

**Matching cost:** Bipartite matching between shape context histograms using the Chi-squared distance:

```
C(p_i, q_j) = (1/2) Σ_k (h_i(k) - h_j(k))² / (h_i(k) + h_j(k))
```

**Strength:** Highly discriminative, captures local context around each boundary point. Robust to small deformations.

**Weakness:** Expensive — O(N²) for matching. Not suitable for real-time applications, but acceptable for the small candidate sets we generate.

---

# 9. PCA-Based One-Shot Learning Refinement

## 9.1 Mathematical Foundation of PCA

### 9.1.1 The Eigenspace Approach

PCA constructs a linear subspace that maximally captures variance in a dataset. In our case, the "dataset" is the augmented template set `{T_1, ..., T_N}` vectorized into `ℝ^d` where `d = h·w`.

**Step 1: Vectorization**
```
t_i = vec(T_i) ∈ ℝ^d    for i = 1,...,N
```

**Step 2: Mean computation**
```
μ = (1/N) Σ_{i=1}^{N} t_i ∈ ℝ^d
```

**Step 3: Center the data**
```
x_i = t_i - μ    for i = 1,...,N
```

**Step 4: Covariance matrix**
```
C = (1/(N-1)) Σ_{i=1}^{N} x_i x_i^T = X^T X / (N-1) ∈ ℝ^{d×d}
```

where `X = [x_1 | x_2 | ... | x_N]^T ∈ ℝ^{N×d}`.

**Note on dimensionality:** For `d = 60×80 = 4800` and `N = 100`, the covariance matrix `C ∈ ℝ^{4800×4800}` would require `4800² × 8 bytes ≈ 184 MB` to store — impractical for naive computation.

### 9.1.2 Compact SVD for High-Dimensional PCA

When N << d (far fewer samples than dimensions — our case), use the compact SVD approach:

**Thin SVD of X:**
```
X = U Σ V^T
```

where:
- `U ∈ ℝ^{N×N}` — left singular vectors
- `Σ ∈ ℝ^{N×N}` diagonal — singular values
- `V ∈ ℝ^{d×N}` — right singular vectors (eigenvectors of C)

The eigenvalues of C are `λ_i = σ_i² / (N-1)` where `σ_i` are the singular values.

**Computational cost:** Thin SVD of `X ∈ ℝ^{N×d}`: O(N²d) when N << d. For N=100, d=4800: O(10⁶ × 4800) = O(4.8×10⁸) — acceptable.

### 9.1.3 Eigenspace Projection

Retain the top-K eigenvectors `V_K = [v_1 | v_2 | ... | v_K] ∈ ℝ^{d×K}` corresponding to the K largest eigenvalues.

**Projection of a test patch `p ∈ ℝ^d` onto eigenspace:**
```
z = V_K^T (p - μ) ∈ ℝ^K
```

**Reconstruction in original space:**
```
p̂ = V_K z + μ = V_K V_K^T (p - μ) + μ
```

**Reconstruction error:**
```
e(p) = ||p - p̂||² = ||(I - V_K V_K^T)(p - μ)||²
```

### 9.1.4 Variance Explained

The fraction of variance explained by K components:

```
R²(K) = Σ_{i=1}^{K} λ_i / Σ_{i=1}^{N} λ_i
```

**Choosing K:** For the MR symbol augmentation set, the intrinsic variance is low (most variation is due to rotation and scale, which are low-dimensional transformations). Empirically:
- K=5 captures ~70–80% of variance (most rotation and scale variation)
- K=10 captures ~90% of variance (including thickness and noise)
- K=20 may capture 95%+ but begins to overfit to augmentation artifacts

**Rule of thumb:** K = min(N/5, 20) for stable PCA estimation.

## 9.2 PCA as Refinement, Not Primary Detector

**Why PCA cannot be the primary detector:**

1. **Rotation sensitivity:** PCA eigenspace learned from templates at θ=0° with minor jitter will not capture instances at θ=45°. The projection error for a valid symbol at an unexpected rotation will be HIGH — incorrectly suggesting it is NOT a match.

2. **Scale dependency:** Patches must be resized to canonical size before PCA projection. If resizing introduces distortion (non-integer scale factors), error is introduced.

3. **False negative risk:** PCA reconstruction error is low for ANYTHING that resembles the training distribution — not just the target symbol. A visually similar but different symbol (single-loop inductor) that was included in the augmentation set would have low error.

**Why PCA IS useful as a refinement step:**

1. **Compact representation:** A K=10 dimensional projection captures the essential appearance. Distance in eigenspace is a coarse but useful similarity measure.

2. **Soft verification:** After Chamfer matching produces top-K candidates, PCA provides an independent similarity signal. Candidates that fail both Chamfer AND PCA are rejected; candidates that pass at least one are retained (recall-favoring logic).

3. **Scoring combination:** The combined score integrates multiple independent signals, reducing the probability that noise artifacts fool the overall system.

## 9.3 Combined Scoring

For each candidate patch `p_i`, compute:

```
score_combined(p_i) = α · NCC(p_i, T) + β · exp(-d_chamfer(p_i, T)/σ_c) + γ · exp(-e_PCA(p_i)/σ_e)
```

where:
- α, β, γ are weighting coefficients (tuned empirically; recommend β = 0.5, γ = 0.3, α = 0.2)
- σ_c, σ_e are normalization constants for Chamfer and PCA scores respectively

**Rationale for weighting:** Chamfer matching is the most reliable signal for edge-dominant images (β highest). PCA provides complementary structural verification (γ second). NCC is least reliable for binary sparse images (α smallest).

## 9.4 Euclidean vs Cosine Similarity in Eigenspace

For two projected vectors `z_1, z_2 ∈ ℝ^K`:

**Euclidean distance:**
```
d_E(z_1, z_2) = ||z_1 - z_2||_2
```

**Cosine similarity:**
```
sim_cos(z_1, z_2) = z_1^T z_2 / (||z_1|| · ||z_2||)
```

**For PCA-whitened spaces:** Cosine similarity is preferred because the eigenspace basis is orthonormal but eigenvalues have different magnitudes. After PCA whitening (dividing by σ_i), Euclidean distance is appropriate.

## 9.5 Strengths, Weaknesses, and Failure Cases

| Aspect | Assessment |
|---|---|
| **Strength: Low-data** | Only requires N=50–200 synthetic samples |
| **Strength: Speed** | O(d·K) projection per candidate |
| **Strength: Interpretability** | Eigenspace can be visualized to verify correctness |
| **Weakness: Rotation** | Sensitive to rotations not covered by augmentation |
| **Weakness: N dependency** | With very few samples, eigenspace is unstable |
| **Failure: PCA degeneracy** | If N < K, eigenspace is rank-deficient; eigenvalues are zero-padded |
| **Failure: Augmentation bias** | If augmentation doesn't cover real variation, PCA misses it |

## 9.6 Computational Complexity Summary

| Step | Cost |
|---|---|
| Build augmentation set | O(N·d) ≈ O(100 × 4800) = O(480,000) |
| Compute SVD of X | O(N²d) ≈ O(4.8×10⁸) — one-time setup |
| Project one candidate | O(d·K) ≈ O(4800 × 10) = O(48,000) |
| Compute reconstruction error | O(d·K) = O(48,000) |
| For 80 candidates | O(80 × 48,000) = O(3.84×10⁶) |

Total PCA stage cost: **negligible** compared to image I/O and preprocessing.

---

# 10. Graph-Based Structural Verification

## 10.1 Motivation

After similarity matching and PCA refinement, a small number of candidates remain (typically 5–20). For these candidates, we can apply a more expensive but highly discriminative structural verification step: graph-theoretic topology comparison.

## 10.2 Skeletonization and Graph Extraction

**Step 1: Skeletonize** the candidate binary patch → 1px-wide medial axis (already described in Section 5.6)

**Step 2: Extract junction points** — skeleton pixels with ≥3 neighbors

**Step 3: Extract endpoint pixels** — skeleton pixels with exactly 1 neighbor

**Step 4: Extract branch segments** — paths between junctions and/or endpoints

**Step 5: Build graph** `G = (V, E)`:
- V = junction points ∪ endpoint points
- E = branch segments (with attributes: length, curvature)

For the MR symbol, the expected graph structure is:
- 2 loop structures (each contributing ~4 junction points at arc entry/exit)
- 1 vertical connector segment
- Terminal marker (+ or dot) above upper loop

**Formally:**
```
V = {top_terminal, loop1_left, loop1_right, loop1_apex, loop2_left, loop2_right, loop2_apex}
|V| ≈ 7-9 nodes
|E| ≈ 8-10 edges
```

## 10.3 Graph Similarity Metrics

### 10.3.1 Node and Edge Count Matching

First-order filter: reject candidates whose graph has `|V| < 4` or `|V| > 15`, or `|E| < 4` or `|E| > 15`.

### 10.3.2 Cycle Count

The number of independent cycles in a graph is the **cyclomatic number**:

```
μ = |E| - |V| + C
```

where C is the number of connected components.

For MR symbol (two loops): `μ = 2`
For single inductor (one loop): `μ = 1`
For wire junction (no loops): `μ = 0`
For transformer (many loops): `μ ≥ 3`

**This single number perfectly discriminates MR from other common symbols!**

### 10.3.3 Subgraph Isomorphism (Full Structural Check)

For a complete structural verification, check if the template graph `G_T` is isomorphic to the candidate graph `G_C`.

**Graph isomorphism:** Two graphs G_T = (V_T, E_T) and G_C = (V_C, E_C) are isomorphic if there exists a bijection `f: V_T → V_C` such that:

```
(u,v) ∈ E_T ↔ (f(u), f(v)) ∈ E_C
```

**Complexity:** General graph isomorphism is in NP (no polynomial algorithm known for general graphs), but for **small, sparse, bounded-degree graphs** (|V| ≤ 20), practical algorithms (Ullmann's algorithm, VF2) solve it in milliseconds.

**VF2 algorithm:** State-space search with pruning using:
- Degree compatibility check
- Neighbor connectivity check
- Branch and bound with feasibility conditions

For our small symbol graphs (|V| ≤ 15), VF2 runs in <1ms per candidate pair.

## 10.4 Why Graph Verification Matters for the MR Symbol

The most challenging false positive scenario is **visual similarity between nearby symbols**. In our example:

- Inductor symbols (single loops) appear near the MR (double loop)
- The + terminal marker on MR makes it visually distinct but may be ambiguous when occluded

Graph verification resolves this: single inductor has `μ = 1`, MR has `μ = 2`. The cyclomatic number alone would cleanly discriminate.

---

# 11. Siamese Network Extension (Research Extension)

> **IMPORTANT CAVEAT:** This section describes a research extension that becomes relevant only when more data becomes available. For the current problem (one template, no labeled data), the classical pipeline in Sections 5–10 is the primary solution. Siamese networks should NOT be deployed without at minimum 50+ positive examples per symbol class.

## 11.1 Metric Learning Foundations

A Siamese network learns an embedding function `f_θ: ℝ^{h×w} → ℝ^K` such that:

```
d(f_θ(x_1), f_θ(x_2)) is small when x_1 and x_2 are same-class
d(f_θ(x_1), f_θ(x_2)) is large when x_1 and x_2 are different-class
```

This is **metric learning**: the network learns a task-specific distance metric rather than a class boundary.

## 11.2 Contrastive Loss

Given pairs (x_1, x_2) with label y = 1 if same-class, y = 0 if different-class:

```
L_contrastive = y · d² + (1-y) · max(margin - d, 0)²
```

where `d = ||f_θ(x_1) - f_θ(x_2)||₂` and margin > 0 (typically 1.0–2.0).

## 11.3 Triplet Loss

Given triplets (anchor, positive, negative):

```
L_triplet = max(d(a, p) - d(a, n) + margin, 0)
```

Triplet loss is more data-efficient than contrastive loss because each triplet creates a relative constraint rather than an absolute distance constraint.

## 11.4 Data Requirements

For Siamese networks to outperform our classical pipeline:
- Minimum: 50–100 positive examples (cropped MR symbols from various diagrams)
- Recommended: 500+ positive examples, 1000+ negative examples (other symbols, wire segments)
- Training time: 1–5 hours on GPU for a small CNN backbone

## 11.5 When to Use the Siamese Extension

| Condition | Recommendation |
|---|---|
| Single diagram, one template | Use classical pipeline (Sections 5–10) |
| Multiple diagrams of same type | Start building labeled dataset; Siamese becomes viable with 50+ examples |
| Enterprise deployment across many diagram types | Invest in Siamese training with proper dataset |
| Real-time performance required (<100ms) | Use optimized Chamfer + NMS; Siamese adds latency |

---

# 12. Non-Maximum Suppression

## 12.1 The Duplicate Detection Problem

The region proposal stage may generate overlapping candidate bounding boxes for the same symbol (from slight differences in connected component thresholding, overlapping grid windows if used, or multi-scale proposals). NMS resolves these into a single detection per symbol.

## 12.2 IoU Definition

The Intersection-over-Union between two bounding boxes A and B:

```
IoU(A, B) = Area(A ∩ B) / Area(A ∪ B)
```

Computed as:

```
x_inter_min = max(A.x, B.x)
y_inter_min = max(A.y, B.y)
x_inter_max = min(A.x+A.w, B.x+B.w)
y_inter_max = min(A.y+A.h, B.y+B.h)

w_inter = max(0, x_inter_max - x_inter_min)
h_inter = max(0, y_inter_max - y_inter_min)

Area_inter = w_inter * h_inter
Area_A = A.w * A.h
Area_B = B.w * B.h

IoU = Area_inter / (Area_A + Area_B - Area_inter)
```

**Range:** IoU ∈ [0, 1]. IoU = 0: no overlap. IoU = 1: identical boxes.

## 12.3 Greedy NMS Algorithm

```
Input: D = [(box_i, score_i)] sorted by score (descending)
Output: D_final = subset of D

D_final = []
while D is not empty:
    best = D.pop(0)           # Take highest-scoring detection
    D_final.append(best)
    D = [d for d in D if IoU(best.box, d.box) < θ_nms]
    # Remove all detections that heavily overlap with 'best'
```

**Complexity:** O(N²) in the worst case, but with typical N = 20–50 candidate detections: negligible.

## 12.4 NMS Threshold for Circuit Diagrams

**Key consideration:** Adjacent MR symbols in the diagram are closely spaced. Their bounding boxes may have non-negligible IoU (e.g., 0.1–0.2) even when they represent genuinely different symbols.

**Recommendation:** θ_nms = 0.3 (standard for object detection)  
**For dense circuit diagrams:** θ_nms = 0.4–0.5 to prevent over-suppression

**Soft-NMS (alternative):** Instead of hard suppression, decay scores by IoU:

```
score_j = score_j · exp(-IoU(best, j)² / σ)
```

This prevents complete removal of valid nearby detections while still reducing duplicate scores.

---

# 13. Evaluation Methodology

## 13.1 Core Metrics

### 13.1.1 Precision and Recall

Given:
- TP = True Positives (detected symbols that match ground truth, IoU > threshold)
- FP = False Positives (detected symbols with no matching ground truth)
- FN = False Negatives (ground truth symbols with no matching detection)

```
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1 = 2 · Precision · Recall / (Precision + Recall)
```

**Match criterion:** A detection is a TP if `IoU(detection, ground_truth) ≥ 0.5` (standard), or `IoU ≥ 0.25` for less strict evaluation appropriate for imperfect bounding boxes.

### 13.1.2 Average Precision (AP)

Plot the Precision-Recall curve by varying the confidence threshold. AP is the area under this curve:

```
AP = Σ_{k=1}^{N} Precision(k) · [Recall(k) - Recall(k-1)]
```

### 13.1.3 Localization Accuracy

Beyond detection, evaluate bounding box quality:

```
mIoU = (1/|TP|) Σ_{tp ∈ TP} IoU(tp.box, gt.box)
```

For engineering applications, we also care about centroid accuracy:

```
d_center = ||center(detected) - center(ground_truth)||_2  (in pixels)
```

## 13.2 Evaluation Without Ground Truth Labels

Since we are working in the low-data regime, formal evaluation may require manual annotation of the few available diagrams. For the example diagram, this requires marking the 7 MR instances — a 10-minute task.

**Protocol:**
1. Manually annotate one circuit diagram (7 MR bounding boxes)
2. Run pipeline, collect detections
3. Compute TP/FP/FN using IoU matching
4. Report F1, AP, mIoU

## 13.3 Robustness Testing Protocol

| Test | Modification | Expected Performance Drop |
|---|---|---|
| Baseline | No modification | Reference |
| +Gaussian noise σ=5 | Add N(0,25) noise to I_gray | <5% F1 drop with preprocessing |
| +Rotation ±5° | Rotate full diagram | <10% F1 drop with rotation augmentation |
| +Scale ×1.2 | Resize diagram to 120% | <5% F1 drop with scale augmentation |
| +JPEG Q=50 | Re-encode at 50% quality | <15% F1 drop |
| Partial occlusion | Mask 25% of each symbol | ~20% F1 drop (expected, acceptable) |
| Template at ±2° skew | Slightly rotate template | <3% F1 drop |

## 13.4 Synthetic Evaluation via Template Injection

For quantitative evaluation without real diagrams:

1. Take the template T
2. Render synthetic occurrences at known positions in a blank canvas
3. Add wire background, other symbols, and Gaussian noise
4. Run pipeline on synthetic canvas
5. Evaluate with exact ground truth

This creates a **controlled evaluation environment** where difficulty can be parametrically varied.

---

# 14. Baseline Systems

## Baseline 1: Raw Template Matching (cv2.matchTemplate with TM_CCORR_NORMED)

**Methodology:**
- No preprocessing beyond grayscale conversion
- Full sliding window over entire image
- Threshold at T_match = 0.8

**Complexity:** O(H·W·h·w) direct, or O(H·W·log(H·W)) with FFT

**Robustness:**
- Sensitive to line thickness variation ✗
- Sensitive to noise ✗
- Scale-invariant: NO (single scale only) ✗
- Works at 0° rotation ✓

**Expected failure modes:**
- Many false positives from partial wire matches
- Fails on scaled instances
- High computational cost

**Expected F1:** ~0.30–0.50 on real scan data

---

## Baseline 2: NCC-Based Matching with Multi-Scale

**Methodology:**
- Adaptive binarization
- Compute NCC map using cv2.matchTemplate(TM_CCOEFF_NORMED)
- At scales: {0.8, 0.9, 1.0, 1.1, 1.2}
- NMS at IoU = 0.3

**Complexity:** O(S·H·W·h·w) per scale count S

**Robustness:**
- Scale-invariant: YES (partially) ✓
- Rotation-invariant: NO ✗
- Noise-robust: MODERATE ✓

**Expected failure modes:**
- Background matching artifacts (sparse binary image)
- Adjacent symbol confusion

**Expected F1:** ~0.50–0.70

---

## Baseline 3: Chamfer Matching

**Methodology:**
- Canny edge detection on both diagram and template
- Compute distance transform of diagram edge map
- Slide template edge over distance transform map, sum DT values at template edge positions
- Threshold on average DT score

**Complexity:** O(H·W) for DT + O(H·W·|E_T|) for sliding match

**Robustness:**
- Noise-robust: HIGH ✓
- Edge deformation robust: HIGH ✓
- Scale-invariant: Partial (with multi-scale variant) ✓
- Rotation-invariant: Partial ✓

**Expected failure modes:**
- Wire clutter inflates false positive scores
- Very thin or broken edges reduce template edge density

**Expected F1:** ~0.70–0.85

---

## Baseline 4: PCA Subspace Matching (with Augmentation)

**Methodology:**
- Build 100-image augmentation set
- Fit PCA (K=10 components)
- For each sliding window position, compute reconstruction error
- Threshold on reconstruction error

**Complexity:** O(H·W·d·K) for full image sliding projection — expensive without region proposal

**Robustness:**
- Captures appearance manifold ✓
- Sensitive to large rotations ✗
- Requires exact patch size ✗

**Expected F1:** ~0.55–0.75 (limited by rotation sensitivity and sliding window approximation)

---

# 15. Final Recommended Pipeline

## 15.1 Architecture Summary

The final recommended system combines the strengths of all approaches while avoiding their individual failure modes:

```
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 0: Template Preparation (One-Time Setup)                      │
│                                                                      │
│ T → Preprocess → Generate N=150 augmentations → Build PCA model    │
│                                                                      │
│ Cost: ~5 seconds. Run once per symbol type.                         │
└─────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 1: Image Preprocessing                                        │
│                                                                      │
│ I_raw → Grayscale → Median(3×3) → Adaptive Threshold(w=21,C=10)    │
│       → Wire Removal (horizontal morph opening, SE width = 50px)   │
│       → Canny Edge Map (σ=1.0, low=50, high=150)                   │
│       → I_bin, I_edges, I_no_wire                                  │
│                                                                      │
│ Cost: ~200ms for 3000×800 image                                     │
└─────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 2: Region Proposal                                            │
│                                                                      │
│ I_no_wire → Connected Components → Filter by:                       │
│   - Area: [A_min, A_max] = [0.5, 4.0] × template_area             │
│   - Aspect: [0.3, 2.5]                                              │
│   - Edge density: [0.15, 0.85]                                      │
│   - Inner contour count: [1, 4]                                     │
│   - Cyclomatic number: [1, 4]                                       │
│                                                                      │
│ → ~20–40 candidate regions R                                        │
│ Cost: ~50ms                                                         │
└─────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 3: Multi-Scale Chamfer + NCC Matching                         │
│                                                                      │
│ For each r ∈ R, for each scale s ∈ {0.85, 0.92, 1.0, 1.08, 1.15}: │
│   - Resize r to template size                                       │
│   - Compute d_chamfer_bidir(r, T)                                   │
│   - Compute NCC(r_edge, T_edge)                                     │
│   - score(r,s) = 0.6·exp(-d_chamfer/σ) + 0.4·NCC                  │
│   - max_score(r) = max_s score(r,s)                                 │
│                                                                      │
│ → Retain top-15 candidates by max_score                             │
│ Cost: ~100ms                                                        │
└─────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 4: PCA Refinement                                             │
│                                                                      │
│ For each of top-15 candidates:                                      │
│   - Resize to canonical template size                               │
│   - Vectorize → ℝ^d                                                 │
│   - Project onto PCA eigenspace                                     │
│   - Compute reconstruction error e_i                               │
│   - refined_score = 0.7·match_score + 0.3·exp(-e_i/σ_e)           │
│                                                                      │
│ Cost: ~20ms                                                         │
└─────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 5: Graph Topology Verification (Top-10 only)                  │
│                                                                      │
│ For each of top-10 refined candidates:                              │
│   - Skeletonize patch                                               │
│   - Extract graph G_candidate                                       │
│   - Check cyclomatic number == 2 (MR has 2 loops)                  │
│   - If μ ∈ {1,2,3}: pass (with confidence modifier)                │
│   - If μ = 0 or μ > 4: REJECT                                      │
│                                                                      │
│ Cost: ~30ms                                                         │
└─────────────────────────────────────────────────────────────────────┘
         ↓
┌─────────────────────────────────────────────────────────────────────┐
│ STAGE 6: NMS + Final Output                                         │
│                                                                      │
│ Apply Soft-NMS (σ=0.5, θ_nms=0.45)                                 │
│ Threshold final scores at score > 0.50                             │
│ Output: {(x_i, y_i, w_i, h_i, score_i)}                           │
│                                                                      │
│ Cost: ~5ms                                                          │
└─────────────────────────────────────────────────────────────────────┘
```

## 15.2 Key Parameter Table

| Parameter | Value | Rationale |
|---|---|---|
| Adaptive threshold window | 21 | ~3× line width at 96 DPI |
| Adaptive threshold C | 10 | Conservative; avoids breaking thin lines |
| Wire removal SE length | 50px | Longer than symbol width, shorter than bus wire |
| Canny low/high | 50/150 | Standard for moderate-noise images |
| Area filter range | [0.5, 4.0]×template | Generous for ±25% scale variance |
| Scale search range | [0.85, 1.15] | Covers typical within-diagram variation |
| Chamfer weight | 0.6 | Most reliable for edge images |
| NCC weight | 0.4 | Complementary to Chamfer |
| PCA components K | 10 | Balances expressivity and overfitting |
| PCA weight | 0.3 | Refinement, not primary |
| NMS threshold | 0.45 | Higher than standard to allow dense symbols |
| Final score threshold | 0.50 | Recall-favoring |

## 15.3 Expected Performance

| Metric | Expected Value |
|---|---|
| Recall on clean diagrams | 0.92–0.98 |
| Precision on clean diagrams | 0.85–0.95 |
| F1 | 0.88–0.96 |
| Processing time (3000×800 image) | ~400–600ms total |
| Memory footprint | <200 MB |

## 15.4 Why This Pipeline Is Optimal for the Given Constraints

1. **Zero training data requirement:** The pipeline is operational with exactly one template image
2. **Mathematically principled:** Every decision is grounded in geometry, topology, or linear algebra
3. **Explainable:** Each stage produces interpretable intermediate outputs — engineers can debug specific failure modes
4. **Computationally efficient:** Region proposal reduces the search space by 4+ orders of magnitude vs brute force
5. **Multi-signal integration:** Chamfer + NCC + PCA + topology provide redundant verification that reduces both false positives and false negatives
6. **Extensible:** Adding new symbol types requires only providing a new template — no retraining

---

# 16. Implementation Plan

## 16.1 Technology Stack

| Library | Version | Purpose |
|---|---|---|
| `opencv-python` | ≥4.8 | Core image processing, morphology, contour analysis, DT |
| `numpy` | ≥1.24 | Matrix operations, vectorization |
| `scikit-image` | ≥0.21 | Skeletonization (Zhang-Suen), Canny, regionprops |
| `scikit-learn` | ≥1.3 | PCA (TruncatedSVD for memory efficiency) |
| `scipy` | ≥1.11 | Distance transforms, spatial operations |
| `networkx` | ≥3.1 | Graph construction and cyclomatic number computation |
| `matplotlib` | ≥3.7 | Visualization, result overlay |

## 16.2 Directory Structure

```
circuit_symbol_localization/
│
├── src/
│   ├── __init__.py
│   ├── preprocessing.py          # Stage 1: All preprocessing operations
│   ├── region_proposal.py        # Stage 2: Connected components + filtering
│   ├── template_augmentation.py  # Stage 0: Augmentation + PCA setup
│   ├── similarity_matching.py    # Stage 3: NCC, Chamfer, Hausdorff
│   ├── pca_refinement.py         # Stage 4: PCA projection + scoring
│   ├── graph_verification.py     # Stage 5: Skeleton + topology check
│   ├── nms.py                    # Stage 6: Non-Maximum Suppression
│   └── pipeline.py               # Full pipeline orchestration
│
├── evaluation/
│   ├── annotate.py               # Manual annotation tool
│   ├── metrics.py                # Precision, Recall, F1, mIoU
│   └── stress_test.py            # Noise/rotation/scale stress tests
│
├── visualization/
│   └── overlay.py                # Draw bounding boxes on original image
│
├── tests/
│   ├── test_preprocessing.py
│   ├── test_matching.py
│   └── test_pipeline.py
│
├── data/
│   ├── templates/                # Symbol template images
│   └── diagrams/                 # Circuit diagram images
│
└── main.py                       # CLI entry point
```

## 16.3 Module Pseudocode

### preprocessing.py

```python
import cv2, numpy as np
from skimage.filters import threshold_local

def preprocess_diagram(image_path: str) -> dict:
    """
    Full preprocessing pipeline for circuit diagram.
    Returns dict with keys: gray, binary, edges, no_wire
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Denoising
    gray = cv2.medianBlur(gray, ksize=3)
    
    # Adaptive binarization
    binary = cv2.adaptiveThreshold(
        gray, 255, 
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY_INV, 
        blockSize=21, C=10
    )
    
    # Wire removal: horizontal morphological opening
    se_horizontal = cv2.getStructuringElement(
        cv2.MORPH_RECT, (50, 1)
    )
    wire_mask = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se_horizontal)
    no_wire = cv2.subtract(binary, wire_mask)
    
    # Edge map for Chamfer matching
    edges = cv2.Canny(gray, threshold1=50, threshold2=150, apertureSize=3)
    
    return {'gray': gray, 'binary': binary, 'edges': edges, 'no_wire': no_wire}


def preprocess_template(template_path: str) -> dict:
    """Preprocessing for template image."""
    img = cv2.imread(template_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 3)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    edges = cv2.Canny(gray, 50, 150)
    return {'gray': gray, 'binary': binary, 'edges': edges}
```

### region_proposal.py

```python
def propose_regions(no_wire_binary: np.ndarray, template_shape: tuple) -> list:
    """
    Extract candidate bounding boxes via connected component analysis.
    Filters by area, aspect ratio, edge density, and inner contour count.
    """
    h_t, w_t = template_shape
    template_area = h_t * w_t
    
    # Connected component analysis
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        no_wire_binary, connectivity=8
    )
    
    candidates = []
    
    for i in range(1, n_labels):  # Skip background (label 0)
        x, y, w, h, area = stats[i]
        
        # Area filter
        if not (0.4 * template_area < area < 5.0 * template_area):
            continue
        
        # Aspect ratio filter
        ar = w / (h + 1e-6)
        if not (0.25 < ar < 3.0):
            continue
        
        # Edge density filter
        density = area / (w * h + 1e-6)
        if not (0.10 < density < 0.90):
            continue
        
        # Inner contour count (topology filter)
        patch = no_wire_binary[y:y+h, x:x+w]
        contours, hierarchy = cv2.findContours(
            patch, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        if hierarchy is not None:
            inner_count = np.sum(hierarchy[0, :, 3] != -1)  # Has parent
            if inner_count < 1 or inner_count > 5:
                continue
        
        candidates.append({'bbox': (x, y, w, h), 'area': area, 'inner_count': inner_count})
    
    return candidates
```

### similarity_matching.py

```python
def chamfer_score(candidate_edges: np.ndarray, template_edges: np.ndarray) -> float:
    """
    Bidirectional Chamfer distance between two edge images.
    Lower score = better match.
    """
    # Resize candidate to template size
    h_t, w_t = template_edges.shape
    cand_resized = cv2.resize(candidate_edges, (w_t, h_t), interpolation=cv2.INTER_NEAREST)
    cand_binary = (cand_resized > 128).astype(np.uint8) * 255
    
    # Distance transforms
    dt_template = cv2.distanceTransform(cv2.bitwise_not(template_edges), cv2.DIST_L2, 5)
    dt_candidate = cv2.distanceTransform(cv2.bitwise_not(cand_binary), cv2.DIST_L2, 5)
    
    # Template -> Candidate direction
    t_edge_pixels = template_edges > 128
    if t_edge_pixels.sum() == 0:
        return float('inf')
    fwd_score = dt_candidate[t_edge_pixels].mean()
    
    # Candidate -> Template direction  
    c_edge_pixels = cand_binary > 128
    if c_edge_pixels.sum() == 0:
        return float('inf')
    bwd_score = dt_template[c_edge_pixels].mean()
    
    return 0.5 * (fwd_score + bwd_score)


def ncc_score(candidate_patch: np.ndarray, template_patch: np.ndarray) -> float:
    """
    Normalized Cross Correlation. Returns value in [-1, 1]; higher = better.
    """
    h_t, w_t = template_patch.shape
    cand_resized = cv2.resize(candidate_patch, (w_t, h_t)).astype(np.float32)
    tmpl = template_patch.astype(np.float32)
    
    cand_centered = cand_resized - cand_resized.mean()
    tmpl_centered = tmpl - tmpl.mean()
    
    numerator = (cand_centered * tmpl_centered).sum()
    denominator = (np.sqrt((cand_centered**2).sum()) * 
                   np.sqrt((tmpl_centered**2).sum()) + 1e-8)
    
    return float(numerator / denominator)
```

### pipeline.py (orchestration)

```python
class CircuitSymbolLocalizer:
    
    def __init__(self, template_path: str, n_augmentations: int = 150):
        # One-time setup
        self.template_data = preprocess_template(template_path)
        self.augmented_templates = generate_augmentations(
            self.template_data, n=n_augmentations
        )
        self.pca_model = build_pca_model(self.augmented_templates, k=10)
    
    def localize(self, diagram_path: str) -> list:
        # Stage 1
        diagram_data = preprocess_diagram(diagram_path)
        
        # Stage 2
        candidates = propose_regions(
            diagram_data['no_wire'], 
            self.template_data['binary'].shape
        )
        
        # Stage 3
        scored = []
        scales = [0.85, 0.92, 1.0, 1.08, 1.15]
        for cand in candidates:
            x, y, w, h = cand['bbox']
            patch_edges = diagram_data['edges'][y:y+h, x:x+w]
            
            best_score = 0
            for s in scales:
                scaled_t_edges = scale_template(self.template_data['edges'], s)
                c = chamfer_score(patch_edges, scaled_t_edges)
                n = ncc_score(patch_edges, scaled_t_edges)
                score = 0.6 * np.exp(-c/10.0) + 0.4 * max(n, 0)
                best_score = max(best_score, score)
            
            cand['match_score'] = best_score
        
        # Sort and take top 15
        scored = sorted(candidates, key=lambda x: x['match_score'], reverse=True)[:15]
        
        # Stage 4: PCA refinement
        for cand in scored:
            x, y, w, h = cand['bbox']
            patch = diagram_data['binary'][y:y+h, x:x+w]
            e = pca_reconstruction_error(patch, self.pca_model, 
                                          self.template_data['binary'].shape)
            cand['refined_score'] = (0.7 * cand['match_score'] + 
                                     0.3 * np.exp(-e / 1000.0))
        
        # Stage 5: Graph topology verification
        for cand in scored:
            x, y, w, h = cand['bbox']
            patch = diagram_data['binary'][y:y+h, x:x+w]
            mu = compute_cyclomatic_number(patch)
            cand['topology_ok'] = (1 <= mu <= 4)
            if not cand['topology_ok']:
                cand['refined_score'] *= 0.3  # Penalize but don't hard-reject
        
        # Filter by threshold
        final = [c for c in scored if c['refined_score'] > 0.35]
        
        # Stage 6: NMS
        detections = [(c['bbox'], c['refined_score']) for c in final]
        detections = soft_nms(detections, sigma=0.5)
        
        return detections
```

## 16.4 CLI Entry Point

```bash
# Usage
python main.py \
  --diagram path/to/circuit_diagram.png \
  --template path/to/mr_symbol.png \
  --output path/to/output_annotated.png \
  --threshold 0.40 \
  --visualize
```

## 16.5 Processing Flow Summary

```
main.py
  └─ CircuitSymbolLocalizer.__init__()
       ├─ preprocess_template()
       ├─ generate_augmentations()      [150 variants]
       └─ build_pca_model()             [K=10 components]

  └─ CircuitSymbolLocalizer.localize()
       ├─ preprocess_diagram()          [~200ms]
       ├─ propose_regions()             [~50ms]
       ├─ chamfer_ncc_matching()        [~100ms]
       ├─ pca_refinement()             [~20ms]
       ├─ graph_verification()          [~30ms]
       ├─ soft_nms()                   [~5ms]
       └─ return detections            [Total: ~405ms]
```

---

## Appendix A: Comparison of All Methods

| Method | Data Needed | Rotation | Scale | Noise | Speed | Explainability |
|---|---|---|---|---|---|---|
| Raw Template Match | 1 template | ✗ | ✗ | ✗ | Fast | High |
| NCC Multi-Scale | 1 template | ✗ | ✓ | Moderate | Moderate | High |
| Chamfer Matching | 1 template | Partial | Partial | ✓✓ | Fast | High |
| Hausdorff | 1 template | ✗ | ✗ | ✗ | Moderate | High |
| PCA Subspace | 50+ augmented | Partial | ✓ | ✓ | Moderate | Moderate |
| Graph Topology | 1 template | ✓ | ✓ | ✓ | Slow | Very High |
| **Hybrid (Proposed)** | **1 template** | **✓** | **✓** | **✓✓** | **Moderate** | **High** |
| Siamese Network | 500+ examples | ✓✓ | ✓✓ | ✓✓ | Fast (inference) | Low |

---

## Appendix B: Expected Intermediate Outputs for MR Symbol Detection

```
Input diagram: 3000×800 px (example image provided)

After preprocessing:
  - no_wire binary: ~95% of wire content removed
  - ~8-12 connected components remain (pure symbol regions)

After region proposal:
  - ~25-35 candidates (including some text and other symbols)
  - After topology filter: ~10-15 candidates

After Chamfer + NCC matching:
  - Top-15 candidates have scores in range [0.3, 0.9]
  - True MR instances score: 0.65–0.90
  - False candidates score: 0.30–0.55

After PCA refinement:
  - True MR refined scores: 0.70–0.92
  - False candidates: 0.25–0.45

After graph verification + NMS:
  - Expected 7 detections (matching 7 MR symbols in diagram)
  - Score distribution: 0.68–0.89
  - All with IoU > 0.6 vs ground truth

Final output: 7 bounding boxes with high confidence
```

---

*Document Version 1.0 — Research-Grade Engineering Design*  
*Prepared for: Circuit Diagram Symbol Localization System*  
*Classification: Technical Architecture + Implementation Reference*
