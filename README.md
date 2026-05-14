# Single Circuit Symbol Identification

A research-oriented hybrid Computer Vision pipeline for one-shot localization of embedded circuit symbols inside complex engineering diagrams.

This project explores the limits of:
- classical computer vision
- symbolic topology reasoning
- PCA-based appearance manifolds
- Siamese metric learning

for detecting a target symbol from only a single low-quality reference template.

---

# Problem Statement

Given:
- A large, noisy, straight-line circuit diagram
- A single low-quality target symbol template

The objective is to localize all occurrences of the target symbol within the diagram.

The challenge is difficult because the symbols are:
- embedded within dense circuit structures
- partially connected to the surrounding geometry
- degraded by wires and text overlap
- and not fully separable as isolated connected components

---

# Key Research Goal

The project investigates:

> How far can local geometric, topological, and learned semantic reasoning go in one-shot symbolic localization before contextual circuit understanding becomes necessary?

---

# Final Experimental Conclusion

The strongest pipeline achieved was:

```text
Dense Chamfer Matching
→ Spatial NMS
→ PCA Appearance Refinement
