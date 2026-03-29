# Title: Computer Vision for Construction Drawing Understanding: A Comprehensive Survey

## Keywords
construction drawing, floor plan, P&ID, blueprint, structural drawing, MEP drawing, symbol detection, document analysis, foundation models, SAM, vision-language models, graph reasoning, benchmark, bibliometric analysis

## TL;DR
The first comprehensive survey covering computer vision methods for all types of construction drawings — floor plans, structural, MEP, P&ID, and civil — with unified taxonomy, bibliometric statistics, benchmark comparisons, and foundation model analysis.

## Abstract
The construction industry produces millions of 2D technical drawings annually — architectural floor plans, structural details, MEP (mechanical, electrical, plumbing) layouts, P&ID (piping and instrumentation) diagrams, and civil site plans. These documents encode critical project information in a visual format that differs fundamentally from natural images: sparse line art, domain-specific symbols, extreme resolutions (10,000+ pixels), dense text annotations, and complex topological relationships. Computer vision research on construction drawings has grown substantially, with deep learning methods for symbol detection (YOLO, DETR, Grounding DINO), semantic segmentation (U-Net, Mask R-CNN, SAM), text recognition (OCR adapted for rotated/dense annotations), and graph-based topology extraction (GNNs, Relationformer) showing promising results. Foundation models — SAM, DINOv2, CLIP — and vision-language models — GPT-4o, Qwen-VL, InternVL — are now entering this domain but their effectiveness on construction drawings versus natural images remains poorly understood.

Despite this growing body of work, the field is highly fragmented. Floor plan recognition, P&ID digitization, and structural drawing analysis communities operate in silos, publishing in different venues with little cross-pollination. Existing surveys cover only narrow slices: floor plan recognition (Dodge et al. 2017), P&ID processing (Lu et al. 2019), or general visual document understanding (Huang et al. 2022) which barely touches construction drawings. No survey provides a unified view of computer vision for construction documents as a whole.

This survey fills that gap. We target publication at IEEE TPAMI (Transactions on Pattern Analysis and Machine Intelligence). Our key contributions are:

1. **Unified Taxonomy**: A two-dimensional taxonomy organizing the field by drawing type (architectural floor plan, structural, MEP, P&ID, civil) and CV task (detection, segmentation, OCR, table extraction, graph/topology reasoning).

2. **Comprehensive Coverage**: 300+ papers spanning 2015–2026, covering all major drawing types and CV methodologies from classical image processing through CNNs, transformers, to foundation models.

3. **Quantitative Bibliometric Analysis**: Statistical charts showing publication trends, method evolution, venue distribution, citation analysis, and geographic research activity — providing a data-driven map of the field.

4. **Benchmark Comparison Tables**: Unified tables comparing method performance across public datasets (FloorplanCAD, CVC-FP, SESYD, DSSE-200, P&ID benchmarks) with standardized metrics.

5. **Foundation Model Assessment**: Systematic analysis of how SAM, Grounding DINO, CLIP, and vision-language models perform on construction drawings — where they succeed, where they fail, and what adaptations are needed.

6. **Open Challenges and Future Directions**: Concrete, actionable research directions including high-resolution processing, few-shot symbol detection, cross-drawing-type transfer learning, and the critical shortage of annotated construction drawing datasets.

## What the BFTS Agent Should Produce

The automated agent should produce the following concrete outputs:

### Code and Data Pipeline
- **Paper collection script**: Python script using the Semantic Scholar API (`requests` library, 1 req/sec rate limit) to search for papers using ~15 targeted query strings covering all construction drawing sub-areas. Output: `papers_database.json` containing structured metadata for 300+ papers.
- **Categorization script**: Script that reads the papers database and classifies each paper by drawing type and CV method using keyword matching on titles and abstracts. Output: categorized database with taxonomy labels.
- **Statistics generation script**: Script using `matplotlib` and `seaborn` to produce publication-quality charts (PDF format, suitable for TPAMI):
  - `fig_papers_per_year.pdf` — Bar chart of papers by year (2015–2026)
  - `fig_method_evolution.pdf` — Stacked area chart showing CNN → Transformer → Foundation Model trends
  - `fig_drawing_type_distribution.pdf` — Bar chart of papers per drawing type
  - `fig_venue_distribution.pdf` — Top 20 venues by paper count
  - `fig_citation_analysis.pdf` — Most cited papers per sub-area
- **Dataset table generator**: Script that outputs a LaTeX table of all public construction drawing datasets with columns: Name, Drawing Type, # Images, Annotation Type, # Classes, Resolution, Availability, Year.
- **Benchmark comparison table**: Script that compiles reported performance metrics (mAP, IoU, F1) from papers evaluated on the same datasets into a unified LaTeX comparison table.

### Key Search Queries for Paper Collection
The agent should use these Semantic Scholar queries (and variations):
- `"floor plan recognition"`, `"floor plan segmentation"`, `"room segmentation floor plan"`
- `"architectural drawing" AND ("detection" OR "recognition" OR "segmentation")`
- `"P&ID" AND ("symbol detection" OR "digitization" OR "recognition")`
- `"piping and instrumentation diagram" AND "deep learning"`
- `"construction drawing" AND ("computer vision" OR "deep learning" OR "object detection")`
- `"engineering drawing" AND ("symbol recognition" OR "text detection" OR "understanding")`
- `"blueprint" AND ("analysis" OR "understanding" OR "AI")`
- `"MEP drawing" AND ("detection" OR "recognition")`
- `"structural drawing" AND ("detection" OR "recognition" OR "deep learning")`
- `"building plan" AND ("segmentation" OR "detection" OR "parsing")`
- `"technical drawing" AND ("object detection" OR "symbol recognition")`
- `"construction document" AND ("AI" OR "machine learning" OR "computer vision")`
- `"CAD drawing" AND ("recognition" OR "understanding" OR "deep learning")`
- `"SAM" AND ("floor plan" OR "construction drawing" OR "engineering drawing")`
- `"vision language model" AND ("construction" OR "architectural" OR "engineering drawing")`

### Known Public Datasets to Include
- **FloorplanCAD** (2021) — 11,000+ floor plans with vectorized annotations
- **CVC-FP** (2014) — 122 floor plan images with wall/room annotations
- **ROBIN** (2021) — Floor plan dataset with room boundary annotations
- **SESYD** (2013) — Synthetic electrical/architectural symbol dataset
- **R2V / Raster-to-Vector** (2017) — Floor plan vectorization dataset
- **DSSE-200** (2019) — Document structure and symbol extraction
- **BRIDGE** (2023) — P&ID symbol detection benchmark
- **ArchCAD-400K** (2024) — Large-scale architectural CAD dataset
- **AECV-Bench** (2024) — AEC visual understanding benchmark for VLMs
- **CEQuest** (2025) — Construction engineering question answering
- **DesignQA** (2024) — Design document visual QA benchmark
