# Survey Requirements: Computer Vision for Construction Drawing Understanding

## Title (Working)

**"Computer Vision for Construction Drawing Understanding: A Comprehensive Survey"**

Alternative titles:
- "Deep Learning and Foundation Models for Construction Drawing Analysis: A Survey"
- "From Blueprints to Intelligence: A Survey of Computer Vision for Construction Documents"

## Target Venue

TPAMI (IF ~24) or IJCV (IF ~19) — top CV journals that accept surveys.

## Scope & Focus

### What this survey IS:
- A comprehensive survey of **Computer Vision methods applied to 2D construction drawing/document analysis**
- Covers: construction blueprints, architectural floor plans, structural drawings, MEP (mechanical/electrical/plumbing) drawings, P&IDs, shop drawings, detail drawings
- Organized by **CV methodology** (detection, segmentation, OCR, VLMs, graph reasoning, etc.) with construction drawings as the application domain
- Includes **quantitative statistics** about the field: paper counts by sub-area, trends over time, method popularity, benchmark comparisons
- Includes **figures and charts** that visually present the landscape of the field

### What this survey is NOT:
- Not a general "Visual Document Understanding" survey (that covers invoices, forms, receipts — not our focus)
- Not covering 3D/point cloud/scan-to-BIM (out of scope — 2D drawings only)
- Not covering construction site monitoring/safety/progress tracking (out of scope — documents only)
- Not a deep experimental paper (experiments are supplementary, statistical in nature)
- Not exposing Bobyard's proprietary methods

### Why Construction Drawing CV:
1. **Massive market**: Quantity Takeoff AI: $1.15B (2024) → $5.87B (2033), CAGR 18.4%
2. **No comprehensive survey exists**: Existing surveys cover only narrow slices (floor plans only, P&IDs only, general VRD only). No one has unified construction drawing analysis as a CV problem
3. **Directly relevant to Bobyard**: Author affiliation shows company; survey establishes thought leadership in exactly our domain
4. **Unique technical challenges**: High-resolution drawings (10K+ pixels), sparse line art vs. natural images, domain-specific symbols, multi-scale objects, few-shot scenarios, no pre-trained models for this domain
5. **Foundation model revolution**: SAM, DINO, VLMs are entering this space but no survey maps where they work and where they fail

## Content Structure (Draft)

### Part I: Foundations
1. **Introduction** — Why construction drawing CV matters, market context, survey contributions
2. **Background & Taxonomy** — Types of construction drawings, task definitions, propose a unified taxonomy of methods
3. **Statistical Overview of the Field** —
   - Paper count by year (trend chart, 2015-2026)
   - Distribution across drawing types (floor plans vs. structural vs. MEP vs. P&ID etc.)
   - Method popularity over time (CNN → Transformer → Foundation Model)
   - Venue distribution (where construction drawing papers get published)
   - Geographic distribution of research groups

### Part II: Core CV Tasks on Construction Drawings
4. **Symbol Detection & Recognition** — Detecting doors, windows, equipment, electrical symbols, plumbing fixtures, structural elements; YOLO family, DETR, Grounding DINO on drawings
5. **Layout Analysis & Segmentation** — Room segmentation, wall detection, zone classification; SAM adaptation for line drawings
6. **Text Detection & Recognition in Drawings** — Dimension text, labels, annotations, title blocks; OCR challenges unique to construction drawings (rotated text, dense annotations)
7. **Table & Schedule Extraction** — Door schedules, finish schedules, equipment lists from drawings
8. **Topology & Graph Extraction** — Connectivity analysis (P&ID flow, electrical circuits, plumbing networks); GNN-based methods, Relationformer

### Part III: Drawing Types (Application-Specific Analysis)
9. **Architectural Floor Plans** — Room detection, wall segmentation, furniture recognition; FloorplanCAD, ArchCAD-400K
10. **Structural Drawings** — Beam/column detection, rebar schedules, section views
11. **MEP Drawings** — Mechanical ductwork, electrical wiring, plumbing routing
12. **P&ID and Process Diagrams** — Symbol detection, line tracing, connectivity graphs
13. **Civil / Site Plans** — Grading, utilities, boundary detection

### Part IV: Cross-Cutting Techniques
14. **Foundation Model Adaptation for Drawings** — SAM/DINOv2/CLIP fine-tuning for non-natural-image domains; LoRA, adapters, domain-specific pre-training
15. **Vision-Language Models for Construction Documents** — GPT-4o/Qwen-VL/InternVL performance on drawing tasks; AECV-Bench, CEQuest, DesignQA benchmarks
16. **High-Resolution & Multi-Scale Processing** — Dynamic tiling, multi-scale detection (SAHI), efficient attention for 10K+ pixel images
17. **Few-Shot & Data-Efficient Methods** — Few-shot symbol detection, synthetic data generation, self-supervised pre-training for drawings

### Part V: Resources & Outlook
18. **Datasets & Benchmarks** — Unified comparison table of all public construction drawing datasets
19. **Industry Landscape** — Key companies (Togal.AI, Beam AI, Drawer AI, etc.), adoption statistics
20. **Open Challenges & Future Directions** — Specific, actionable research directions
21. **Conclusion**

## Experiments & Statistics Plan

### Type A: Bibliometric / Statistical Analysis (Primary — MUST DO)
- Collect papers from Semantic Scholar, arXiv, Google Scholar using targeted queries
- Categorize by: drawing type, CV method, year, venue
- Generate statistical charts:
  - **Timeline chart**: Papers per year in construction drawing CV (2015-2026)
  - **Taxonomy distribution**: Paper count per sub-area (bar chart)
  - **Method evolution**: CNN → Transformer → Foundation Model trends (stacked area chart)
  - **Benchmark comparison tables**: Cross-method performance on standard datasets (FloorplanCAD, P&ID datasets, etc.)
  - **Dataset statistics table**: Size, annotation type, availability, drawing type for each dataset

### Type B: Lightweight Benchmark Experiments (Secondary — OPTIONAL)
- Run representative models on public construction drawing datasets for fair comparison
- Focus on key tasks:
  - Symbol detection on FloorplanCAD
  - Symbol detection on P&ID datasets
  - Floor plan segmentation
- Use AWS g5.2xlarge spot instances (~$0.58/hr)
- Goal: Fill comparison gaps where no paper provides head-to-head results

### Type C: Qualitative Analysis (Supplementary)
- Visual examples showing model outputs on construction drawings
- Failure case analysis of foundation models on construction drawings
- Domain gap visualization (natural images vs. line drawings)

## Key Differentiators

1. **First unified survey** covering CV for all types of construction drawings
2. **Quantitative landscape analysis** — statistics and charts, not just literature listing
3. **Foundation model lens** — systematic analysis of how SAM/DINO/VLMs perform on construction drawings
4. **Practical industry perspective** — from a company (Bobyard) actually building construction CV products
5. **Cross-drawing-type benchmark comparison** — unified tables across floor plans, P&IDs, structural, MEP

## Reusable Resources from Previous Work

Located in `/Volumes/Storage/Server/Startup/AI-Scientist-v2 - old/`:
- `experiments/2026-03-29_01-18-12_.../papers_database.json` — 807 papers (partially relevant, need re-filtering for construction drawing focus)
- `survey-direction-research.md` — Extensive prior research (sections on floor plans, P&IDs, engineering diagrams directly relevant)
- `ai_scientist/blank_survey_latex/template.tex` — TPAMI LaTeX template (reusable)
- `aws/ec2_manager.py` — EC2 management code (reusable with fixes)

## Technical Setup

- **AI Scientist v2**: Fresh clone at `/Volumes/Storage/Server/Startup/AI-Scientist-v2/`, minimal modifications
- **LLM**: GPT-5.4 (OpenAI) + Claude (Anthropic)
- **Semantic Scholar API**: For paper collection (rate limit: 1 req/sec)
- **AWS**: us-west-2, g5.2xlarge spot instances, `zelin` key pair
- **Local**: Mac Mini M4, conda env `ai_survey`, Python 3.12

## Success Criteria

1. **Coverage**: 300+ papers surveyed, spanning all construction drawing sub-areas
2. **Statistics**: 5+ quantitative charts/figures showing field landscape
3. **Tables**: Unified benchmark comparison tables with actual numbers
4. **Quality**: Publishable at TPAMI/IJCV level — insightful analysis, not just listing papers
5. **Length**: 25-35 pages (TPAMI format)

---

*Requirements drafted: 2026-03-29*
*Author: Zelin (Bobyard)*
