# Multimodal Cross-Retrieval using Sheaf-Theoretic Laplacian

This repository contains code and experiments for a **multimodal cross-retrieval task** using a **sheaf-theoretic Laplacian framework**. The goal is to merge visual and textual embeddings into a shared semantic space to improve cross-modal retrieval (image-to-text and text-to-image).

## Experiment Overview

- **Dataset:** [COCO](https://cocodataset.org/) (each image has 5 captions).  
- **Models Used:** 
  - Vision: `facebook/dinov2-small`  
  - Text: `distilbert-base-uncased`  
- **Framework:**  
  - Visual and textual embeddings are projected via **linear restriction maps** ($P_{12}$, $P_{21}$) into a **shared 128-dimensional space**.  
  - Sheaf Laplacian enforces **local consistency** between modalities.  
  - Downstream task: **multimodal cross-retrieval**.  
- **Loss Function:** Combination of sheaf Laplacian loss + variance regularization of the restriction maps.

## Results

- **Sheaf Training Metrics:** Low discrepancy norm and moderate cosine similarity indicate effective merging of embeddings.  
- **Cross-Retrieval Performance (COCO):**
  - Image → Text: Recall@10 = 0.631  
  - Text → Image: Recall@10 = 0.232  

This demonstrates that the sheaf-theoretic framework improves embedding coherence and facilitates meaningful cross-modal retrieval.

## Framework Illustration

![Multimodal Sheaf Framework](images/Multimodal.png)  
*Figure: Visual representation of the multimodal sheaf-theoretic framework with restriction maps $P_{12}$ and $P_{21}$.*

## Report

A detailed report summarizing the experiment, methodology, results, and discussion is included in this repository. Please refer to `[Multimodal_Semantic_Communication_Papers_3.pdf](https://github.com/MiguelBenalcazar/Multimodal_sheaf_theoretic/blob/main/Multimodal_Semantic_Communication_Papers_3.pdf)` for a complete description.

---

## How to Run

1. Clone the repository:  
```bash
git clone https://github.com/yourusername/multimodal-sheaf-retrieval.git
