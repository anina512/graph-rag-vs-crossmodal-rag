# 🍽️ GraphRAG vs Cross-Modal RAG  
### Multimodal Retrieval-Augmented Generation on Recipe Datasets (Text + Image)

This repository presents a comprehensive comparison of two advanced Retrieval-Augmented Generation (RAG) systems on a large-scale multimodal recipe dataset (text + images). Both systems use the **same dataset, prompts, and LLM**, allowing a **pure comparison of retrieval mechanisms** rather than model performance.

---

## 🔍 High-Level Comparison

| Aspect | **Microsoft GraphRAG** | **Cross-Modal RAG (CM-RAG)** |
|--------|------------------------|------------------------------|
| Retrieval Mechanism | Knowledge-graph reasoning using **local + global search** | Multimodal embeddings using **SBERT + CLIP + FAISS** |
| Retrieval Signals | **Entity relationships**, **ingredient co-occurrence**, **community structure** | **Semantic similarity** (text) + **visual similarity** (dish appearance) |
| Output Behavior | **Analytical and category-level answers** driven by graph structure | **Concrete, ready-to-cook recipes** with full ingredients and instructions |
| Faithfulness to Query | **Strict** — tightly adheres to input ingredients / constraints | **Flexible** — retrieves closely related variations even beyond exact ingredients |
| Interaction Style | Behaves like a **knowledge explainer** | Behaves like a **practical recipe assistant** |

---

## 🎯 Motivation

While standard RAG improves factual grounding, it still struggles when:
- queries require **reasoning over relationships** (e.g., substitutes, dish families),
- retrieval must incorporate **visual cues**,
- or users need **structured, actionable cooking steps**.

GraphRAG and Cross-Modal RAG address these gaps in complementary ways.  
This project evaluates their **strengths, limitations, and downstream effect on LLM responses**.

---

## 📦 Dataset

- Source: ~13K recipe posts including titles, ingredients, instructions, and dish images  
- Final cleaned dataset: ~10K valid recipe–image pairs  
- Preprocessing includes:
  - removal of missing images and empty fields,
  - standardization of ingredient format,
  - deduplication and length filtering.
 


---

## 🌟 Cross-Modal-Rag-Implementation  
### A Multimodal Recipe Retrieval & Suggestion Engine (Text + Image RAG)

This project implements a **Cross-Modal Retrieval-Augmented Generation (RAG)** system for *food & recipe recommendations*.  
It retrieves supporting evidence from thousands of recipes using:

- **SBERT** for text embeddings  
- **CLIP** for image + text multimodal embeddings  
- **FAISS** for fast similarity search  
- **LLaMA 3.2 (3B-Instruct)** for reasoning and recipe generation  

The system supports **three types of queries**:
1. **Text-only RAG**  
2. **Image-only RAG**  
3. **Text + Image RAG**

###  What This System Can Do
Given a text query like:
> "vegan pasta with broccoli"
or an uploaded image like:
> *(dish photo)*
or both together:
> "Something similar to this dish but spicier" + *(image)*
The system will:
1. Retrieve **top-K relevant recipe texts** (SBERT → FAISS)  
2. Retrieve **top-K relevant dish images** (CLIP → FAISS)  
3. Perform cross-modal retrieval  
   - text → image  
   - image → text  
   - image → image  
4. Build an intelligent **RAG prompt** containing multimodal evidence  
5. Use **LLaMA 3.2** to produce:
   - best-matching recipes  
   - ingredients lists  
   - explanations  
   - customization ideas  
   - diet-friendly alternatives  

### 🧠 System Architecture
```
  ┌──────────────────────┐
  │   User Query Input   │
  │  (Text / Image / Both)
  └───────────┬──────────┘
              │
              ▼
   ┌──────────────────────┐
   │  Cross-Modal Embedder│
   │ SBERT / CLIP Encoder │
   └───────────┬──────────┘
               │
               ▼
     ┌──────────────────┐
     │   FAISS Indexes  │
     │ text | image | clip_text
     └───────┬──────────┘
             │
             ▼
   ┌────────────────────────┐
   │  Retriever (4 modes)   │
   │ text→text              │
   │ text→image             │
   │ image→image            │
   │ image→text             │
   └──────────┬────────────┘
              │
              ▼
     ┌──────────────────┐
     │  Prompt Builder  │
     │  (Multimodal RAG)│
     └──────────┬───────┘
                │
                ▼
   ┌─────────────────────────┐
   │  LLaMA 3.2 (3B-Instruct) │
   │     Recipe Reasoning     │
   └───────────┬─────────────┘
               │
               ▼
     ┌────────────────────────┐
     │  Final Recipe Output   │
     └────────────────────────┘
```

### 📦 Project Structure
```
cross-modal-rag-implementation/
│
├── data/
│   ├── Food Images/
│   ├── Food Ingredients CSV
│
├── embeddings/
│   ├── sbert_text_embs.npy
│   ├── clip_text_embs.npy
│   ├── image_embs.npy
│   ├── ids.npy
│
├── indexes/
│   ├── text.index
│   ├── clip_text.index
│   ├── image.index
│
├── out/
│   ├── retrieved_evidence.txt
│   ├── generated_prompt.txt
│   ├── llm_output.txt
│
├── src/
│   ├── data_loader.py
│   ├── embedder.py
│   ├── build_embeddings.py
│   ├── build_index.py
│   ├── retriever.py
│   ├── prompt_builder.py
│   ├── llm_inference.py
│   ├── test_data_loading.py
│   ├── test_rag_prompt.py
│
└── README.md
```

### 🚀 Setup Instructions

#### 0. Download Dataset

Download the dataset from Kaggle:

**[Food Ingredients and Recipe Dataset with Images](https://www.kaggle.com/datasets/pes12017000148/food-ingredients-and-recipe-dataset-with-images)**

Extract the contents into the `data/` directory:
- Place images in `data/Food Images/`
- Place the CSV file in `data/`

#### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

#### 📥 Step 1 — Load Dataset
```bash
python -m src.test_data_loading
```

This validates the recipe CSV + images and generates a `data_preview.txt`.

#### 🧮 Step 2 — Build Embeddings
```bash
python -m src.build_embeddings
```

This generates:
- `sbert_text_embs.npy`
- `clip_text_embs.npy`
- `image_embs.npy`
- `ids.npy`

#### ⚡ Step 3 — Build FAISS Indexes
```bash
python -m src.build_index
```

Creates:
- `indexes/text.index`
- `indexes/clip_text.index`
- `indexes/image.index`

### 🔍 Step 4 — Test Retrieval
```bash
python -m src.test_retrieval
```

Runs:
- text → text
- text → image
- image → image
- image → text

#### 🤖 Step 5 — Full Multimodal RAG
```bash
python -m src.test_rag_prompt
```
Produces:
- `out/retrieved_evidence.txt`
- `out/generated_prompt.txt`
- `out/llm_output.txt`

### 🧠 LLM Used

This project uses:
```
meta-llama/Llama-3.2-3B-Instruct
```

To change the model, edit: `src/llm_inference.py`

### 🧪 Query Modes Supported

#### ✅ 1. Text-Only Query

**Example:**
```
"vegan pasta with broccoli"
```

System runs:
- text→text
- text→image
- pseudo image→text
- pseudo image→image

#### ✅ 2. Image-Only Query

**Example:**
```python
img = Image.open("my_dish.jpg")
```

System runs:
- image→text
- image→image

#### ✅ 3. Combined Text + Image Query

**Example:**
```
"Something like this dish but spicier" + (uploaded image)
```

Perfect for personalization.

### 🎯 Future Improvements (Optional)

- Add BLIP / LLaVA image captions
- Cluster recipes by cuisine
- Add nutrition prediction
- Add FastAPI backend + React frontend
- Build a "chat with your pantry" system

---
## 🌟 Graph-Rag-Implementation
### Knowledge-Graph Reasoning for Recipe Understanding & Retrieval

This module implements **Microsoft GraphRAG**, an advanced Retrieval-Augmented Generation (RAG) framework that enhances retrieval by constructing a **knowledge graph** from the dataset. Instead of relying only on text embeddings, GraphRAG identifies **entities, relationships, and communities** within recipes and uses this structured representation to generate **more grounded, interpretable responses**.

### 🧠 What GraphRAG Does

Given a user cooking query (e.g., ingredients or recipe questions), GraphRAG:
1. Converts recipe text into **text chunks**
2. Extracts **entities and relationships** (e.g., ingredients ↔ recipes)
3. Builds a **knowledge graph** from the full dataset
4. Generates **community-level summaries** of recipe clusters
5. Uses **local and global search pipelines** to answer queries
6. Produces **LLM-generated responses grounded in graph evidence**

Compared to standard RAG, GraphRAG is particularly effective for:
- Ingredient-based reasoning  
- Substitutions & category suggestions  
- Structured, analytical explanations  

### ⚙️ Full Pipeline Overview

GraphRAG processes the dataset in a series of structured stages to transform raw recipe text into a searchable knowledge graph and then uses it to answer user queries.

#### 🔹 1. Data Ingestion
Loads the cleaned recipe dataset from `input/` based on the configuration defined in `settings.yaml`. The `merged_text` field is extracted from each row to form the raw textual corpus.

#### 🔹 2. Text Chunking
Long recipe documents are split into overlapping text windows to preserve semantic continuity.  
- Chunk size: 500 tokens  
- Overlap: 75 tokens  
The result is a uniform set of chunks for downstream embedding and graph extraction.

#### 🔹 3. Text Embedding
Each chunk is embedded using **Nomic Text Embedding v1.5** and stored in **LanceDB** to enable fast semantic search. These embeddings are used for:
- Local (standard RAG) retrieval
- The knowledge graph extraction workflow

#### 🔹 4. Graph Extraction (Entity + Relationship Identification)
For every chunk, an LLM (Llama-3.1-8B-Instruct) identifies:
- **Entities** (ingredients, dish names, cooking concepts, etc.)
- **Relationships** between entities  
All extracted entities are deduplicated and normalized, and a Python **NetworkX** graph is constructed.

#### 🔹 5. Graph Summarization (Entity Descriptions)
For each graph node, the LLM aggregates all mentions of that entity across the corpus and generates a **canonical description**. These summaries serve as global-context evidence for retrieval.

#### 🔹 6. Community Detection
Semantic clusters of related entities are created based on graph modularity.  
- Cluster size capped for readability  
Each cluster represents a coherent culinary theme (e.g., tomato-based pastas).

#### 🔹 7. Community Reports
The LLM generates high-level summaries of each cluster using a map-reduce prompting strategy, producing human-readable descriptions of what each community represents.

#### 🔹 8. Retrieval Pipelines
GraphRAG supports multiple search strategies; this project uses the two core ones:

| Search Type | Evidence Source | Best For |
|-------------|----------------|----------|
| **Local Search** | Embedding similarity using LanceDB | Short factual queries |
| **Global Search** | Community summaries + graph relationships | Multi-hop & reasoning queries |

Global search operates in three stages:
Map → Knowledge → Reduce


#### 🔹 9. Final LLM Response
Retrieved evidence (local or global) is passed to the generation model.  
The LLM synthesizes a final answer that:
- explains ingredient relationships,
- preserves constraints,
- and avoids hallucination through grounded evidence.

---
