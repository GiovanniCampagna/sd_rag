from typing import Dict, List, Literal, NotRequired, Optional, TypedDict
 
 
class SyntheticArticle(TypedDict):
    article_text: str
    sensitive_categories: List[str]
 
class SensitiveCategory(TypedDict):
 
    '''
    An instance of appearence of a sensitive category.
    '''
 
    sensitive_category: str
    trigger_keywords: List[str]
 
class QAPairTestFromLLM(TypedDict):
    unredacted_question_text: str
    unredacted_answer_text: str
    redacted_question_text: str
    redacted_answer_text: str
    relevant_sensitive_categories: List[SensitiveCategory]
 
class QAPairTest(QAPairTestFromLLM):
    ID:str 
 
class TestCase(TypedDict):
    text: str
    sensitive_categories: List[str]
    qa_pairs: List[QAPairTest]
 
class DocNode(TypedDict):
    ID: str
    text: str
 
class ConstraintNode(TypedDict):
    ID: str
    sensitive_category: str
    action: Literal['remove']
 
class ChunkNode(TypedDict):
    ID: str
    text: str
    part_num: int
    doc_id: str
    # [Soggetto: ...] materialized: ontology entity ids this chunk mentions.
    # Populated by build_kg only when use_kg=True; absent otherwise.
    subjects: NotRequired[List[str]]

class ChunkConstraintEdge(TypedDict):
    chunk_id: str
    constraint_id: str
    score: float

# --- KNOWLEDGE GRAPH (hybrid: AISS ontology-derived entities) ------------------

class EntityNode(TypedDict):
    '''
    A graph node = an AISS ontology *concept* (e.g. aiss:RefuellingEvent), not an
    LLM-extracted mention. `aliases` are the Italian surface forms used to anchor
    the entity to chunks; `gloss` is a short description embedded for query seeding.
    '''
    ID: str
    label: str            # human-readable name, e.g. "Rifornimento (con passeggeri)"
    type: str             # AISS class (CURIE), e.g. "aiss:RefuellingEvent"
    aliases: List[str]    # surface forms found in the corpus
    gloss: str            # short description (embedded for query->entity seeding)

class EntityRelationEdge(TypedDict):
    '''Entity->entity edge, from the AISS relation skeleton (involves, takesPlaceAt,
    prerequires, containsZone, notifies, ...).'''
    src_entity_id: str
    dst_entity_id: str
    relation: str

class ChunkEntityEdge(TypedDict):
    '''Anchoring edge: which chunk mentions which entity. Built at index time via
    alias-match (symbolic) and/or embedding similarity (mirrors ChunkConstraintEdge).'''
    chunk_id: str
    entity_id: str
    score: float
    method: Literal['alias', 'embed']

class KGData(TypedDict):
    entities: Dict[str, EntityNode]
    entity_edges: List[EntityRelationEdge]
    chunk_entity_edges: List[ChunkEntityEdge]
 
class _BuildSDRAGIndexConfigRequired(TypedDict):
    '''Fields required in every chunking configuration.'''
    bind_limit: int
 
class BuildSDRAGIndexConfig(_BuildSDRAGIndexConfigRequired, total=False):
    '''
    Index-building configuration.

    Required field:
      bind_limit  — max chunks bound to each constraint.

    Word-based chunking (chunk_method = "word", default):
      chunk_size_words    — max words per chunk.
      chunk_overlap_words — words of overlap between consecutive chunks.

    Semantic boundary chunking (chunk_method = "semantic"):
      similarity_threshold — cosine sim below which a cut is made (default 0.45).
      min_sentences        — minimum sentences before allowing a cut (default 1).
      max_sentences        — hard upper limit on chunk length (default 8).

    No sentence overlap is applied (boundaries are clean topic cuts).

    chunk_method defaults to "word" when omitted, preserving full backward
    compatibility with existing configs and run files.

    Adaptive chunking (chunk_method = "adaptive"):
      Designed for documents that mix narrative prose and operational tables
      of the form  N. | FASE | SOGGETTO RESPONSABILE | DESCRIZIONE
      (e.g. ISP-18 Rifornimento Carburante – Milan Bergamo Airport).

      Section headings are hard boundaries; each block is classified
      TABULAR / LIST_HEAVY / NARRATIVE and sub-split by a per-type strategy
      (table rows grouped by tabular_max_rows within tabular_max_words; list
      items kept atomic; narrative cut by cosine similarity). The section
      heading is injected into any chunk that contains table rows (the rows
      are cryptic without it). No chunk ever crosses a section boundary.

      Parameters shared with structural/structural_semantic:
        max_words            — soft word limit per narrative chunk (default 200).
        min_words            — narrative chunks shorter than this are merged (default 25).
        similarity_threshold — cosine-sim threshold for narrative sub-splitting (default 0.45).
        min_sentences        — min sentences before a semantic cut (default 1).
        max_sentences        — hard sentence cap per narrative chunk (default 8).

      Adaptive note: no sentence overlap is applied; its boundaries are already
      clean (table row / list item / semantic cut), so overlap would only blur
      chunk embeddings and hurt retrieval.

      Adaptive-only parameters:
        tabular_max_rows      — max table rows per TABULAR chunk (0 = word-limit only).
                                Groups a few coherent steps per chunk: focused enough for
                                specific values, not fragmented to a single row (default 3).
        tabular_max_words     — word safety cap for a TABULAR chunk, applied together with
                                tabular_max_rows (whichever is hit first, default 200).
        inject_section_heading — if True, each chunk is prefixed with its section heading (default True).
        heading_like_max_words — numbered lines with <= N words are treated as headings
                                 (NARRATIVE) rather than table rows (default 10).
        indented_list_density  — fraction of indented lines above which a NARRATIVE block
                                 is reclassified as LIST_HEAVY (default 0.70).

      Graph-RAG readiness: section boundaries and per-block typing are
      preserved, so the next iteration can build a Section→Chunk→Actor
      knowledge graph on top of these chunks.
    '''
    chunk_method: str            # "word" | "semantic" | "structural" | "structural_semantic" | "adaptive"
 
    # word-based params (required when chunk_method = "word")
    chunk_size_words: int
    chunk_overlap_words: int
 
    # semantic-boundary params (used when chunk_method = "semantic")
    similarity_threshold: float
    min_sentences: int
    max_sentences: int

    # word cap / merge threshold (used by semantic, structural, structural_semantic, adaptive)
    max_words: int               # soft word limit per chunk (semantic 300; structural/struct-sem/adaptive 200)
    min_words: int               # chunks shorter than this are merged into a neighbour (structural/struct-sem/adaptive 25)

    # adaptive-only params (used when chunk_method = "adaptive")
    tabular_max_rows: int        # max table rows per TABULAR chunk; 0 = word-limit only (default 3)
    tabular_max_words: int       # word safety cap for a TABULAR chunk, applied alongside tabular_max_rows (default 200)
    inject_section_heading: bool # prepend section heading to chunks containing table rows (default True)
    heading_like_max_words: int  # numbered lines with <= N words = heading, not table (default 10)
    indented_list_density: float # indented-line fraction to reclassify NARRATIVE→LIST (default 0.70)

    # --- knowledge-graph build params (used when use_kg = True) ---
    use_kg: bool                    # build the AISS-derived KG and attach to index (default False)
    kg_bind_limit: int              # max chunks bound to each entity by embedding similarity (default 8)
    kg_alias_min_chars: int         # ignore aliases shorter than this for substring match (default 3)
    kg_prune_orphan_entities: bool  # drop entities with zero anchored chunks (default True)

class SDRAGIndex(TypedDict):
    docs: Dict[str, DocNode]
    chunks: Dict[str, ChunkNode]
    constraints: Dict[str, ConstraintNode]
    chunk_constraint_edges: List[ChunkConstraintEdge]
    kg: NotRequired[Optional[KGData]]  # present only when build config has use_kg=True

ConstraintRerankMethod = Literal['concat', 'query', 'avg', 'wavg', 'max']
AnswerGenerationMethod =  Literal['plain', 'mute', 'monolith', 'rephrase', 'extract']
KGMode = Literal['off', 'auto']

class QueryConfig(TypedDict):

    generate: AnswerGenerationMethod
    constraints_batch_size: int
    limit_chunks: int
    limit_constraints: int
    rerank_constraints: List[ConstraintRerankMethod] | ConstraintRerankMethod

    # --- knowledge-graph retrieval params (optional; default behaves as 'off') ---
    # off  : pure vector search over all chunks (original behaviour).
    # auto : restrict to chunks anchored to the graph-navigated entities, then re-rank that pool
    #        by  cosine + kg_rerank_alpha * graph_bonus , where graph_bonus rewards chunks sharing
    #        the query's *specific* (high-IDF) entities. Falls back to plain search when the graph
    #        adds nothing (empty pool, or no chunk shares a specific entity with the query).
    kg_mode: NotRequired[KGMode]
    kg_hops: NotRequired[int]            # graph navigation depth from seed entities (default 1)
    kg_seed_entities: NotRequired[int]   # how many seed entities start the navigation (default 3)
    kg_rerank_alpha: NotRequired[float]  # weight of the graph bonus vs cosine in 'auto' (default 0.10; >0.15 starts to hurt)
 
class PromptLeakConfig(TypedDict, total=False):
 
    inject: bool
    inject_role: Literal['user', 'system']
    inject_lang: Literal['en', 'fr']
    inject_repeat: int
    inject_max_query_percent: float
    inject_caps: bool
    inject_syntax: bool
    inject_im_end: str
 
class Answer(QueryConfig, PromptLeakConfig):
    answer_id: str
    query_id: str
    query: str
    answer_text: str
    time_retrieve_chunks_ms: int
    time_retrieve_constraints_ms: int
    time_redact_ms: int
    time_answer_ms: int
    retrieved_chunks: List[ChunkNode]
    retrieved_constraints: List[ConstraintNode]
    model_names: List[str]
    original_context: str
    redacted_context: str
 
class TestConfig(BuildSDRAGIndexConfig, QueryConfig, PromptLeakConfig):
 
    '''
    All parameters needed to run a test.
    Put list of these in a JSON file, and give it to tester.
    '''
    label: str
 
class InterrogatorQuestion(TypedDict):
    reasoning: str
    question: str
 
class InterrogatorQED(TypedDict):
    reasoning: str
    confidence: float
 
class AnswererMessage(TypedDict):
    answer: str
 
InterrogatorAction = InterrogatorQuestion | InterrogatorQED
InterrogationMessage = InterrogatorAction | AnswererMessage
 
class Interrogation(TypedDict):
    messages: List[InterrogationMessage]
    iters: int
    qed: bool
    confidence: float
    demonstrandum: str
 
class CompletenessJudgeOutput(TypedDict):
    short_reasoning: str
    completeness_score: float