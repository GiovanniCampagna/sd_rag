from typing import Dict, List, Literal, TypedDict
 
 
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
 
class ChunkConstraintEdge(TypedDict):
    chunk_id: str
    constraint_id: str
    score: float
 
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
      overlap_sentences    — sentences of overlap between chunks (default 1).

    chunk_method defaults to "word" when omitted, preserving full backward
    compatibility with existing configs and run files.

    Procedural chunking (chunk_method = "procedural"):
      Designed for documents that mix narrative prose and operational tables
      of the form  N. | FASE | SOGGETTO RESPONSABILE | DESCRIZIONE
      (e.g. ISP-18 Rifornimento Carburante – Milan Bergamo Airport).

      Each table row becomes ONE self-contained chunk prefixed with
      machine-readable metadata tags:
          [section:<label>] [type:table_row] [fase:<N>] [soggetto:<name>]
      Narrative blocks use structural-semantic sub-splitting.
      No chunk ever crosses a section boundary.

      Parameters shared with structural/structural_semantic:
        structural_max_words — soft word limit per narrative chunk (default 200).
        structural_min_words — narrative chunks shorter than this are merged (default 20).
        similarity_threshold — cosine-sim threshold for narrative sub-splitting (default 0.45).
        min_sentences        — min sentences before a semantic cut (default 1).
        max_sentences        — hard sentence cap per narrative chunk (default 8).
        overlap_sentences    — sentences of overlap for narrative chunks only (default 1).

      Adaptive-only parameters:
        tabular_max_rows      — max table rows per TABULAR chunk; 0 = word-limit only (default 0).
                                Set to 1 to give each procedural step its own focused chunk.
        inject_section_heading — if True, each chunk is prefixed with its section heading (default True).
        heading_like_max_words — numbered lines with <= N words are treated as headings
                                 (NARRATIVE) rather than table rows (default 10).
        indented_list_density  — fraction of indented lines above which a NARRATIVE block
                                 is reclassified as LIST_HEAVY (default 0.70).

      Graph-RAG readiness: the metadata tags on each chunk allow the next
      iteration to build a knowledge graph of Section→Chunk→Actor edges
      without any additional parsing step.
    '''
    chunk_method: str            # "word" | "semantic"
 
    # word-based params (required when chunk_method = "word")
    chunk_size_words: int
    chunk_overlap_words: int
 
    # semantic-boundary params (used when chunk_method = "semantic")
    similarity_threshold: float
    min_sentences: int
    max_sentences: int
    overlap_sentences: int

    # structural params (used when chunk_method = "structural")
    structural_max_words: int    # soft word limit per chunk (default 200)
    structural_min_words: int    # chunks shorter than this are merged (default 20)
    # overlap_sentences is shared with semantic (sentences of overlap, default 1)

    # adaptive-only params (used when chunk_method = "adaptive")
    tabular_max_rows: int        # max table rows per chunk; 0 = word-limit only (default 0)
    inject_section_heading: bool # prepend section heading to every chunk (default True)
    heading_like_max_words: int  # numbered lines with <= N words = heading, not table (default 10)
    indented_list_density: float # indented-line fraction to reclassify NARRATIVE→LIST (default 0.70)
 
class SDRAGIndex(TypedDict):
    docs: Dict[str, DocNode]
    chunks: Dict[str, ChunkNode]
    constraints: Dict[str, ConstraintNode]
    chunk_constraint_edges: List[ChunkConstraintEdge]
 
ConstraintRerankMethod = Literal['concat', 'query', 'avg', 'wavg', 'max']
AnswerGenerationMethod =  Literal['plain', 'mute', 'monolith', 'rephrase', 'extract']
 
class QueryConfig(TypedDict):
 
    generate: AnswerGenerationMethod
    constraints_batch_size: int
    limit_chunks: int
    limit_constraints: int
    rerank_constraints: List[ConstraintRerankMethod] | ConstraintRerankMethod
 
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
    # Token del contesto recuperato (pre-redazione): isola il costo del metodo
    # di chunking, indipendente dalla modalita' privacy. Metrica per il
    # confronto di efficienza token/completeness tra strategie di chunking.
    original_context_tokens: int
    # Token del contesto realmente passato al generatore finale (contesto
    # redatto per extract/rephrase, contesto originale per plain/monolith,
    # 0 per mute).
    answer_context_tokens: int
 
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