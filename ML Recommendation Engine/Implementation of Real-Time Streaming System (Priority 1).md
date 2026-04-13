# Implementation of Real-Time Streaming System (Priority 1)

This plan addresses the implementation of the **Priority 1 (Phase 3 Streaming Gaps)** block defined in the recommended execution order of `gap_analysis.md`. This effort will convert the existing mock Python skeletons into fully functional, production-ready PyFlink streaming topologies that calculate real-time metrics, user embeddings, item onboarding updates, and handle event enrichment.

## User Review Required

> [!WARNING]
> Please confirm if "P1" corresponds to **Priority 1 (Phase 3 Streaming Gaps - items 1-6)** as defined in the `Recommended Execution Order` of your `gap_analysis.md`, or if you meant **Phase 1 (MVP)** which only has the Go gRPC Sidecar missing. This plan assumes Priority 1.

## Proposed Changes

---

### Priority 1: Streaming Processing

Currently, all the Python Flink streaming job files use simulated functions (`pass`, random value generators, pseudo-python dictionaries instead of actual PyFlink types). We will update the PyFlink API to use actual streams.

#### [MODIFY] [item_stats_job.py](file:///d:/Claude%20Code%20Practice/ML%20Recommendation%20Engine/streaming/item_stats/item_stats_job.py)
- Convert the mocked Python class into a proper PyFlink Execution graph.
- Setup `StreamExecutionEnvironment`.
- Create a Kafka Source for event ingestion.
- Implement the tumbling/sliding window function `ProcessWindowFunction` for item metrics.
- Write the Redis sink implementation.

#### [MODIFY] [user_embedding_job.py](file:///d:/Claude%20Code%20Practice/ML%20Recommendation%20Engine/streaming/user_embeddings/user_embedding_job.py)
- Replace `random.gauss()` dummy outputs with proper integration of `UserEmbeddingInferencer`.
- Replace the `pass` within `build_pipeline()` with real PyFlink stream operators routing embeddings to dual sinks (Redis and Milvus).

#### [MODIFY] [onboarding_job.py](file:///d:/Claude%20Code%20Practice/ML%20Recommendation%20Engine/streaming/item_onboarding/onboarding_job.py)
- Implement `build_pipeline()`.
- Add proper logic to intercept the `item-onboarding` Kafka topic and compute realistic content embeddings using a specified tokenizer/model.

#### [MODIFY] [enrichment_job.py](file:///d:/Claude%20Code%20Practice/ML%20Recommendation%20Engine/streaming/enrichment/enrichment_job.py)
- Replace mock `async_invoke()` implementation.
- Introduce `AsyncWaitOperator` for high-throughput, non-blocking asynchronous requests against Redis/Elasticsearch.

#### [MODIFY] [session_features_job.py](file:///d:/Claude%20Code%20Practice/ML%20Recommendation%20Engine/streaming/session_features/session_features_job.py)
- Fully integrate the currently defined `handle_late_event()` logic into a true Flink `OutputTag` side-output configuration, rather than using standard dictionary returns.

### Configuration

#### [MODIFY] [flink-conf.yaml](file:///d:/Claude%20Code%20Practice/ML%20Recommendation%20Engine/streaming/flink-conf.yaml)
- Verify that `state.backend`, checkpoint intervals, and PyFlink dependencies exactly match the scale of the required execution jobs. (The file already has 64 lines of config, we will perform a review and adjust anything missing for `ItemStats` or `SessionFeatures` State size requirements).

## Open Questions

> [!IMPORTANT]
> 1. **Ambiguity on "P1"**: To be absolutely certain so we do not build the wrong components, did you mean **Priority 1 (Phase 3 Streaming Gaps)** out of the `gap_analysis.md` recommended execution order, or did you mean **Phase 1 (MVP)** which only has the Go gRPC Serving Sidecar missing?
> 2. What PyFlink version should be targeted for dependencies (e.g., 1.17, or 1.18)?

## Verification Plan

### Automated Tests
- Validate PyFlink typing using static analyzers or testing with `pytest`.
- Run a quick DAG print test inside Flink to ensure the `StreamExecutionEnvironment` successfully builds a robust runtime topology without runtime evaluation errors.

### Manual Verification
- We can write a scratch test file to emit synthetic mock records and view if the PyFlink window components correctly output into local storage or stdout before integrating cleanly into production K8s mappings.
