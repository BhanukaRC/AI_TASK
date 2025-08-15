# Pathfinder Rules Interrogation System

## Assignment Submission – Bhanuka Siriwardana

> **AI Assistance Disclosure:**
> I used AI (OpenAI GPT-4 via Cursor, Claude AI) as a coding assistant throughout the development of this project. The design decisions were all taken by me. All architectural decisions, code, and documentation were reviewed and finalized by me, but AI was leveraged for brainstorming, code generation, and best-practice validation. I take the full responsibility of the code.

---

## Quickstart

1. Install dependencies: `pnpm install`
2. Set up your `.env` file with the required API keys (see `config.mts` for details)
3. Place your PDFs in the `resources/` folder
4. Run: `pnpm run dev`
5. You can also run a custom test suite I made with `pnpm run test`;

Note: It is likely that the Groq API Rate Limits are hit when running the queries all at once. Therefore, I have commented a few of the tests I added in `qa_questions.mts` and have added a custom timeout in each script. Please feel free to uncomment all the test cases and increase the timeout if necessary. Console logging in the test script may appear out of order due to asynchronous execution, but this is handled in the `main` function of `index.mts`.

---

## Architecture & Design Decisions

### 1. Cost Optimization Through Smart Caching

**Why Cache?**
- **AI embeddings are expensive**: Each chunk costs some money to embed
- **Documents rarely change**: Game rules, legal documents, etc. are static
- **Server restarts can be common**: In-memory only would lose all processed data

**Implementation Strategy:**
- **File system persistence** for simplicity (vs. MongoDB/document stores)
- **Metadata-based change detection**: File size, creation time, modification time, page count
- **Incremental updates**: Only reprocess changed documents, not entire cache

### 2. Vector Store Limitations & Workarounds

I tried to integrate ChromaDB, FaissStore and HNSWLib but ran into native dependency compilation issues on macOS ARM64 and other dependency problems.

**Current Approach:**
- **In-memory vector store** with file-based metadata persistence
- **Graceful fallback** to keyword-based search when vector search unavailable
- **Hybrid system**: Vector search when possible, keyword search as backup

### 3. Flexible AI Integration

**OpenAI Key Optional:**
- **With key**: Full vector search + AI generation + response validation
- **Without key**: Rule-based keyword search + AI generation, no OpenAI validation
- **No forced dependencies**: System works regardless of API key availability

### 4. Multi-Document Conflict Resolution

**Smart Content Management:**
- **Priority-based conflict resolution**: Core rules > Advanced rules > Optional content
- **Source attribution**: Every answer cites specific documents and pages
- **Conflict awareness**: AI acknowledges when sources disagree

### 5. Multilingual Support & Robust Moderation

- **Automatic language detection and translation**: User questions are translated to English for processing, and answers are translated back to the user's language if needed. This was done because AI models tend to work better with English queries.
- **Moderation pipeline**: Rule-based and AI-based (OpenAI/Groq) moderation to block inappropriate or off-topic queries.
- **Response validation**: OpenAI is used to rate the AI's answer (0–10) for relevance/quality, with a configurable threshold and retry logic.

---

## How It Works

### Cache Management Flow
```
1. Check if documents have changed (metadata comparison)
2. If unchanged: Load from cache (instant)
3. If changed: Reprocess only changed documents
4. Update cache with new content
5. Create vector embeddings (if OpenAI key available)
```

### Query Processing Flow
```
User Question (any language)
→ Language Detection & Translation (to English)
→ Moderation (rule-based + AI)
→ Vector Search (if available) or Keyword Search
→ AI Generation (Groq, in English)
→ Response Validation (OpenAI, 0–10 score, retry if needed)
→ Translation (back to user's language if needed)
→ Answer with Sources
```

---

## Production Considerations / Thought Process

### 1. Background Processing for High-Volume Systems
**Current Limitation**: Updates only detected on user queries
**Production Solution**: Can implement a background worker that periodically:
- Monitors document sources for changes (example: News aggregation system where news content resources may come up every hour)
- Updates cached content proactively
- Maintains fresh vector store
- User queries would not be affected or delayed.

### 2. Feedback-Driven Optimization
**Proposed System:**
- Collect user feedback on answer quality (if the user queries come as chat messages)
- Dedicated ML model analyzes feedback patterns
- Automatically tunes parameters:
  - `CHUNK_SIZE`: Optimal text chunking
  - `CHUNK_OVERLAP`: Balance between context and efficiency
  - `VECTOR_SEARCH_TOP_K`: Optimal result count
  - `KEYWORD_SEARCH_TOP_K`: Fallback result count
- Note: `RESPONSE_VALIDATION_THRESHOLD`: Should also be determined experimentally.

### 3. Usage Analytics for Better Fallbacks
**Current Limitation**: No tracking of chunk popularity
**Production Enhancement:**
- Track which chunks are most referenced
- Use popularity for fallback selection
- Improve answer quality when exact matches aren't found

### 4. Robust Error Handling
**Current**: Failed queries are discarded
**Production**: Implement retry with exponential backoff
- Retry failed queries up to threshold
- Label persistently failing queries as "poisonous"
- Maintain system stability under load

---

## Technical Implementation

### Change Detection
- **File metadata comparison**: Size, modification time
- **Incremental updates**: Only changed documents trigger reprocessing

### Fallback Mechanisms
1. **Vector search fails** → Keyword search
2. **No OpenAI key** → Keyword search only
3. **No relevant chunks** → Return first few chunks. But ideally, return most popular chunks

### Multilingual & Moderation Pipeline
- **Language detection and translation**: All queries processed in English, answers translated back if needed.
- **Moderation**: Rule-based and AI-based (OpenAI/Groq) checks.
- **Response validation**: OpenAI rates answer (0–10); retries if below threshold.

---

## What I'd Do with More Time
- Make this a WebSocket server (ideal for chat/interactive use cases; REST API could also be offered for simple integrations).
- Add automated tests for all pipeline nodes and edge cases
- Integrate a persistent vector store (e.g. ChromaDB) for true embedding persistence
- Even with a persistent vector store like ChromaDB, it’s important to persist resource-level metadata (file size, modification time, page count, etc.) to efficiently detect changes and only reprocess PDFs that have actually changed. This minimizes unnecessary computation and cost.
- Add usage analytics and feedback-driven parameter optimization
- Implement background document monitoring and chunk popularity tracking

---