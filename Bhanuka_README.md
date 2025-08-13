## Architecture & Design Decisions

### 1. Cost Optimization Through Smart Caching

**Why Cache?**
- **AI embeddings are expensive**: Each chunk costs some money to embed
- **Documents rarely change**: Game rules, legal documents, etc. are static
- **Server restarts can be common**: In-memory only would lose all processed data

**Implementation Strategy**:
- **File system persistence** for simplicity (vs. MongoDB/document stores)
- **Metadata-based change detection**: File size, creation time, modification time, page count
- **Incremental updates**: Only reprocess changed documents, not entire cache

### 2. Vector Store Limitations & Workarounds

I tried to intergreate ChromaDB, FaissStore and HNSWLib but ran into native dependency compilation issues on macOS ARM64 and other dependency problems.

**Current Approach**:
- **In-memory vector store** with file-based metadata persistence
- **Graceful fallback** to keyword-based search when vector search unavailable
- **Hybrid system**: Vector search when possible, keyword search as backup

### 3. Flexible AI Integration

**OpenAI Key Optional**:
- **With key**: Full vector search + AI generation
- **Without key**: Rule-based keyword search + AI generation
- **No forced dependencies**: System works regardless of API key availability

### 4. Multi-Document Conflict Resolution

**Smart Content Management**:
- **Priority-based conflict resolution**: Core rules > Advanced rules > Optional content
- **Source attribution**: Every answer cites specific documents and pages
- **Conflict awareness**: AI acknowledges when sources disagree

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
User Question → Vector Search (if available) → Keyword Search (fallback) → AI Generation → Answer with Sources
```

## Production Considerations / Thought Process

### 1. Background Processing for High-Volume Systems
**Current Limitation**: Updates only detected on user queries
**Production Solution**: Background worker that periodically:
- Monitors document sources for changes (example: News aggregation system where news content resources may come up every hour)
- Updates cached content proactively
- Maintains fresh vector store
- User queries would not be affecred.

### 2. Feedback-Driven Optimization
**Proposed System**:
- Collect user feedback on answer quality (if the user queries come as chat messages)
- Dedicated ML model analyzes feedback patterns
- Automatically tunes parameters:
  - `CHUNK_SIZE`: Optimal text chunking
  - `CHUNK_OVERLAP`: Balance between context and efficiency
  - `VECTOR_SEARCH_TOP_K`: Optimal result count
  - `KEYWORD_SEARCH_TOP_K`: Fallback result count

### 3. Usage Analytics for Better Fallbacks
**Current Limitation**: No tracking of chunk popularity
**Production Enhancement**:
- Track which chunks are most referenced
- Use popularity for fallback selection
- Improve answer quality when exact matches aren't found

### 4. Robust Error Handling
**Current**: Failed queries are discarded
**Production**: Implement retry with exponential backoff
- Retry failed queries up to threshold
- Label persistently failing queries as "poisonous"
- Maintain system stability under load

## Technical Implementation

### Change Detection
- **File metadata comparison**: Size, modification time
- **Incremental updates**: Only changed documents trigger reprocessing

### Fallback Mechanisms
1. **Vector search fails** → Keyword search
2. **No OpenAI key** → Keyword search only
3. **No relevant chunks** → Return first few chunks. But ideally, return most popular chunks

## Future Enhancements

1. **True embedding persistence** (use ChromaDB)
2. **Background document monitoring** for high-volume systems
3. **Feedback-driven parameter optimization**
4. **Usage analytics and popularity tracking**
5. **Robust retry mechanisms with poison detection**
6. **Document store integration** (MongoDB) for persisting processed resources content
