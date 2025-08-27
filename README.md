# Pathfinder Rules Q&A System

A powerful question-answering system specifically designed for Pathfinder RPG rulebooks, built with LangGraph and AI models. This system can process Pathfinder PDF documents and answer questions about game rules, mechanics, and gameplay using advanced AI techniques including vector search, keyword matching, and multilingual support.

## Features

- **Pathfinder-specific optimization**: Tailored for RPG rulebooks and game mechanics
- **Multi-document support**: Process multiple Pathfinder PDF documents simultaneously
- **Smart caching**: Efficient caching system to avoid reprocessing unchanged documents
- **Vector search**: Semantic search using OpenAI embeddings (optional)
- **Keyword fallback**: Robust keyword-based search when vector search is unavailable
- **Multilingual support**: Automatic language detection and translation for global players
- **Content moderation**: Built-in safety checks and relevance validation for gaming content
- **Response validation**: AI-powered response quality assessment with retry logic
- **Conflict resolution**: Handles multiple rule sources with priority-based resolution

## Prerequisites

1. Create a Groq API key at [Groq Console](https://console.groq.com/keys)
2. (Optional) Create an OpenAI API key for enhanced vector search capabilities
3. Add your API keys to a `.env` file in the project root

## Installation

```bash
pnpm install
```

## Configuration

Create a `.env` file with your API keys:

```env
GROQ_API_KEY=your_groq_api_key_here
OPENAI_API_KEY=your_openai_api_key_here  # Optional
```

## Usage

1. Place your Pathfinder PDF documents in the `resources/` folder
2. Run the system:

```bash
pnpm dev
```

3. The system will process your documents and run through a set of test questions
4. You can also run the custom test suite:

```bash
pnpm test
```

## Architecture

The system uses a LangGraph-based pipeline with the following components:

- **Document Processing**: PDF loading, chunking, and metadata extraction
- **Caching**: File-based cache with metadata change detection
- **Search**: Hybrid vector and keyword search optimized for RPG content
- **AI Generation**: Groq-powered answer generation with Pathfinder expertise
- **Validation**: Response quality assessment and retry logic
- **Translation**: Multilingual support for international gaming communities

## Configuration Options

Key configuration options in `config.mts`:

- `CHUNK_SIZE`: Size of text chunks (default: 1000)
- `CHUNK_OVERLAP`: Overlap between chunks (default: 200)
- `VECTOR_SEARCH_TOP_K`: Number of vector search results (default: 3)
- `KEYWORD_SEARCH_TOP_K`: Number of keyword search results (default: 3)
- `RESPONSE_VALIDATION_THRESHOLD`: Minimum response quality score (default: 5)
- `RETRY_COUNT`: Maximum retry attempts for failed responses (default: 3)

## Use Cases

- **Game Masters**: Quick rule lookups during sessions
- **Players**: Learning game mechanics and character creation
- **International Groups**: Multilingual rule explanations
- **Rule Disputes**: Resolving conflicts between different rule sources
- **Learning**: Understanding complex Pathfinder systems

## License

ISC