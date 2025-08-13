import dotenv from "dotenv";
dotenv.config();

import {
  CONFIG,
  validateEnvironment,
  ensureDirectoriesExist,
  getFeatureFlags,
} from "./config.mjs";

import { ChatGroq } from "@langchain/groq";
import { MemorySaver } from "@langchain/langgraph";
import { createReactAgent } from "@langchain/langgraph/prebuilt";
import {
  type BaseMessage,
  HumanMessage,
  SystemMessage,
} from "@langchain/core/messages";
import { StateGraph, Annotation } from "@langchain/langgraph";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { RecursiveCharacterTextSplitter } from "langchain/text_splitter";
import { OpenAIEmbeddings } from "@langchain/openai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";
import * as fs from "fs";
import * as path from "path";

// Global cache management
let globalCache: CacheData | null = null;

// Types
interface ChunkWithScore {
  chunk: string;
  score: number;
  index: number;
}

interface DocumentChunk {
  content: string;
  metadata: {
    source: string;
    sourceName: string;
    page: number;
    priority: number; // For conflict resolution
  };
}

interface CacheData {
  documents: Array<{
    path: string;
    filename: string;
    size: number;
    pageCount: number;
    created: string;
    lastModified: string;
    priority: number;
  }>;
  chunks: string[];
  chunksWithMetadata: DocumentChunk[];
  totalChunks: number;
  timestamp: string;
  hasVectorStore: boolean;
}

interface PDFCacheData {
  chunks: DocumentChunk[];
  content: string;
  filename: string;
  fromCache: boolean;
  error?: string;
}

// Utility functions
const getDocumentPriority = (filename: string): number => {
  // Only a PoC for this assignment. Not a real priority system.
  const name = filename.toLowerCase();
  if (name.includes("core") || name.includes("basic")) return 10;
  if (name.includes("advanced") || name.includes("rule")) return 7;
  if (name.includes("optional") || name.includes("variant")) return 5;
  if (name.includes("adventure") || name.includes("fans")) return 3;
  return 6; // Default priority
};

const readFileAsync = async (filePath: string): Promise<string> => {
  try {
    const fileBuffer = await fs.promises.readFile(filePath);
    return fileBuffer.toString();
  } catch (error) {
    console.error(`Failed to read file ${filePath}:`, error.message);
    throw error;
  }
};

const retryWithBackoff = async (
  name: string,
  fn: () => Promise<any>,
  maxRetries: number = 3,
  delay: number = 1000
): Promise<any> => {
  let retries = 0;
  while (retries < maxRetries) {
    try {
      return await fn();
    } catch (error) {
      retries++;
      console.log(`Retrying... (${retries}/${maxRetries}) for ${name}`);
      await new Promise((resolve) => setTimeout(resolve, delay));
    }
  }
  throw new Error(`Failed to execute function after ${maxRetries} retries`);
};

// Cache management functions
// Using a file-based cache is a simple solution for now, but we should consider using a proper database for persistence
const loadCacheFromDisk = async (): Promise<CacheData | null> => {
  try {
    if (!fs.existsSync(CONFIG.CACHE_PATH)) {
      console.log("No cache file found on disk");
      return null;
    }

    const cacheContent = await readFileAsync(CONFIG.CACHE_PATH);
    const cached: CacheData = JSON.parse(cacheContent);

    // Basic validation - ensure we have the essential fields
    if (
      !cached.chunks ||
      !cached.documents ||
      !Array.isArray(cached.chunks) ||
      !Array.isArray(cached.documents)
    ) {
      console.log("Cache format invalid, ignoring disk cache");
      return null;
    }

    console.log(
      `Loaded cache from disk: ${cached.totalChunks} chunks from ${cached.documents.length} documents`
    );
    return cached;
  } catch (error) {
    console.error("Failed to load cache from disk:", error);
    return null;
  }
};

// Agent functions
const extractUserQuestion = (messages: BaseMessage[]): string => {
  const humanMessages = messages.filter((msg) => msg instanceof HumanMessage);
  return humanMessages[humanMessages.length - 1]?.content?.toString() || "";
};

const createStateUpdate = (
  state: typeof StateAnnotation.State,
  updates: Partial<typeof StateAnnotation.State>
): typeof StateAnnotation.State => ({
  ...state,
  ...updates,
});

// Agent setup
const agentModel = new ChatGroq({
  apiKey: process.env.GROQ_API_KEY,
  model: CONFIG.GROQ_MODEL,
});

const agentCheckpointer = new MemorySaver();
const agent = createReactAgent({
  llm: agentModel,
  tools: [],
  checkpointSaver: agentCheckpointer,
});

// Graph node functions
const checkCache = async (state: typeof StateAnnotation.State) => {
  console.log("Checking multi-document cache...");

  try {
    const cached: CacheData | null = await loadCacheFromDisk();

    if (!cached) {
      console.log("No valid cache available, will process PDFs");
      return createStateUpdate(state, { cacheUsed: false });
    }

    const currentPdfPaths = getResourcePDFs();
    let cacheNeedsUpdate: boolean = false;
    let updatedCache: CacheData = { ...cached };

    // Step 1: Check for new/removed documents
    const currentFilenames = currentPdfPaths.map((p) => path.basename(p));
    const cachedFilenames = cached.documents.map((d) => d.filename);

    // Add new documents
    const newDocuments = currentFilenames.filter(
      (name) => !cachedFilenames.includes(name)
    );
    if (newDocuments.length > 0) {
      console.log(`New documents found: ${newDocuments.join(", ")}`);
      cacheNeedsUpdate = true;
    }

    // Remove deleted documents
    const deletedDocuments: string[] = cachedFilenames.filter(
      (name) => !currentFilenames.includes(name)
    );
    if (deletedDocuments.length > 0) {
      console.log(`Deleted documents: ${deletedDocuments.join(", ")}`);
      cacheNeedsUpdate = true;
      // Remove from cache
      updatedCache.documents = updatedCache.documents.filter(
        (doc) => !deletedDocuments.includes(doc.filename)
      );
    }

    // Step 2: Check metadata for existing documents
    for (const currentPath of currentPdfPaths) {
      const currentFilename: string = path.basename(currentPath);
      const cachedDoc = cached.documents.find(
        (doc) => doc.filename === currentFilename
      );

      if (!cachedDoc) {
        // New document - will be processed later
        continue;
      }

      try {
        const currentStats: fs.Stats = fs.statSync(currentPath);
        const currentSize: number = currentStats.size;
        const currentModified: string = currentStats.mtime.toISOString();

        // Check if metadata changed
        if (
          cachedDoc.size !== currentSize ||
          cachedDoc.lastModified !== currentModified
        ) {
          console.log(
            `PDF metadata changed: ${currentFilename} (size: ${cachedDoc.size} → ${currentSize}, modified: ${cachedDoc.lastModified} → ${currentModified})`
          );
          cacheNeedsUpdate = true;
        }
      } catch (statsError) {
        console.log(
          `Failed to get stats for ${currentFilename}, will reprocess:`,
          statsError.message
        );
        cacheNeedsUpdate = true;
      }
    }

    if (!cacheNeedsUpdate) {
      console.log("All PDFs unchanged, loading from in-memory cache...");
      console.log(
        `Loaded ${cached.totalChunks} chunks from ${cached.documents.length} documents`
      );

      // Log which documents were loaded
      cached.documents.forEach((doc) => {
        console.log(
          `  ${doc.filename} (${((doc.size || 0) / 1024).toFixed(2)} KB, ${
            doc.pageCount
          } pages, priority: ${doc.priority})`
        );
      });

      return createStateUpdate(state, {
        chunks: cached.chunks,
        chunksWithMetadata: cached.chunksWithMetadata,
        documentChunks: cached.chunksWithMetadata, // For compatibility
        cacheUsed: true,
      });
    }

    console.log("Cache needs update, will process changed/new documents");
    return createStateUpdate(state, { cacheUsed: false });
  } catch (error) {
    console.log("Cache check failed, will process PDFs:", error);
    return createStateUpdate(state, { cacheUsed: false });
  }
};

const getResourcePDFs: () => string[] = () => {
  try {
    if (!fs.existsSync(CONFIG.RESOURCES_DIR)) {
      console.log(`Resources folder ${CONFIG.RESOURCES_DIR} not found`);
      return [];
    }

    // Only processing PDFs for simplicity
    return fs
      .readdirSync(CONFIG.RESOURCES_DIR)
      .filter((file) => file.toLowerCase().endsWith(".pdf"))
      .map((file) => path.join(CONFIG.RESOURCES_DIR, file));
  } catch (error) {
    console.error("Error reading resources folder:", error);
    return [];
  }
};

const loadMultiplePDFs = async (state: typeof StateAnnotation.State) => {
  console.log(
    "Loading PDFs from resources folder (with incremental cache updates)..."
  );

  const pdfPaths: string[] = getResourcePDFs();

  if (pdfPaths.length === 0) {
    console.log("No PDFs found in resources folder");
    return createStateUpdate(state, { pdfContent: "No PDFs found" });
  }

  console.log(
    `Found ${pdfPaths.length} PDFs to process:`,
    pdfPaths.map((p) => path.basename(p))
  );

  // Get existing cache data for incremental updates
  const existingCache: CacheData | null = globalCache;
  const existingChunks: DocumentChunk[] =
    existingCache?.chunksWithMetadata || [];

  // Process each PDF independently and return results
  const pdfPromises: Promise<PDFCacheData>[] = pdfPaths.map(async (pdfPath) => {
    const filename: string = path.basename(pdfPath);
    const existingDoc = existingCache?.documents.find(
      (doc) => doc.filename === filename
    );

    try {
      // Check if we can use existing cache for this document
      if (existingDoc && existingCache) {
        const currentStats: fs.Stats = fs.statSync(pdfPath);
        const currentSize: number = currentStats.size;
        const currentModified: string = currentStats.mtime.toISOString();

        // If metadata unchanged, use existing chunks
        if (
          existingDoc.size === currentSize &&
          existingDoc.lastModified === currentModified
        ) {
          console.log(`Using cached data for ${filename}`);
          const existingChunksForDoc = existingChunks.filter(
            (chunk) =>
              chunk.metadata.sourceName === path.basename(pdfPath, ".pdf")
          );

          console.log(
            `Used ${existingChunksForDoc.length} cached chunks from ${filename}`
          );

          return {
            chunks: existingChunksForDoc,
            content: existingChunksForDoc
              .map(
                (chunk) =>
                  `\n\n=== ${chunk.metadata.sourceName} (Page ${chunk.metadata.page}) ===\n${chunk.content}`
              )
              .join(""),
            filename,
            fromCache: true,
          };
        }
      }

      // Load and process new/changed PDF
      console.log(`Loading ${filename}...`);
      const loader: PDFLoader = new PDFLoader(pdfPath);
      const docs = await loader.load();

      const sourceName: string = path.basename(pdfPath, ".pdf");
      const priority: number = getDocumentPriority(sourceName);

      let validPages: number = 0;
      const newChunks: DocumentChunk[] = [];
      let newContent: string = "";

      docs.forEach((doc, pageIndex) => {
        if (doc.pageContent && doc.pageContent.trim().length > 10) {
          // Only include pages with substantial content
          newContent += `\n\n=== ${sourceName} (Page ${pageIndex + 1}) ===\n${
            doc.pageContent
          }`;

          // store metadata now, process later
          newChunks.push({
            content: doc.pageContent,
            metadata: {
              source: pdfPath,
              sourceName,
              page: pageIndex + 1,
              priority,
            },
          });
          validPages++;
        }
      });

      console.log(
        `Loaded ${validPages} valid pages from ${filename} (Priority: ${priority})`
      );

      return {
        chunks: newChunks,
        content: newContent,
        filename,
        fromCache: false,
      };
    } catch (error) {
      console.error(`Failed to load ${filename}:`, error.message);
      // Return empty result for failed PDFs
      return {
        chunks: [],
        content: "",
        filename,
        fromCache: false,
        error: error.message,
      };
    }
  });

  // Wait for all PDFs to process and combine results
  const results: PDFCacheData[] = await Promise.all(pdfPromises);

  // Combine all results
  const allDocumentChunks: DocumentChunk[] = results.flatMap((r) => r.chunks);
  const combinedContent: string = results.map((r) => r.content).join("");

  // Log summary
  const fromCache: number = results.filter((r) => r.fromCache).length;
  const newlyProcessed: number = results.filter(
    (r) => !r.fromCache && !r.error
  ).length;
  const failed: number = results.filter((r) => r.error).length;

  console.log(
    `Processing complete: ${fromCache} from cache, ${newlyProcessed} newly processed, ${failed} failed`
  );

  return createStateUpdate(state, {
    pdfContent: combinedContent,
    documentChunks: allDocumentChunks,
  });
};

const chunkTextWithMetadata = async (state: typeof StateAnnotation.State) => {
  console.log("Chunking text with metadata preservation...");

  if (
    !state.pdfContent ||
    !state.documentChunks ||
    state.documentChunks.length === 0
  ) {
    console.log("No PDF content to chunk");
    return createStateUpdate(state, { chunks: [], chunksWithMetadata: [] });
  }

  // Validate that we have substantial content
  const totalContentLength = state.documentChunks.reduce(
    (sum, doc) => sum + doc.content.length,
    0
  );
  if (totalContentLength < 100) {
    console.log("Insufficient content to chunk (too short)");
    return createStateUpdate(state, { chunks: [], chunksWithMetadata: [] });
  }

  try {
    const splitter = new RecursiveCharacterTextSplitter({
      chunkSize: CONFIG.CHUNK_SIZE,
      chunkOverlap: CONFIG.CHUNK_OVERLAP,
      separators: ["\n\n", "\n", ".", "!", "?", " "],
    });

    let allChunksWithMetadata: DocumentChunk[] = [];
    let allChunks: string[] = [];

    // Process each document separately to preserve metadata
    for (const docChunk of state.documentChunks) {
      const chunks = await splitter.splitText(docChunk.content);

      chunks.forEach((chunk, chunkIndex) => {
        const chunkWithMetadata: DocumentChunk = {
          content: chunk,
          metadata: {
            ...docChunk.metadata,
          },
        };

        allChunksWithMetadata.push(chunkWithMetadata);
        allChunks.push(chunk);
      });
    }

    console.log(
      `Text split into ${allChunks.length} chunks from ${state.documentChunks.length} documents`
    );

    return createStateUpdate(state, {
      chunks: allChunks,
      chunksWithMetadata: allChunksWithMetadata,
    });
  } catch (error) {
    console.error("Error chunking text:", error);
    return createStateUpdate(state, { chunks: [], chunksWithMetadata: [] });
  }
};

// I wanted to use a proper vector store like ChromaDB (embedded, docker based or with pip install), FaissStore, HNSWLib, but I couldn't get them to work.
// So I'm using an in-memory vector store for now.
const createVectorStore = async (state: typeof StateAnnotation.State) => {
  if (!getFeatureFlags().ENABLE_VECTOR_EMBEDDINGS) {
    console.log("Vector embeddings disabled - skipping vector store creation");
    return createStateUpdate(state, { vectorStore: null });
  }

  if (!state.chunks || state.chunks.length === 0) {
    console.log("No chunks to embed");
    return createStateUpdate(state, { vectorStore: null });
  }

  try {
    // Validate OpenAI API key before attempting to use it
    if (!process.env.OPENAI_API_KEY) {
      console.log("No OpenAI API key found - falling back to keyword search");
      return createStateUpdate(state, { vectorStore: null });
    }

    const embeddings = new OpenAIEmbeddings({
      openAIApiKey: process.env.OPENAI_API_KEY,
    });

    const documents = state.chunks.map((chunk, index) => ({
      pageContent: chunk,
      metadata: { index },
    }));

    console.log(`Creating embeddings for ${documents.length} chunks...`);
    const vectorStore = await MemoryVectorStore.fromDocuments(
      documents,
      embeddings
    );

    console.log("Vector store created successfully!");

    return createStateUpdate(state, { vectorStore });
  } catch (error) {
    console.error("Error creating vector store:", error);

    // Check if it's an authentication error
    if (error.message && error.message.includes("Invalid API Key")) {
      console.log("Invalid OpenAI API key - falling back to keyword search");
    } else {
      console.log("OpenAI API error - falling back to keyword search");
    }

    return createStateUpdate(state, { vectorStore: null });
  }
};

const saveCache = async (state: typeof StateAnnotation.State) => {
  if (state.cacheUsed) {
    // cache is up to date, so no need to save it
    return state;
  }

  console.log("Saving multi-document cache...");

  try {
    const currentPdfPaths: string[] = getResourcePDFs();

    // Process document info with better error handling
    const documents = currentPdfPaths.map((pdfPath) => {
      try {
        const filename: string = path.basename(pdfPath);
        const stats: fs.Stats = fs.statSync(pdfPath);

        return {
          path: pdfPath,
          filename,
          size: stats.size,
          pageCount:
            state.chunksWithMetadata?.filter(
              (chunk) =>
                chunk.metadata.sourceName === path.basename(pdfPath, ".pdf")
            ).length || 0,
          created: stats.birthtime.toISOString(),
          lastModified: stats.mtime.toISOString(),
          priority: getDocumentPriority(filename),
        };
      } catch (error) {
        console.error(
          `Failed to process document info for ${pdfPath}:`,
          error.message
        );
        // Return a fallback object for failed documents
        return {
          path: pdfPath,
          filename: path.basename(pdfPath),
          size: 0,
          pageCount: 0,
          created: new Date().toISOString(),
          lastModified: new Date().toISOString(),
          priority: 1,
        };
      }
    });

    const cacheData: CacheData = {
      documents,
      chunks: state.chunks,
      chunksWithMetadata: state.chunksWithMetadata || [],
      totalChunks: state.chunks.length,
      timestamp: new Date().toISOString(),
      hasVectorStore: !!state.vectorStore,
    };

    // No need to await this.
    const writeCache = () =>
      fs.promises.writeFile(
        CONFIG.CACHE_PATH,
        JSON.stringify(cacheData, null, 2)
      );
    writeCache().catch((error) => {
      console.error("Failed to save cache:", error);
      // Does not make sense to stop the process here. The updated cache will be in memory until the program ends.
      // Even if the cache is not saved, when the program restarts, the actually outdated cache loaded from disk will be identified and discarded.
      retryWithBackoff("saveCache", writeCache);
    });

    // Update in-memory cache
    globalCache = cacheData;

    console.log("Cache saved successfully:");
    console.log(`  ${documents.length} documents`);
    console.log(`  ${cacheData.totalChunks} total chunks`);
    console.log(
      `  Vector store: ${
        cacheData.hasVectorStore ? "included" : "not created"
      }`
    );
  } catch (error) {
    console.error("Failed to save cache:", error);
  }

  return state;
};

const searchRelevantChunks = async (state: typeof StateAnnotation.State) => {
  console.log("Searching for relevant chunks...");

  if (!state.chunks || state.chunks.length === 0) {
    console.log("No chunks to search");
    return createStateUpdate(state, { relevantChunks: [] });
  }

  const userQuestion = extractUserQuestion(state.messages);

  // Validate user question
  if (!userQuestion || userQuestion.trim().length < 3) {
    console.log(
      "Invalid question - too short or empty, returning first few chunks"
    );
    return createStateUpdate(state, {
      relevantChunks: state.chunks.slice(0, CONFIG.KEYWORD_SEARCH_TOP_K),
    });
  }

  console.log(`User question: "${userQuestion}"`);

  // Enhanced keyword extraction with better filtering
  const stopWords: Set<string> = new Set([
    "what",
    "how",
    "when",
    "where",
    "why",
    "can",
    "will",
    "does",
    "do",
    "is",
    "are",
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
  ]);

  const questionKeywords: string[] = userQuestion
    .toLowerCase()
    .replace(/[^\w\s]/g, " ") // Remove punctuation
    .split(/\s+/)
    .filter((word) => word.length > 2 && !stopWords.has(word));

  console.log(`Looking for keywords: ${questionKeywords.join(", ")}`);

  // Enhanced chunk scoring with better relevance calculation
  // Consider this a PoC for the assignment. Not a real scoring system.
  const scoredChunks: ChunkWithScore[] = state.chunks.map((chunk, index) => {
    const chunkLower: string = chunk.toLowerCase();
    let score: number = 0;
    let keywordsFound: number = 0;
    let exactWordMatches: number = 0;

    questionKeywords.forEach((keyword) => {
      const matches: number = (chunkLower.match(new RegExp(keyword, "g")) || [])
        .length;

      if (matches > 0) {
        keywordsFound++; // Count unique keywords found
        score += matches; 

        // Bonus for exact word boundaries (not just substring)
        const wordBoundaryRegex: RegExp = new RegExp(`\\b${keyword}\\b`, "gi");
        exactWordMatches += (
          chunkLower.match(wordBoundaryRegex) || []
        ).length;
      }
    });

    score += (exactWordMatches + keywordsFound) * 2; // More marks 
    return { chunk, score, index };
  });

  // Get top relevant chunks
  const relevantChunks: string[] = scoredChunks
    .filter((item) => item.score > 0)
    .sort((a, b) => b.score - a.score)
    .slice(0, CONFIG.KEYWORD_SEARCH_TOP_K)
    .map((item) => item.chunk);

  // If no relevant chunks found, return first few chunks as fallback
  // Ideally we can have a counter for each chunk that gets referenced by user queries. For these kind of scenarios, we can return the most referenced chunks.
  if (relevantChunks.length === 0) {
    console.log("No keyword matches found");
    return createStateUpdate(state, {
      relevantChunks: state.chunks.slice(0, CONFIG.KEYWORD_SEARCH_TOP_K),
      userQuestion,
    });
  }

  console.log(`Found ${relevantChunks.length} relevant chunks`);

  return createStateUpdate(state, {
    relevantChunks,
    userQuestion,
  });
};

const vectorSearchWithSources = async (state: typeof StateAnnotation.State) => {
  const userQuestion = extractUserQuestion(state.messages);

  if (!userQuestion || userQuestion.trim().length < 3) {
    console.log(
      "Invalid question for vector search, falling back to keyword search"
    );
    return await searchRelevantChunks(state);
  }

  if (!state.chunks || state.chunks.length === 0) {
    console.log("No chunks available for search");
    return createStateUpdate(state, { relevantChunks: [] });
  }

  if (state.vectorStore && getFeatureFlags().ENABLE_VECTOR_EMBEDDINGS) {
    try {
      console.log(`Vector searching for: "${userQuestion}"`);

      const results = await state.vectorStore.similaritySearch(
        userQuestion,
        CONFIG.VECTOR_SEARCH_TOP_K
      );

      if (!results || results.length === 0) {
        console.log(
          "Vector search returned no results, falling back to keyword search"
        );
        return await searchRelevantChunks(state);
      }

      // Map results back to chunks with metadata
      const relevantChunksWithMetadata: DocumentChunk[] = results
        .map((result) => {
          const matchingChunk = state.chunksWithMetadata?.find(
            (chunk) => chunk.content === result.pageContent
          );
          return matchingChunk;
        })
        .filter(Boolean) as DocumentChunk[];

      console.log(
        `Vector search found ${relevantChunksWithMetadata.length} relevant chunks`
      );

      // Log sources for debugging
      const sourceBreakdown = relevantChunksWithMetadata.reduce(
        (acc, chunk) => {
          const source = chunk.metadata.sourceName;
          acc[source] = (acc[source] || 0) + 1;
          return acc;
        },
        {} as Record<string, number>
      );

      console.log("Sources found:", sourceBreakdown);

      return createStateUpdate(state, {
        relevantChunks: relevantChunksWithMetadata.map((c) => c.content),
        userQuestion,
        chunksWithMetadata: relevantChunksWithMetadata,
      });
    } catch (error) {
      console.error("Vector search failed:", error);
      console.log("Falling back to keyword search...");
      return await searchRelevantChunks(state);
    }
  }

  // Fallback to keyword search
  console.log("Vector store not available, using keyword search");
  return await searchRelevantChunks(state);
};

const invokeAndPrint = async (state: typeof StateAnnotation.State) => {
  console.log("Generating answer with context...");

  // Build context with source attribution
  let contextText = "";
  if (state.relevantChunks && state.relevantChunks.length > 0) {
    contextText = "Here are the relevant rules:\n\n";

    state.relevantChunks.forEach((chunk, index) => {
      const chunkMetadata = state.chunksWithMetadata?.[index];
      const sourceInfo = chunkMetadata
        ? `[Source: ${chunkMetadata.metadata.sourceName}, Page ${chunkMetadata.metadata.page}]`
        : `[Source: Unknown]`;

      contextText += `${chunk}\n${sourceInfo}\n\n---\n\n`;
    });
  }

  // Enhanced system prompt that handles conflicts naturally
  const systemWithContext = new SystemMessage(
    `You are a helpful Pathfinder rules expert. 
Use the provided rules context to answer questions accurately.

${contextText}

IMPORTANT INSTRUCTIONS:
- If multiple sources provide different information about the same topic, acknowledge both perspectives
- When citing rules, mention the source document (e.g., "According to the Core Rules..." or "The Advanced Player Guide states...")
- If sources conflict, explain both versions and note which might take precedence
- If the context doesn't contain enough information to answer the question, say so
- Be clear about which source each piece of information comes from`
  );

  const userQuestion = state.userQuestion || "How can I help you?";

  try {
    const agentNextState = await agent.invoke(
      {
        messages: [systemWithContext, new HumanMessage(userQuestion)],
      },
      { configurable: { thread_id: "42" } }
    );

    const aiResponse = agentNextState.messages.at(-1)?.content;
    if (!aiResponse) {
      throw new Error("AI agent returned no response");
    }

    console.log("=== AI RESPONSE ===");
    console.log(aiResponse);
    console.log("==================");

    // Show source breakdown for transparency
    if (state.chunksWithMetadata && state.chunksWithMetadata.length > 0) {
      console.log("\n=== SOURCES USED ===");
      const sources = state.chunksWithMetadata.reduce((acc, chunk) => {
        const source = chunk.metadata.sourceName;
        const page = chunk.metadata.page;
        if (!acc[source]) acc[source] = [];
        if (!acc[source].includes(page)) acc[source].push(page);
        return acc;
      }, {} as Record<string, number[]>);

      Object.entries(sources).forEach(([source, pages]) => {
        console.log(
          `${source}: Pages ${pages.sort((a, b) => a - b).join(", ")}`
        );
      });
      console.log("==================");
    }

    return createStateUpdate(state, {
      messages: agentNextState.messages,
      sentiment: "positive",
    });
  } catch (error) {
    console.error("Error in invokeAndPrint:", error);
    return createStateUpdate(state, {
      messages: state.messages,
      sentiment: "error",
    });
  }
};

// State definition
const StateAnnotation = Annotation.Root({
  sentiment: Annotation<string>({
    value: (left: string, right: string) => right,
    default: () => "neutral",
  }),
  messages: Annotation<BaseMessage[]>({
    reducer: (left: BaseMessage[], right: BaseMessage | BaseMessage[]) => {
      if (Array.isArray(right)) {
        return left.concat(right);
      }
      return left.concat([right]);
    },
    default: () => [],
  }),
  pdfContent: Annotation<string>({
    value: (left: string, right: string) => right,
    default: () => "",
  }),
  chunks: Annotation<string[]>({
    value: (left: string[], right: string[]) => right,
    default: () => [],
  }),
  relevantChunks: Annotation<string[]>({
    value: (left: string[], right: string[]) => right,
    default: () => [],
  }),
  userQuestion: Annotation<string>({
    value: (left: string, right: string) => right,
    default: () => "",
  }),
  cacheUsed: Annotation<boolean>({
    value: (left: boolean, right: boolean) => right,
    default: () => false,
  }),
  vectorStore: Annotation<any>({
    value: (left: any, right: any) => right,
    default: () => null,
  }),
  documentChunks: Annotation<DocumentChunk[]>({
    value: (left: DocumentChunk[], right: DocumentChunk[]) => right,
    default: () => [],
  }),
  chunksWithMetadata: Annotation<DocumentChunk[]>({
    value: (left: DocumentChunk[], right: DocumentChunk[]) => right,
    default: () => [],
  }),
});

// Graph construction
const graphBuilder = new StateGraph(StateAnnotation);

// graph structure
const graph = graphBuilder
  .addNode("checkCache", checkCache)
  .addNode("loadPDF", loadMultiplePDFs)
  .addNode("chunkText", chunkTextWithMetadata)
  .addNode("createVectorStore", createVectorStore)
  .addNode("saveCache", saveCache)
  .addNode("vectorSearchChunks", vectorSearchWithSources)
  .addNode("invokeAndPrint", invokeAndPrint)
  .addEdge("__start__", "checkCache")
  .addConditionalEdges("checkCache", (state) =>
    state.cacheUsed ? "createVectorStore" : "loadPDF"
  )
  .addEdge("loadPDF", "chunkText")
  .addEdge("chunkText", "createVectorStore")
  .addEdge("createVectorStore", "saveCache")
  .addEdge("saveCache", "vectorSearchChunks")
  .addEdge("vectorSearchChunks", "invokeAndPrint")
  .addEdge("invokeAndPrint", "__end__")
  .compile();

// Main execution
const main = async () => {
  try {
    // Validate environment variables
    validateEnvironment();

    // Ensure required directories exist
    await ensureDirectoriesExist();

    console.log("Starting Pathfinder Rules Interrogation System");

    // Run the graph with a test question
    await graph.invoke({
      messages: [
        new SystemMessage("You are a helpful Pathfinder rules expert."),
        new HumanMessage("What equipment slot does a magical amulet use?"),
      ],
    });

    await graph.invoke({
      messages: [
        new SystemMessage("You are a helpful Pathfinder rules expert."),
        new HumanMessage(
          "Can you summarize the rules of the game for my 10 year old son?"
        ),
      ],
    });

    console.log("\nGraph execution completed successfully!");
  } catch (error) {
    console.error("Graph execution failed:", error);
    process.exit(1);
  }
};

// Run the main function
main();
