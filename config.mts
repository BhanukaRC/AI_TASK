import * as fs from 'fs';

export const CONFIG = {
  PDF_PATH: process.env.PDF_PATH || "./resources/documents.pdf",
  CACHE_PATH: process.env.CACHE_PATH || "./cache/pdf_cache.json",
  
  CHUNK_SIZE: parseInt(process.env.CHUNK_SIZE || "1000"),
  CHUNK_OVERLAP: parseInt(process.env.CHUNK_OVERLAP || "200"),
  
  VECTOR_SEARCH_TOP_K: parseInt(process.env.VECTOR_SEARCH_TOP_K || "3"),
  KEYWORD_SEARCH_TOP_K: parseInt(process.env.KEYWORD_SEARCH_TOP_K || "3"),
  RESPONSE_VALIDATION_THRESHOLD: parseFloat(process.env.RESPONSE_VALIDATION_THRESHOLD || "5"),
  RETRY_COUNT: parseInt(process.env.RETRY_COUNT || "3"),

  GROQ_MODEL: process.env.GROQ_MODEL || "llama-3.3-70b-versatile",
  
  RESOURCES_DIR: "./resources",
  CACHE_DIR: "./cache",
  
} as const;

export const getFeatureFlags = () => ({
  ENABLE_VECTOR_EMBEDDINGS: Boolean(process.env.OPENAI_API_KEY),
  DISABLE_GROQ_QUESTION_RELAVANCE_CHECK: Boolean(process.env.DISABLE_GROQ_QUESTION_RELAVANCE_CHECK),
});

export const validateEnvironment = () => {
  if (!process.env.GROQ_API_KEY) {
    throw new Error('GROQ_API_KEY environment variable is required');
  }
  
  if (process.env.OPENAI_API_KEY) {
    console.log('OpenAI API key found - vector embeddings enabled');
  } else {
    console.log('No OpenAI API key found - falling back to keyword search only');
  }
};

export const ensureDirectoriesExist: () => Promise<void> = async () => {
  const dirs = [CONFIG.RESOURCES_DIR, CONFIG.CACHE_DIR];
  await Promise.all(dirs.map(async (dir) => {
    if (!fs.existsSync(dir)) {
      await fs.promises.mkdir(dir, { recursive: true });
      console.log(`Created directory: ${dir}`);
    }
  }));
};
