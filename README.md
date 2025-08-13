# Prerequisites
1. Create a Groq API key at [this link](https://console.groq.com/keys).
2. Copy the key and add it at the top of the `index.mts` file inside `process.env.GROQ_API_KEY`.

# Run the repo

```bash
pnpm install
pnpm dev
```


# Tech challenge

The repo contains a large PDF called `pathfinder_rule_book.pdf`. 
In `index.mts` there's a `LangGraph` graph that invokes an LLM model.

We want to interrogate the model and ask questions that are relevant to the player (i.e. "Summarize the rules of the game for me").

Focus on efficient solutions, optimize for speed and consider how the solution could be expanded to work with multiple documents at once.