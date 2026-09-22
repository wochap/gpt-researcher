# Configuration

The config.py enables you to customize GPT Researcher to your specific needs and preferences.

Thanks to our amazing community and contributions, GPT Researcher supports multiple LLMs and Retrievers.
In addition, GPT Researcher can be tailored to various report formats (such as APA), word count, research iterations depth, etc.

GPT Researcher defaults to our recommended suite of integrations: [OpenAI](https://platform.openai.com/docs/overview) for LLM calls and [Tavily API](https://app.tavily.com) for retrieving real-time web information.

As seen below, OpenAI still stands as the superior LLM. We assume it will stay this way for some time, and that prices will only continue to decrease, while performance and speed increase over time.

<div style={{ marginBottom: '10px' }}>
<img align="center" height="350" src="/img/leaderboard.png" />
</div>

The default config.py file can be found in `/gpt_researcher/config/`. It supports various options for customizing GPT Researcher to your needs.
You can also include your own external JSON file `config.json` by adding the path in the `config_path` param.
The config JSON should follow the format/keys in the default config. Below is a sample config.json file to help get you started:
```json
{
  "RETRIEVER": "tavily",
  "EMBEDDING": "openai:text-embedding-3-small",
  "SIMILARITY_THRESHOLD": 0.42,
  "FAST_LLM": "openai:gpt-5.4-mini",
  "SMART_LLM": "openai:gpt-5.4",
  "STRATEGIC_LLM": "openai:gpt-5.4",
  "LANGUAGE": "english",
  "CURATE_SOURCES": false,
  "FAST_TOKEN_LIMIT": 3000,
  "SMART_TOKEN_LIMIT": 6000,
  "STRATEGIC_TOKEN_LIMIT": 4000,
  "BROWSE_CHUNK_MAX_LENGTH": 8192,
  "SUMMARY_TOKEN_LIMIT": 700,
  "TEMPERATURE": 0.4,
  "DOC_PATH": "./my-docs",
  "REPORT_SOURCE": "web"
}
```


For example, to start GPT-Researcher and specify a specific config you would do this:
```bash
python gpt_researcher/main.py --config_path my_config.json
```




 **Please follow the config.py file for additional future support**.

Below is a list of current supported options:

- **`RETRIEVER`**: Search engine or research retriever used for retrieving sources. Defaults to `tavily`. Options include `tavily`, `duckduckgo`, `bing`, `brave`, `google`, `searchapi`, `serper`, `serpapi`, `searx`, `arxiv`, `openalex`, `semantic_scholar`, `pubmed_central`, `exa`, `crw`, `groundroute`, `bocha`, `xquik`, `custom`, and `mcp`. You can also combine retrievers with commas, such as `tavily,openalex,semantic_scholar`. [Check here](https://github.com/assafelovic/gpt-researcher/tree/master/gpt_researcher/retrievers) for supported retrievers
- **`EMBEDDING`**: Embedding model. Defaults to `openai:text-embedding-3-small`. Options: `ollama`, `huggingface`, `azure_openai`, `custom`.
- **`SIMILARITY_THRESHOLD`**: Threshold value for similarity comparison when processing documents. Defaults to `0.42`.
- **`FAST_LLM`**: Model name for fast LLM operations such summaries. Defaults to `openai:gpt-5.4-mini`.
- **`SMART_LLM`**: Model name for smart operations like generating research reports and reasoning. Defaults to `openai:gpt-5.4`.
- **`STRATEGIC_LLM`**: Model name for strategic operations like generating research plans and strategies. Defaults to `openai:gpt-5.4`.
- **`LANGUAGE`**: Language to be used for the final research report. Defaults to `english`.
- **`CURATE_SOURCES`**: Whether to curate sources for research. This step adds an LLM run which may increase costs and total run time but improves quality of source selection. Defaults to `False`.
- **`FAST_TOKEN_LIMIT`**: Maximum token limit for fast LLM responses. Defaults to `3000`.
- **`SMART_TOKEN_LIMIT`**: Maximum token limit for smart LLM responses. Defaults to `6000`.
- **`STRATEGIC_TOKEN_LIMIT`**: Maximum token limit for strategic LLM responses. Defaults to `4000`.

#### Recommended values for modern long-output models

The default token limits are calibrated for GPT-4o-class models
(16k max output). For models with larger output capacity, increase
these limits to avoid truncated reports:

| Model family            | Max output | Recommended SMART_TOKEN_LIMIT |
|-------------------------|-----------:|------------------------------:|
| GPT-4o / GPT-4.1        |        16k |                          8000 |
| Claude Haiku 4.5        |        64k |                         16000 |
| Claude Sonnet 4.6       |        64k |                         16000 |
| Claude Opus 4.7         |       128k |                         32000 |
| GPT-5 family            |       128k |                         32000 |

Apply proportional values to `FAST_TOKEN_LIMIT` and
`STRATEGIC_TOKEN_LIMIT` if you use distinct models for those roles.
The hard upper bound is 200k (sanity guard against typos).

- **`BROWSE_CHUNK_MAX_LENGTH`**: Maximum length of text chunks to browse in web sources. Defaults to `8192`.
- **`SUMMARY_TOKEN_LIMIT`**: Maximum token limit for generating summaries. Defaults to `700`.
- **`TEMPERATURE`**: Sampling temperature for LLM responses, typically between 0 and 1. A higher value results in more randomness and creativity, while a lower value results in more focused and deterministic responses. Defaults to `0.4`.
- **`USER_AGENT`**: Custom User-Agent string for web crawling and web requests.
- **`MAX_SEARCH_RESULTS_PER_QUERY`**: Maximum number of search results to retrieve per query. Defaults to `5`.
- **`MEMORY_BACKEND`**: Backend used for memory operations, such as local storage of temporary data. Defaults to `local`.
- **`TOTAL_WORDS`**: Total word count limit for document generation or processing tasks. Defaults to `1200`.
- **`REPORT_FORMAT`**: Preferred format for report generation. Defaults to `APA`. Consider formats like `MLA`, `CMS`, `Harvard style`, `IEEE`, etc.
- **`MAX_ITERATIONS`**: Maximum number of iterations for processes like query expansion or search refinement. Defaults to `3`.
- **`AGENT_ROLE`**: Role of the agent. This configures the behavior of specialized research agents. Defaults to `None`. When set, it activates role-specific prompting and techniques tailored to particular research domains.
- **`MAX_SUBTOPICS`**: Maximum number of subtopics to generate or consider. Defaults to `3`.
- **`SCRAPER`**: Web scraper to use for gathering information. Defaults to `bs` (BeautifulSoup). You can also use [newspaper](https://github.com/codelucas/newspaper).
- **`MAX_SCRAPER_WORKERS`**: Maximum number of concurrent scraper workers per research. Defaults to `15`.
- **`REPORT_SOURCE`**: Source for the research report data. Defaults to `web` for online research. Can be set to `doc` for local document-based research. This determines where GPT Researcher gathers its primary information from.
- **`DOC_PATH`**: Path to read and research local documents. Defaults to `./my-docs`.
- **`PROMPT_FAMILY`**: The family of prompts and prompt formatting to use. Defaults to prompting optimized for GPT models. See the full list of options in [enum.py](https://github.com/assafelovic/gpt-researcher/blob/master/gpt_researcher/utils/enum.py#L56).
- **`LLM_KWARGS`**: Json formatted dict of additional keyword args to be passed to the LLM provider class when instantiating it. This is primarily useful for clients like Ollama that allow for additional keyword arguments such as `num_ctx` that influence the inference calls.
- **`EMBEDDING_KWARGS`**: Json formatted dict of additional keyword args to be passed to the embedding provider class when instantiating it.
- **`DEEP_RESEARCH_BREADTH`**: Controls the breadth of deep research, defining how many parallel paths to explore. Defaults to `3`.
- **`DEEP_RESEARCH_DEPTH`**: Controls the depth of deep research, defining how many sequential searches to perform. Defaults to `2`.
- **`DEEP_RESEARCH_CONCURRENCY`**: Controls the concurrency level for deep research operations. Defaults to `4`.
- **`REASONING_EFFORT`**: Controls the reasoning effort of strategic models. Default to `medium`.
- **`RETRIEVAL_PIPELINE`**: Which context retrieval pipeline to run after scraping. `default` keeps the built-in embedding filter. `local_gpu` enables configurable chunking, a cosine top-K stage and an optional reranker (see [Local GPU retrieval](#local-gpu-retrieval-ollama-embeddings--llama-server-reranker)). Defaults to `default`.
- **`COMPRESSION_CHUNK_SIZE`**: Chunk size for the text splitter in the `local_gpu` pipeline. Defaults to `2000`.
- **`COMPRESSION_CHUNK_OVERLAP`**: Chunk overlap for the text splitter in the `local_gpu` pipeline. Defaults to `200`.
- **`EMBEDDING_TOP_K`**: Number of chunks kept by cosine similarity before reranking. Defaults to `30`.
- **`EMBEDDING_BATCH_SIZE`**: Maximum texts per embedding request. `0` sends every chunk in one request. Defaults to `16`.
- **`RERANKER_ENABLED`**: Turns on the rerank stage of the `local_gpu` pipeline. Defaults to `False`.
- **`RERANKER_PROVIDER`**: Reranker backend. Only `llamacpp` (llama.cpp `llama-server --rerank`) is supported; any other value logs a warning and disables reranking. Defaults to `llamacpp`.
- **`RERANKER_BASE_URL`**: Base URL of the reranker server. Defaults to `http://localhost:8001`.
- **`RERANKER_ENDPOINT`**: Path of the rerank endpoint on that server. llama-server serves `/rerank`, `/v1/rerank` and `/v1/reranking`. Defaults to `/v1/rerank`.
- **`RERANKER_MODEL`**: Model name sent in the request. llama-server ignores it in single-model mode and uses it to pick a model in router mode. Defaults to `Qwen/Qwen3-Reranker-4B`.
- **`RERANKER_TOP_K`**: Number of reranked chunks handed to the LLM. Defaults to `8`.
- **`RERANKER_BATCH_SIZE`**: Documents per rerank request. Defaults to `8`.
- **`RERANKER_TIMEOUT`**: Timeout in seconds for each rerank request. Defaults to `30.0`.
- **`RERANKER_MAX_RETRIES`**: Retries per rerank batch on transient failures: HTTP 502, 503 or 504, connection errors, dropped connections and read timeouts. Waits use exponential backoff with jitter and honor `Retry-After`. Other errors are not retried. Defaults to `4`.
- **`RERANKER_RETRY_MAX_WAIT`**: Total time budget in seconds for the retries of one batch, counted from the first request. A retry whose wait would exceed the budget is skipped, and the batch falls back to embedding order. Defaults to `60.0`.
- **`RERANKER_INSTRUCTION`**: Task instruction embedded in the Qwen3 reranker prompt. Defaults to `Given a web search query, retrieve relevant passages that answer the query`.
- **`RERANKER_APPLY_QWEN3_TEMPLATE`**: Wrap the query and documents in the Qwen3-Reranker chat template on the client. Leave off for GGUFs that carry `tokenizer.chat_template.rerank`: llama-server applies that template itself and turning this on nests it twice (see [Templating](#templating)). Defaults to `False`.

## Deep Research Configuration

The deep research parameters allow you to fine-tune how GPT Researcher explores complex topics that require extensive knowledge gathering. These parameters work together to determine the thoroughness and efficiency of the research process:

- **`DEEP_RESEARCH_BREADTH`**: Controls how many parallel research paths are explored simultaneously. A higher value (e.g., 5) causes the researcher to investigate more diverse subtopics at each step, resulting in broader coverage but potentially less focus on core themes. The default value of `3` provides a balanced approach between breadth and depth.

- **`DEEP_RESEARCH_DEPTH`**: Determines how many sequential search iterations GPT Researcher performs for each research path. A higher value (e.g., 3-4) allows for following citation trails and diving deeper into specialized information, but increases research time substantially. The default value of `2` ensures reasonable depth while maintaining practical completion times.

- **`DEEP_RESEARCH_CONCURRENCY`**: Sets how many concurrent operations can run during deep research. Higher values speed up the research process on capable systems but may increase API rate limit issues or resource consumption. The default value of `4` is suitable for most environments, but can be increased on systems with more resources or decreased if you experience performance issues.

For academic or highly specialized research, consider increasing both breadth and depth (e.g., BREADTH=4, DEPTH=3). For quick exploratory research, lower values (e.g., BREADTH=2, DEPTH=1) will provide faster results with less detail.

To change the default configurations, you can simply add env variables to your `.env` file as named above or export manually in your local project directory.

For example, to manually change the search engine and report format:

```bash
export RETRIEVER=bing
export REPORT_FORMAT=IEEE
```

For academic literature reviews, you can combine web and scholarly retrievers:

```bash
export RETRIEVER=tavily,openalex,semantic_scholar
```

Please note that you might need to export additional env vars and obtain API keys for other supported search retrievers and LLM providers. Please follow your console logs for further assistance.
To learn more about additional LLM support you can check out the docs [here](/docs/gpt-researcher/llms/llms).

## Local GPU retrieval: Ollama embeddings + llama-server reranker

Set `RETRIEVAL_PIPELINE=local_gpu` to replace the built-in embedding filter with a two-stage retrieval pipeline designed for a single consumer GPU. The embedding model (Ollama) and the reranker (llama.cpp `llama-server`) stay resident on the GPU at the same time; llama.cpp only allocates weights plus KV cache, so a 4B reranker and a 4B embedding model fit together on 16 GB and a Q4 reranker fits next to the embedding model on 8 GB. Per sub-query:

```
scrape -> RecursiveCharacterTextSplitter(COMPRESSION_CHUNK_SIZE, COMPRESSION_CHUNK_OVERLAP)
       -> embeddings (Ollama, EMBEDDING_BATCH_SIZE per request) -> cosine top EMBEDDING_TOP_K
       -> reranker (llama-server RERANKER_ENDPOINT, RERANKER_BATCH_SIZE per request) -> top RERANKER_TOP_K -> LLM
```

With `RERANKER_ENABLED=false` the pipeline stops after the cosine top-K and behaves like the default compressor with configurable chunking. With `RETRIEVAL_PIPELINE=default` (the default) nothing changes.

If a rerank request fails (connection error, HTTP error, unexpected response shape) the pipeline logs a warning and hands the first `RERANKER_TOP_K` chunks in embedding order to the LLM, so a reranker outage degrades instead of failing the research run.

Embeddings use the existing `EMBEDDING=ollama:<model>` and `OLLAMA_BASE_URL` settings. Example `.env`:

```bash
RETRIEVAL_PIPELINE=local_gpu
EMBEDDING=ollama:qwen3-embedding:4b
OLLAMA_BASE_URL=http://localhost:11434
RERANKER_ENABLED=true
RERANKER_BASE_URL=http://localhost:8012
RERANKER_MODEL=Qwen/Qwen3-Reranker-4B
```

### Serving the reranker with llama-server

```bash
llama-server -m Qwen3-Reranker-4B.Q8_0.gguf --rerank --pooling rank \
  -c 4096 -ngl 99 --host 127.0.0.1 --port 8012
```

`--rerank` enables the endpoints, `--pooling rank` selects the classifier head, `-c` must cover the template plus query plus one chunk (`COMPRESSION_CHUNK_SIZE` characters), and `-ngl 99` offloads every layer. The server processes one document per slot; the default slot count is fine for `RERANKER_BATCH_SIZE=8`.

**The GGUF must be converted with llama.cpp's reranker path.** Such files carry the tensor `cls.output.weight [2560, 2]`, `qwen3.pooling_type = 4` and the metadata key `tokenizer.chat_template.rerank`. Generic Qwen3-Reranker conversions (plain causal-LM GGUFs) lack the classifier head and cannot rerank. Verified working: `giladgd/Qwen3-Reranker-4B-GGUF` and `Voodisss/Qwen3-Reranker-4B-GGUF-llama_cpp` on Hugging Face. Check a file with `llama-server --verbose` (the key list is printed at load) before relying on it.

Endpoint contract as verified against llama.cpp b9190: `POST /rerank`, `/v1/rerank` and `/v1/reranking` all accept `{"model", "query", "documents", "top_n"}` (`model` is ignored in single-model mode, `top_n` truncates the result list) and return `{"results": [{"index", "relevance_score"}, ...]}` sorted by score. An empty `documents` array is a 400.

### Templating

Qwen3-Reranker expects each query/document pair wrapped in a chat-style prompt ("Judge whether the Document meets the requirements..."). llama-server applies the GGUF's `tokenizer.chat_template.rerank` to every pair itself, substituting `{query}` and `{document}`. With `RERANKER_APPLY_QWEN3_TEMPLATE=true` the client wraps the text first and the server wraps it again, so the model sees the system prompt and the `<Query>:`/`<Document>:` markers twice. Observed with `giladgd/Qwen3-Reranker-4B-GGUF` Q8_0 on llama.cpp b9190 (query "What is the boiling point of water at sea level?"):

| Document | template off | template on (double) |
|---|---|---|
| exact answer (100 °C at sea level) | 0.998334 | 0.990064 |
| related (pressure vs boiling point) | 0.128992 | 0.156106 |
| weak (water is H2O) | 0.000058 | 0.000601 |
| off-topic (Eiffel Tower) | 0.000002 | 0.000021 |
| off-topic (Python lists) | 0.000001 | 0.000019 |

Both settings produce the same order for this set, but double templating compresses the spread between relevant and irrelevant documents. Keep the default (`false`) unless the GGUF lacks the rerank template. `RERANKER_INSTRUCTION` only takes effect when client-side templating is on; with the server template the instruction baked into the GGUF is used.

Scores from this build lie in `(0, 1)` and behave like a "yes" probability. The pipeline only uses them for ordering; nothing thresholds `rerank_score`, and other builds or GGUFs may return raw logits.

### Device profiles

**RTX 4060 8 GB (CUDA)**: Q4_K_M reranker (~2.5 GB) next to a 4B embedding model.

```bash
llama-server -m Qwen3-Reranker-4B.Q4_K_M.gguf --rerank --pooling rank -c 4096 -ngl 99 --port 8012
```

```bash
RERANKER_BASE_URL=http://localhost:8012
EMBEDDING_BATCH_SIZE=16
RERANKER_BATCH_SIZE=8
```

**RX 6800 XT 16 GB (ROCm, gfx1030)**: Q8_0 reranker (~4.3 GB) next to the embedding model. Measured: llama-server alone ~5.5 GB VRAM, plus `qwen3-embedding:4b` in Ollama ~10.8 GB total.

```bash
llama-server -m Qwen3-Reranker-4B.Q8_0.gguf --rerank --pooling rank -c 4096 -ngl 99 --port 8012
```

```bash
RERANKER_BASE_URL=http://localhost:8012
EMBEDDING_BATCH_SIZE=32
RERANKER_BATCH_SIZE=16
```

Use the ROCm build of Ollama for the embedding model on this card.

### Verifying the setup

```bash
curl http://localhost:11434/api/ps          # Ollama: loaded models and size_vram
curl http://localhost:8012/health           # llama-server: {"status":"ok"}
curl -X POST http://localhost:8012/v1/rerank -H 'Content-Type: application/json' \
  -d '{"query":"capital of France","documents":["Paris is the capital of France.","Bananas are yellow."]}'
nvidia-smi   # or rocm-smi: both models should be resident at once
```
