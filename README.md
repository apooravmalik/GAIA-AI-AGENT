# Basic Agent Evaluation Runner

A LangGraph-based agent evaluation system that retrieves answers from a Supabase vector database and submits them to an evaluation API. This project combines semantic search with exact matching for optimal question-answering performance.

## Features

- **Hybrid Retrieval**: Combines exact dictionary lookup with semantic similarity search
- **Vector Database**: Uses Supabase with HuggingFace embeddings for document storage
- **LangGraph Integration**: Built with LangGraph for agent workflow management
- **Gradio Interface**: Web-based UI for easy interaction and evaluation
- **Automated Evaluation**: Fetches questions, runs agent, and submits answers automatically

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Gradio UI     │───▶│   BasicAgent     │───▶│   find_answer   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                        │
                                                        ▼
                                              ┌─────────────────┐
                                              │ Supabase Vector │
                                              │     Store       │
                                              └─────────────────┘
```

## Setup

### Prerequisites

- Python 3.8+
- Supabase account and project
- HuggingFace account (for deployment)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/apooravmalik/GAIA-AI-AGENT/
cd basic-agent-evaluation
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
```

Edit `.env` with your configuration:
```env
SUPABASE_URL=your_supabase_project_url
SUPABASE_SERVICE_KEY=your_supabase_service_key
SPACE_ID=your_huggingface_space_id
SPACE_HOST=your_huggingface_space_host
```

### Supabase Setup

1. Create a new Supabase project
2. Create a table called `docs` with the following structure:
```sql
CREATE TABLE docs (
  id SERIAL PRIMARY KEY,
  content TEXT NOT NULL,
  embedding VECTOR(768)  -- Adjust dimension based on your embedding model
);
```

3. Create a function for similarity search:
```sql
CREATE OR REPLACE FUNCTION match_documents(
  query_embedding VECTOR(768),
  match_threshold FLOAT DEFAULT 0.5,
  match_count INT DEFAULT 1
)
RETURNS TABLE (
  id INT,
  content TEXT,
  similarity FLOAT
)
LANGUAGE SQL STABLE
AS $$
  SELECT
    docs.id,
    docs.content,
    1 - (docs.embedding <=> query_embedding) AS similarity
  FROM docs
  WHERE 1 - (docs.embedding <=> query_embedding) > match_threshold
  ORDER BY docs.embedding <=> query_embedding
  LIMIT match_count;
$$;
```

## Usage

### Local Development

Run the application locally:
```bash
python app.py
```

The Gradio interface will be available at `http://localhost:7860`

### Deployment on HuggingFace Spaces

1. Create a new Space on HuggingFace
2. Upload your code files
3. Add your environment variables in the Space settings
4. The Space will automatically deploy and be accessible via the provided URL

### Using the Agent

1. **Login**: Click the "Login" button to authenticate with HuggingFace
2. **Run Evaluation**: Click "Run Evaluation & Submit All Answers" to:
   - Fetch questions from the evaluation API
   - Run your agent on each question
   - Submit all answers for scoring
   - Display results and performance metrics

## Code Structure

### `agent.py`
- **`find_answer()`**: Core retrieval function combining exact matching and semantic search
- **`build_graph()`**: Creates the LangGraph workflow
- **Vector Store Setup**: Configures Supabase vector store with HuggingFace embeddings

### `app.py`
- **`BasicAgent`**: Main agent class that orchestrates the question-answering process
- **`run_and_submit_all()`**: Handles the complete evaluation workflow
- **Gradio Interface**: Web UI for interaction and result display

## Key Components

### Retrieval Strategy

The agent uses a two-step retrieval approach:

1. **Exact Match**: First checks if the query exactly matches a stored question
2. **Semantic Search**: Falls back to embedding-based similarity search if no exact match

### Answer Extraction

The system intelligently extracts answers from stored content by looking for:
- `Final answer :` tags
- `Answer:` tags  
- Fallback to the last line of content

### Error Handling

Comprehensive error handling covers:
- Network timeouts and connection issues
- Invalid API responses
- Agent execution errors
- Missing environment variables

## Configuration

### Environment Variables

| Variable | Description | Required |
|----------|-------------|----------|
| `SUPABASE_URL` | Your Supabase project URL | Yes |
| `SUPABASE_SERVICE_KEY` | Supabase service role key | Yes |
| `SPACE_ID` | HuggingFace Space ID | For deployment |
| `SPACE_HOST` | HuggingFace Space host | For deployment |

### Embedding Model

The system uses `sentence-transformers/all-mpnet-base-v2` by default. You can modify this in `agent.py`:

```python
embeddings = HuggingFaceEmbeddings(model_name="your-preferred-model")
```

## Evaluation API

The system expects an evaluation API with the following endpoints:

- `GET /questions`: Returns list of questions with `task_id` and `question` fields
- `POST /submit`: Accepts submissions with username, agent_code, and answers

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and test thoroughly
4. Commit your changes: `git commit -am 'Add feature'`
5. Push to the branch: `git push origin feature-name`
6. Submit a pull request

## Troubleshooting

### Common Issues

**Supabase Connection Error**
- Verify your `SUPABASE_URL` and `SUPABASE_SERVICE_KEY`
- Check if your Supabase project is active

**Empty Results from Vector Search**
- Ensure your `docs` table has data
- Verify the embedding dimensions match your model

**API Submission Failures**
- Check if you're logged in to HuggingFace
- Verify the evaluation API is accessible
- Check network connectivity

### Debug Mode

Enable debug logging by setting:
```python
print(f"Debug info: {variable}")  # Add throughout your code
```

## License

[Add your chosen license here]

## Acknowledgments

- Built with [LangGraph](https://github.com/langchain-ai/langgraph)
- Uses [Supabase](https://supabase.com/) for vector storage
- Embeddings powered by [HuggingFace Transformers](https://huggingface.co/transformers/)
- UI created with [Gradio](https://gradio.app/)

---

For questions or support, please open an issue in this repository.
