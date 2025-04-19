# AI Exercise - Retrieval

## Project requirements

### uv

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) to install and manage python dependencies.

### Docker Engine (optional)

Install [Docker Engine](https://docs.docker.com/engine/install/) to build and run the API's Docker image locally.

## Installation

```bash
make install
```

## API

The project includes an API built with [FastAPI](https://fastapi.tiangolo.com/). Its code can be found at `src/api`.

The API is containerized using a [Docker](https://docs.docker.com/get-started/) image, built from the `Dockerfile` and `docker-compose.yml` at the root. This is optional, you can also run the API without docker.

### Environment Variables

Copy .env_example to .env and fill in the values.

### Build and start the API

To build and start the API, use the following Makefile command:

```bash
make dev-api
```

you can also use `make start-api` to start the API using Docker.

## Frontend

The project includes a frontend built with [Streamlit](https://streamlit.io/). Its code can be found at `demo`.

Run the frontend with:

```bash
make start-app
```

## Testing

To run unit tests, run `pytest` with:

```bash
make test
```

## Formatting and static analysis

There is some preset up formatting and static analysis tools to help you write clean code. check the make file for more details.

```bash
make lint
```

```bash
make format
```

```bash
make typecheck
```

# Get Started

Have a look in `ai_exercise/constants.py`. Then check out the server routes in `ai_exercise/main.py`. 

1. Load some documents by calling the `/load` endpoint. Does the system work as intended? Are there any issues?

2. Find some method of evaluating the quality of the retrieval system.

3. See how you can improve the retrieval system. Some ideas:
- Play with the chunking logic
- Try different embeddings models
- Other types of models which may be relevant
- How else could you store the data for better retrieval?


-------------
# Running an Evaluation

```bash
make eval
```


# Features and Improvements 

1. Comprehensive Evaluation Questions. I used your example questions as a template to generate more questions using the RAG system. This is not ideal as is it using an untested RAG system to generate the test set, so the test is obviously biased and neither are they verifiably good or useful questions. A better way would be to utilise knowledge from StackOne developers and customers to develop list of common questions targeting known areas where the RAG system struggles. Ideally, include a reference list of answers that provide the correct answer. 

2. Improve on the chunking logic

3. Carry out a Hyperparameter Optimisation on parameters such as `k_neighbours` or `embedding_model` to gain extra performance. This could be multi-objective to optimise for the multiple metrics given in `eval.py`. In a similar vein, include more metrics of performance. 

4. Carry out some prompt engineering to give the LLM more context on the task.

5. Improve on the nearest neighbours search in the embedding model for context reterival 
- BM-25 + re-ranker

6. Expand on the unit tests. 

7. knowledge graph representation of the json for rag
