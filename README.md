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

Use `make-install` and fill out `.env` file, then this will run the evaluation comparing the bad and the better chunking methods: 

```bash
make eval
```


# Features and Improvements 

1. Comprehensive Evaluation Questions. I used your example questions as a template to generate more questions using the RAG system. This is not ideal as is it using an untested RAG system to generate the test set, so the test is obviously biased and neither are they verifiably good or useful questions. A better way would be to utilise knowledge from StackOne developers and customers to develop list of common questions targeting known areas where the RAG system struggles. Ideally, include a reference list of answers that provide the correct answer. 

2. Chunking logic. I have implemented a basic semantic chunking to split the json into more complete chunks. This is still not optimal as sometimes the information is cut at different levels of the hierarchy.

3. Knowledge graph representation of the json for RAG. A better representation of the json file might be a graph as it is more complete way to represent the structure. A graph RAG system would fetch the relevant nodes which might correspond to levels of the json hierarchy. 

4. Hyperparameter Optimisation. One could experiment with different parameters such as `k_neighbours` or `embedding_model` to gain extra performance. This could be multi-objective + Bayesian to effectively optimise for the multiple metrics given in `eval.py`. Other metrics of performance can also be included as this is a rather limited scope as is. 

5. Prompt engineering. Carry out some prompt engineering to give the LLM more context on the task.

6. Improve on the nearest neighbours search in the embedding model for context retrieval using something like BM-25 to rerank documents based on query term occurrence and rarity across the documents.

7. Make code ready for Production... Expand on the unit tests. Host API for the model so that the rate limits in the eval system are not painful to any end user. Make sure code is safe from prompt injection. 
