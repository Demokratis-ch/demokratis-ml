# Data pipelines

We use [Prefect](https://www.prefect.io/) to orchestrate our data pipelines.

Our pipelines are:

- [preprocess_consultation_documents.py](./preprocess_consultation_documents.py) downloads consultation documents from the Demokratis API
  and turns them into a dataset conforming to [FullConsultationDocumentSchemaV1](../data/schemata.py).

Note that all commands from this readme are to be executed from the root directory of the repository.

## Setting up the Prefect instance
Run this to register custom block definitions, and repeat every time they change:

```
uv run prefect block register --file demokratis_ml/pipelines/blocks.py
```

Run this to create block documents (configured block instances), and repeat every time they change:

```
PYTHONPATH=. uv run --env-file=.env demokratis_ml/pipelines/create_blocks.py
```

Note that you will need some environment variables defined to correctly configure some secrets.
See the `.env.example` file in the repository root.


## Running pipelines
First, ensure that the Prefect server is running. In a local dev environment, this is done by running
this command in a spare terminal:

```
uv run prefect server start
```

The Prefect web interface will be available at http://127.0.0.1:4200.

Then, run the pipeline you need:

```
PYTHONPATH=. uv run demokratis_ml/pipelines/preprocess_consultation_documents.py
```
