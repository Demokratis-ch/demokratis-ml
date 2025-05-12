# Data pipelines

We use [Prefect](https://www.prefect.io/) to orchestrate our data pipelines.

Our pipelines are:

- [preprocess_consultation_documents.py](./preprocess_consultation_documents.py) downloads consultation documents from the Demokratis API
  and turns them into a dataset conforming to [FullConsultationDocumentSchemaV1](../data/schemata.py).

Note that all commands from this readme are to be executed from the root directory of the repository.

## Setting up the Prefect instance
Run this to register custom block definitions, and repeat every time they change:

```
uv run --env-file=.env prefect block register --file demokratis_ml/pipelines/blocks.py
```

Run this to create block documents (configured block instances), and repeat every time they change:

```
PYTHONPATH=. uv run --env-file=.env demokratis_ml/pipelines/create_blocks.py
```

Note that you will need some environment variables defined to correctly configure some secrets.
See the `.env.example` file in the repository root.


## Running pipelines locally
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

## Pipeline deployment
First, build a Docker image containing all demokratis_ml code:

```
YOUR_DOCKER_NAMESPACE=example-org
VERSION=0.1.0
docker buildx build --platform linux/amd64 . -t $YOUR_DOCKER_NAMESPACE/demokratis-ml:$VERSION
docker push $YOUR_DOCKER_NAMESPACE/demokratis-ml:$VERSION
```

(You can also use and modify our script for this: [scripts/build_docker_image.sh](../../scripts/build_docker_image.sh).)

Then ensure that whatever orchestrator is used runs a container from this image. This container must have the `PREFECT_API_URL` variable set, and run the script `flow_server.py` which serves all the pipelines (called flows in Prefect). For example, a Docker compose snippet:

```yaml
  prefect-flow-deployment:
    image: example-org/demokratis-ml:0.1.0
    command: uv run demokratis_ml/pipelines/flow_server.py
    environment:
      - PREFECT_API_URL=https://prefect.example.com/api
```
