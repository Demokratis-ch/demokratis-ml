#!/bin/bash

set -o nounset
set -o errexit
set -o pipefail

DOCKER_ORG=vitawasalreadytaken
VERSION=${1:-""}

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    exit 1
fi

docker buildx build --platform linux/amd64 . -t $DOCKER_ORG/demokratis-ml:$VERSION
docker push $DOCKER_ORG/demokratis-ml:$VERSION
