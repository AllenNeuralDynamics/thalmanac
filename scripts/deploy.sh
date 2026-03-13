#!/bin/bash
set -euo pipefail

REPO_URL="https://github.com/AllenNeuralDynamics/thalmanac.git"
REPO_DIR="/opt/thalmanac"
COMPOSE_DIR="$REPO_DIR/environment"

echo "=== Step 1: Clone or pull repo ==="
if [ ! -d "$REPO_DIR" ]; then
  echo "Cloning repo..."
  git clone "$REPO_URL" "$REPO_DIR"
else
  echo "Pulling latest main..."
  cd "$REPO_DIR"
  git fetch origin
  git reset --hard origin/main
fi

echo "=== Step 2: Patch Dockerfile ==="
DOCKERFILE="$COMPOSE_DIR/Dockerfile"

# Remove code ocean capsule specific parameters from the Dockerfile to make it work in our EC2 environment
python3 - <<'EOF'
import re

dockerfile_path = "/opt/thalmanac/environment/Dockerfile"

with open(dockerfile_path, "r") as f:
    content = f.read()

# Replace the two-line RUN command with a simple RUN /postInstall
patched = re.sub(
    r'RUN --mount=type=secret,id=secrets \. /run/secrets/secrets \\\s*\n\s*&& /postInstall',
    'RUN /postInstall',
    content,
    flags=re.MULTILINE
)

with open(dockerfile_path, "w") as f:
    f.write(patched)

print("Dockerfile patched successfully")
EOF

echo "=== Step 3: Rebuild and restart docker compose ==="
cd "$COMPOSE_DIR"

# Stop only the streamlit container to avoid disrupting rclone mount
docker compose down streamlit || true
docker compose build streamlit
docker compose up -d streamlit

echo "=== Deploy complete ==="
