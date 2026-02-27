# clusteringForVarietyIdentification
Python module to identify and label crop varieties using genotyping data

The code in this repository was developed by IDM to support our research into seed system capacity. We've made it publicly available under the MIT License to provide others with a better understanding of our research and an opportunity to build upon it for their own work. We make no representations that the code works as intended or that we will provide support, address issues that are found, or accept pull requests. You are welcome to create your own fork and modify the code to suit your own modeling needs as contemplated under the MIT License.

To run the code, you will need counts data from a panel of SNPs, a corresponding metadata file with sample information, and a json with parameter values. Example data and instructions for how to run the pipeline can be found in the tutorial folder.

## Running with Docker

Running the analysis inside Docker ensures reproducible results across different machines and architectures (e.g. Apple Silicon vs x86_64). The Dockerfile targets `linux/amd64` (Debian Bookworm) and pins all dependency versions via `constraints.txt`. It also sets `NUMBA_CPU_NAME=generic` and single-threaded execution to eliminate CPU-specific JIT and threading non-determinism in the numba/UMAP stack.

### Prerequisites

Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) (macOS / Windows) or [Docker Engine](https://docs.docker.com/engine/install/) (Linux).

### Build the image

```bash
docker build --platform linux/amd64 -t clustering-analysis .
```

### Run the tutorial example

```bash
docker run --platform linux/amd64 \
  -v $(pwd)/tutorial:/data -w /data \
  clustering-analysis \
  -c "import sys; sys.path.insert(0, '/app'); from base import runPipeline; runPipeline('/data/parametersRiceTutorial.json')"
```

The `-w /data` flag sets the container's working directory to `/data` so that relative file names in the parameters JSON (e.g. `"countsRiceTutorial.csv"`) resolve correctly. `sys.path.insert(0, '/app')` adds the application directory back to Python's module search path so that the pipeline modules can still be imported.

### Run with your own data

Place your counts CSV, metadata CSV, and parameters JSON in a local directory, then mount it to `/data`:

```bash
docker run --platform linux/amd64 \
  -v /path/to/your/data:/data -w /data \
  clustering-analysis \
  -c "import sys; sys.path.insert(0, '/app'); from base import runPipeline; runPipeline('/data/yourParameters.json')"
```

File paths in your parameters JSON can be either relative (e.g. `"counts.csv"`) or absolute with the `/data/` prefix (e.g. `"/data/counts.csv"`). Output files (plots and CSVs) will be written back to the mounted directory.

### Running outside Docker

To get results matching the Docker image on a bare-metal Linux (x86_64) machine, install dependencies with constraints and set the same environment variables:

```bash
pip install -r requirements.txt -c constraints.txt
export NUMBA_CPU_NAME=generic NUMBA_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
```