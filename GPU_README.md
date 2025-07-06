# GPU Support - Deliberate Hard Fail Design

## Philosophy: GPU Must Work or Fail Clearly

This system uses a **deliberate hard-fail approach** for GPU support:

- ✅ **Single docker-compose.yml** with integrated GPU configuration
- ✅ **NVIDIA runtime required** - no fallback to CPU-only mode
- ✅ **Clear failure** if GPU/NVIDIA runtime unavailable

## Why Hard Fail?

1. **Prevents Silent Degradation**: CPU-only Ollama runs very slowly, creating poor user experience
2. **Clear Requirements**: Users know immediately if their system supports GPU acceleration
3. **No Complexity**: Single configuration file, no override complexity
4. **Production Ready**: Real deployments need GPU - this enforces it

## Prerequisites

```bash
# Install NVIDIA Docker runtime
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

## Deployment

```bash
# Will work with GPU or fail clearly
docker compose -f docker-compose.dev.yml up

# Expected behavior:
# ✅ With NVIDIA runtime: Ollama uses GPU acceleration
# ❌ Without NVIDIA runtime: Container fails to start with clear error
```

## Error Messages

If GPU support is not available, you'll see clear Docker errors:
- `Error response from daemon: could not select device driver "nvidia"`
- `nvidia-container-runtime not found`

This is **intentional behavior** - fix your GPU setup rather than running degraded.

## Verification

```bash
# Check GPU is being used
docker exec rag-ollama nvidia-smi

# Should show Ollama processes using GPU memory
```

## Migration from Old Dual-Compose Setup

- ❌ `docker-compose.gpu.yml` (removed)
- ✅ `docker-compose.dev.yml` (integrated GPU support)

**Old approach**: `docker compose -f docker-compose.dev.yml -f docker-compose.gpu.yml up`
**New approach**: `docker compose -f docker-compose.dev.yml up`