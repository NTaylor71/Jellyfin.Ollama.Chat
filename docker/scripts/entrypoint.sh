#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}🚀 Starting Jelly service with unified model management${NC}"

# Function to log with timestamp
log() {
    echo -e "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Wait for dependencies to be ready
wait_for_dependency() {
    local service_name=$1
    local check_command=$2
    local max_attempts=30
    local attempt=0

    log "${YELLOW}Waiting for $service_name to be ready...${NC}"
    
    while ! eval "$check_command" >/dev/null 2>&1; do
        attempt=$((attempt + 1))
        if [ $attempt -gt $max_attempts ]; then
            log "${RED}❌ $service_name is not ready after $max_attempts attempts${NC}"
            return 1
        fi
        log "  Attempt $attempt/$max_attempts..."
        sleep 2
    done
    
    log "${GREEN}✅ $service_name is ready${NC}"
    return 0
}

# Wait for Redis (if configured)
if [ -n "${REDIS_HOST:-}" ]; then
    if command_exists redis-cli; then
        wait_for_dependency "Redis" "redis-cli -h ${REDIS_HOST} -p ${REDIS_PORT:-6379} ping"
    else
        log "${YELLOW}⚠️  redis-cli not available, skipping Redis check${NC}"
    fi
fi

# Wait for MongoDB (if configured)
if [ -n "${MONGODB_HOST:-}" ]; then
    if command_exists mongosh; then
        wait_for_dependency "MongoDB" "mongosh --host ${MONGODB_HOST}:${MONGODB_PORT:-27017} --eval 'db.adminCommand(\"ping\")'"
    elif command_exists mongo; then
        wait_for_dependency "MongoDB" "mongo --host ${MONGODB_HOST}:${MONGODB_PORT:-27017} --eval 'db.adminCommand(\"ping\")'"
    else
        log "${YELLOW}⚠️  MongoDB client not available, skipping MongoDB check${NC}"
    fi
fi

# Ensure Python path is set
export PYTHONPATH="${PYTHONPATH:-/app}"

# Only run model management for WORKER containers, not API
if [ "${CONTAINER_TYPE:-}" = "worker" ]; then
    log "${BLUE}📦 Ensuring all required models are available...${NC}"

# Validate volume mounts
validate_volume_mounts() {
    local models_path="/app/models"
    
    log "${BLUE}🔍 Validating volume mounts...${NC}"
    
    if [ ! -d "$models_path" ]; then
        log "${YELLOW}⚠️  Creating models directory: $models_path${NC}"
        mkdir -p "$models_path"
    fi
    
    # Check if models directory is writable
    if [ ! -w "$models_path" ]; then
        log "${RED}❌ Models directory is not writable: $models_path${NC}"
        return 1
    fi
    
    # Check available space (at least 5GB recommended)
    available_space=$(df "$models_path" | awk 'NR==2 {print $4}')
    available_gb=$((available_space / 1024 / 1024))
    
    if [ $available_gb -lt 5 ]; then
        log "${YELLOW}⚠️  Low disk space in models directory: ${available_gb}GB available${NC}"
        log "${YELLOW}    Recommend at least 5GB for all models${NC}"
    else
        log "${GREEN}✅ Sufficient disk space: ${available_gb}GB available${NC}"
    fi
    
    return 0
}

# Display model inventory before download
show_model_inventory() {
    log "${BLUE}📊 Checking current model inventory...${NC}"
    
    # Use the enhanced manage_models.py tool with error handling
    if python manage_models.py status --models-path /app/models --json > /tmp/model_status.json 2>/tmp/model_error.log; then
        # Parse and display key metrics only if JSON is valid
        if command_exists jq && jq empty /tmp/model_status.json 2>/dev/null; then
            local total_models=$(jq -r '.total_models' /tmp/model_status.json)
            local available_models=$(jq -r '.available_models' /tmp/model_status.json)
            local required_models=$(jq -r '.required_models' /tmp/model_status.json)
            local total_size=$(jq -r '.total_size_mb' /tmp/model_status.json)
            
            log "${BLUE}📈 Model Status: ${available_models}/${required_models} required models available${NC}"
            log "${BLUE}💾 Total model storage: ${total_size} MB${NC}"
            
            # Show missing models
            local missing_models=$(jq -r '.models | to_entries[] | select(.value.status == "missing" and .value.required == true) | .value.name' /tmp/model_status.json 2>/dev/null | tr '\n' ' ')
            if [ -n "$missing_models" ]; then
                log "${YELLOW}📥 Missing required models: $missing_models${NC}"
            fi
        else
            log "${YELLOW}⚠️  Invalid JSON response from model status check${NC}"
        fi
    else
        log "${YELLOW}⚠️  Could not get detailed model status${NC}"
        if [ -f /tmp/model_error.log ]; then
            log "${RED}📋 Error: $(head -1 /tmp/model_error.log)${NC}"
        fi
    fi
    
    rm -f /tmp/model_status.json /tmp/model_error.log
}

# Validate volume mounts
if ! validate_volume_mounts; then
    log "${RED}❌ Volume mount validation failed${NC}"
    if [ "${FAIL_ON_MODEL_ERROR:-true}" = "true" ]; then
        exit 1
    fi
fi

# Show current model inventory
show_model_inventory

# Run model management with progress reporting
log "${BLUE}📦 Running unified model management...${NC}"

# Check if we need to download models first
if python manage_models.py check --models-path /app/models --json > /tmp/model_check.json 2>/tmp/model_check_error.log; then
    if command_exists jq && jq empty /tmp/model_check.json 2>/dev/null; then
        missing_required=$(jq -r '.models | to_entries[] | select(.value.status == "missing" and .value.required == true) | .key' /tmp/model_check.json 2>/dev/null | wc -l)
        
        if [ "$missing_required" -gt 0 ]; then
            log "${YELLOW}📥 Found $missing_required missing required models, starting download...${NC}"
            
            # Show estimated download size
            total_download_size=$(jq -r '.models | to_entries[] | select(.value.status == "missing" and .value.required == true) | .value.size_mb' /tmp/model_check.json 2>/dev/null | awk '{sum+=$1} END {print sum}')
            if [ -n "$total_download_size" ] && [ "$total_download_size" != "" ]; then
                log "${BLUE}💾 Estimated download size: ${total_download_size} MB${NC}"
            fi
        else
            log "${GREEN}✅ All required models already available${NC}"
        fi
    else
        log "${YELLOW}⚠️  Model check returned invalid JSON${NC}"
    fi
else
    log "${YELLOW}⚠️  Model check failed${NC}"
    if [ -f /tmp/model_check_error.log ]; then
        log "${RED}📋 Error: $(head -1 /tmp/model_check_error.log)${NC}"
    fi
fi

rm -f /tmp/model_check.json /tmp/model_check_error.log

# Run the actual model management
python -m src.shared.model_manager --models-path /app/models --verbose

model_status=$?

if [ $model_status -eq 0 ]; then
    log "${GREEN}✅ All required models are available${NC}"
    
    # Final verification (optional, controlled by env var)
    if [ "${VERIFY_MODELS_ON_START:-false}" = "true" ]; then
        log "${BLUE}🔍 Running model verification...${NC}"
        python manage_models.py verify --models-path /app/models --json > /tmp/verify_results.json 2>/dev/null
        
        if [ $? -eq 0 ] && command_exists jq; then
            failed_models=$(jq -r '. | to_entries[] | select(.value.valid == false) | .key' /tmp/verify_results.json | wc -l)
            if [ "$failed_models" -gt 0 ]; then
                log "${YELLOW}⚠️  $failed_models models failed verification${NC}"
            else
                log "${GREEN}✅ All models verified successfully${NC}"
            fi
        fi
        
        rm -f /tmp/verify_results.json
    fi
else
    log "${RED}❌ Model setup failed (exit code: $model_status)${NC}"
    
    # Show detailed error information
    if [ -f /tmp/model_error.log ]; then
        log "${RED}📋 Error details:${NC}"
        tail -5 /tmp/model_error.log | while read line; do
            log "${RED}   $line${NC}"
        done
    fi
    
    # Determine if we should fail or continue
    if [ "${FAIL_ON_MODEL_ERROR:-true}" = "true" ]; then
        log "${RED}💥 Exiting due to model setup failure${NC}"
        exit $model_status
    else
        log "${YELLOW}⚠️  Continuing despite model setup issues (FAIL_ON_MODEL_ERROR=false)${NC}"
    fi
fi

# Set environment variables for model paths
export NLTK_DATA="/app/models/nltk_data"
export GENSIM_DATA_DIR="/app/models/gensim_data"

else
    log "${BLUE}📡 API container - skipping model management${NC}"
fi

log "${GREEN}🎯 Starting main application: $@${NC}"

# Execute the main command
exec "$@"