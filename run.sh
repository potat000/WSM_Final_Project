#!/bin/bash
set -e

log() {
    local timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    local message="$timestamp - $1"
    local len=${#message}
    local border=$(printf '=%.0s' $(seq 1 $len))
    
    echo "$border"
    echo "$message"
    echo "$border"
}

run_results() {
    local language=$1
    
    log "[INFO] Running inference for language: ${language}"
    
    # 使用 Hybrid Retrieval + 調整參數
    python3 ./My_RAG/main.py \
        --query_path ./dragonball_dataset/queries_test/test_queries_${language}.jsonl \
        --docs_path ./dragonball_dataset/dragonball_docs.jsonl \
        --language ${language} \
        --output ./predictions/predictions_${language}.jsonl \
        --use_hybrid \
        --retrieval_method rrf \
        --top_k 5 \
        --rrf_k 60
    
    log "[INFO] Checking output format for language: ${language}"
    
    python3 ./check_output_format.py \
        --query_file ./dragonball_dataset/queries_test/test_queries_${language}.jsonl \
        --processed_file ./predictions/predictions_${language}.jsonl
    
    if [ $? -eq 0 ]; then
        echo Format check passed.
    fi
}

run_results "en"
run_results "zh"

log "[INFO] All inference tasks completed."