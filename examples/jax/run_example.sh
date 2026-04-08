#!/usr/bin/env bash
set -e

if [ -z "$1" ]; then
    echo "Usage: ./run_example.sh <example_name|all> [args forwarded to script]"
    echo ""
    echo "  e.g. ./run_example.sh lqr --solver MOREAU"
    echo "       ./run_example.sh all --solver MOREAU"
    echo ""
    echo "Available examples:"
    for d in */; do
        ex="${d%/}"
        [ -f "$ex/$ex.py" ] && echo "  $ex"
    done
    exit 1
fi

run_one() {
    local ex="$1"
    shift
    echo "━━━ $ex ━━━"
    python "$ex/$ex.py" "$@"
    echo ""
}

if [ "$1" = "all" ]; then
    shift
    for d in */; do
        ex="${d%/}"
        [ -f "$ex/$ex.py" ] && run_one "$ex" "$@"
    done
else
    EXAMPLE="$1"
    shift
    if [ ! -d "$EXAMPLE" ]; then
        echo "Error: example '$EXAMPLE' not found"
        exit 1
    fi
    run_one "$EXAMPLE" "$@"
fi
