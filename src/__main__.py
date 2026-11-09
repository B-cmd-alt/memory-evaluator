"""Entry point for the memory benchmark experiment.

Usage:
    # Full experiment
    python -m src

    # Pilot (S0 only)
    python -m src --pilot

    # Specific architectures
    python -m src --arch A_structured E_lattice --scale S0 S1

    # Skip Neo4j-dependent architectures
    python -m src --skip B_graph D_hybrid
"""

from src.runner import main

main()
