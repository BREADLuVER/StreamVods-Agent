clip_generation/
├── __init__.py
├── config.py          # Tunable defaults
├── types.py           # Data classes
├── loader.py          # Data loading
├── seeding.py         # Reaction-based seeding
├── grouping.py        # Arc building + extensions
├── windowing.py       # Dynamic padding + boundary snapping
├── scoring.py         # Quality gates
├── selection.py       # Deduplication
├── title_llm.py       # LLM for titles only
├── manifest.py        # Output formatting
├── pipeline.py        # OOP orchestrator
└── cli_generate.py    # CLI interface
