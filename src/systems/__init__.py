"""src.systems — adapters for the 10 memory systems + 2 baselines evaluated by SixAxis.

Every adapter implements the same `MemorySystem` contract (src.architectures.base) and obtains
ALL language-model and embedding calls through the OpenAI shim (tools/openai_shim.py), so one
generator + one embedder serve every system and the shim log is the write-path cost ledger.
"""
