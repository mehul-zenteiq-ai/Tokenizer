"""
special_tokens.py
=================
Special tokens for the 150,016-vocab scientific BPE tokenizer.

Structure:
    IDs  0 –  3   Foundational (pad, eos, bos, unk)
    IDs  4 –  8   Utility (mask, multimodal placeholder, alt-bos, image)
    IDs  9 – 13   Chat format (system, user, assistant, turn delimiters)
    IDs 14 – 24   Tool use (tools, tool_call, arg_key, arg_value,
                             tool_response, tool_declare, observation)
    IDs 25 – 27   Reasoning (think, nothink)
    IDs 28 – 32   Language tags (en, hi, ta, te, kn)
    IDs 33 – 35   FIM / code infilling (prefix, middle, suffix)
    IDs 36 – 1059 Reserved (1024 slots for future use)

Total special tokens: 1060

Notes:
  - <unk> is kept for framework compatibility even though BBPE has zero OOV.
    It will never fire during tokenization.
  - <2mass> (Qwen astronomy tag) was dropped — irrelevant to this model.
    ID 7 is repurposed to <lang_en> to keep language tags contiguous.
  - [multimodal] and <|image_soft_token|> are retained as placeholders
    in case image inputs are ever added. They do nothing in the current model.
  - Reserved slots cost BPE merges (1024 fewer merges out of ~148k budget,
    ~0.7% loss). Worthwhile for future-proofing without SFT data remapping.

Relationship to training data:
  - GENERATE (model learns to emit these in SFT outputs):
        <think>, </think>, <|nothink|>
        <tool_call>, </tool_call>, <arg_key>, </arg_key>,
        <arg_value>, </arg_value>, <tool_response>, </tool_response>
        <|fim_prefix|>, <|fim_middle|>, <|fim_suffix|>
        <lang_*> tags (in multilingual SFT responses)

  - UNDERSTAND but NOT generate (inserted by inference template):
        <|system|>, <|user|>, <|assistant|>
        <|start_of_turn|>, <|end_of_turn|>
        <tools>, </tools>, <|tool_declare|>, <|observation|>
        <bos>, <eos>

  - NEVER used in practice:
        <unk>  (BBPE has no OOV)
        <mask> (causal LM, not MLM)
        <|reserved_0|> ... <|reserved_1023|>  (empty slots)
"""


# ── Foundational ───────────────────────────────────────────────────────────────
# IDs 0–3. Framework-critical. Do not reorder.
# <unk> will never fire (BBPE) but must exist for HuggingFace compatibility.

FOUNDATIONAL = [
    "<pad>",       # 0  — padding token
    "<eos>",       # 1  — end of sequence
    "<bos>",       # 2  — beginning of sequence
    "<unk>",       # 3  — unknown (unused in BBPE, kept for compatibility)
]


# ── Utility ────────────────────────────────────────────────────────────────────
# IDs 4–8. Miscellaneous slots inherited from base config.
# <mask> is MLM-style; unused in causal training but harmless.
# [multimodal] and <|image_soft_token|> are vision placeholders; unused now.
# [@BOS@] is a Qwen-style document-level BOS variant; kept for compatibility.

UTILITY = [
    "<mask>",                  # 4  — MLM mask (unused in causal LM)
    "[multimodal]",            # 5  — multimodal placeholder (unused)
    # "[@BOS@]",                 # 6  — document-level BOS variant
    # "<|image_soft_token|>",    # 8  — image token placeholder (unused)
]


# ── Chat format ────────────────────────────────────────────────────────────────
# IDs 9–13. Define the conversation structure.
# These appear in the PROMPT side — model understands but does NOT generate them.
# Inference template inserts these; they must be consistent across
# pretraining continued training, SFT, and inference.
#
# Format example:
#   <bos><|system|>You are a scientific assistant.<|end_of_turn|>
#   <|start_of_turn|><|user|>Solve ∇²ψ = 0<|end_of_turn|>
#   <|start_of_turn|><|assistant|><think>...</think>The solution is...<eos>

CHAT = [
    "<|system|>",          # 9   — system prompt delimiter
    "<|user|>",            # 10  — user turn delimiter
    "<|assistant|>",       # 11  — assistant turn delimiter
    "<|start_of_turn|>",   # 12  — turn start marker
    "<|end_of_turn|>",     # 13  — turn end marker (also used as stop token)
]


# ── Tool use ───────────────────────────────────────────────────────────────────
# IDs 14–24. Structured tool calling format.
# Model generates <tool_call>...</tool_call> in its responses when invoking tools.
# <tools>...</tools> wraps tool definitions in the system prompt (not generated).
# <|observation|> precedes tool results injected back into context.
#
# Format example (assistant output):
#   <tool_call>
#   <arg_key>function</arg_key><arg_value>solve_ode</arg_value>
#   <arg_key>equation</arg_key><arg_value>dy/dx = y</arg_value>
#   </tool_call>
#   <|observation|>y = Ce^x</|observation|>

TOOL_USE = [
    "<tools>",              # 14  — opens tool definition block (in system prompt)
    "</tools>",             # 15  — closes tool definition block
    "<tool_call>",          # 16  — model emits this to invoke a tool
    "</tool_call>",         # 17  — closes tool call
    "<arg_key>",            # 18  — argument name
    "</arg_key>",           # 19
    "<arg_value>",          # 20  — argument value
    "</arg_value>",         # 21
    "<tool_response>",      # 22  — wraps tool result returned to model
    "</tool_response>",     # 23
    "<|tool_declare|>",     # 24  — tool schema declaration marker
    "<|observation|>",      # 25  — precedes injected tool output
]


# ── Reasoning ──────────────────────────────────────────────────────────────────
# IDs 26–28. Chain-of-thought control tokens.
# <think>/</think>: model emits these to wrap its scratchpad reasoning.
# <|nothink|>: signals skip-reasoning mode for fast inference on simple queries.
# These MUST appear in SFT training data as model-generated tokens or the
# model will not learn to produce them.
#
# Format example (assistant output with reasoning):
#   <think>
#   The PDE is parabolic. I should apply separation of variables...
#   </think>
#   The solution is u(x,t) = Σ aₙ sin(nπx) e^{-n²π²t}

REASONING = [
    "<think>",        # 26  — opens reasoning scratchpad
    "</think>",       # 27  — closes reasoning scratchpad
    "<|nothink|>",    # 28  — suppresses reasoning (fast inference mode)
]


# ── Language tags ──────────────────────────────────────────────────────────────
# IDs 29–33. One tag per language in the training mix.
# NOT inserted during pretraining (step 2 confirmed no markup).
# Used in SFT and inference to signal response language.
# <lang_en> is included for completeness and symmetric handling.
# Order follows training data volume (largest to smallest) for mnemonic clarity.

LANGUAGE_TAGS = [
    "<lang_hin>",   # 28  Hindi        (Devanagari)
    "<lang_tam>",   # 29  Tamil
    "<lang_tel>",   # 30  Telugu
    "<lang_kan>",   # 31  Kannada
    "<lang_mal>",   # 32  Malayalam
    "<lang_mar>",   # 33  Marathi      (Devanagari)
    "<lang_guj>",   # 34  Gujarati
    "<lang_ben>",   # 35  Bengali
    "<lang_pan>",   # 36  Punjabi      (Gurmukhi)
    "<lang_ory>",   # 37  Odia
    "<lang_urd>",   # 38  Urdu
    "<lang_npi>",   # 39  Nepali       (Devanagari)
    "<lang_pus>",   # 40  Pashto
    "<lang_sin>",   # 41  Sinhala
    "<lang_mya>",   # 42  Burmese
    "<lang_fas>",   # 43  Dari/Persian
    "<lang_eng>",   # 46  English
    "<lang_deu>",   # 47  German
    "<lang_fra>",   # 48  French
    "<lang_rus>",   # 49  Russian      (Cyrillic)
    "<lang_cmn>",   # 50  Mandarin Chinese
    "<lang_jpn>",   # 51  Japanese
    "<lang_kor>",   # 52  Korean
]


# # ── FIM (Fill-in-the-Middle) ───────────────────────────────────────────────────
# # IDs 34–36. Code infilling tokens.
# # Required for the model to learn to complete code given prefix + suffix context.
# # Training data must be formatted with these tokens for FIM to work.
# # Standard PSM (prefix-suffix-middle) format:
# #   <|fim_prefix|>def solve(x):<|fim_suffix|>    return result<|fim_middle|>
# #       return x**2 +

# FIM = [
#     "<|fim_prefix|>",   # 38  — precedes the code prefix
#     "<|fim_middle|>",   # 39  — precedes the span to be infilled (model generates after this)
#     "<|fim_suffix|>",   # 40  — precedes the code suffix
# ]


# ── Reserved ───────────────────────────────────────────────────────────────────
# IDs 37–1060. 1024 empty slots for future use.
# Cost: ~0.7% of BPE merge budget (1024 fewer merges out of ~148k).
# Benefit: adding a new special token later requires only a weight resize
#          of the embedding matrix, not a full vocabulary remapping.
# Use cases anticipated: additional language tags, domain tags, new tool tokens,
#          additional reasoning control tokens, distillation tokens.
# These tokens are NEVER used in training data and are never generated.

RESERVED = [f"<|reserved_{i}|>" for i in range(2048)]   # 37–1060


# ══════════════════════════════════════════════════════════════════════════════
# ASSEMBLED LIST
# Flat list in ID order. Pass this to the tokenizer trainer.
# ══════════════════════════════════════════════════════════════════════════════

SPECIAL_TOKENS = (
    FOUNDATIONAL    # IDs   0 –    3   (4 tokens)
    + UTILITY       # IDs   4 –    7   (4 tokens)  note: gap from <2mass> removal
    + CHAT          # IDs   8 –   12   (5 tokens)
    + TOOL_USE      # IDs  13 –   24   (12 tokens)
    + REASONING     # IDs  25 –   27   (3 tokens)
    + LANGUAGE_TAGS # IDs  28 –   32   (5 tokens)
    # + FIM           # IDs  33 –   35   (3 tokens)
    + RESERVED      # IDs  36 – 1059   (1024 tokens)
)

# ── Named references for use in training/inference code ───────────────────────
# Import these instead of hardcoding string literals.

PAD_TOKEN           = "<pad>"
EOS_TOKEN           = "<eos>"
BOS_TOKEN           = "<bos>"
UNK_TOKEN           = "<unk>"

SYSTEM_TOKEN        = "<|system|>"
USER_TOKEN          = "<|user|>"
ASSISTANT_TOKEN     = "<|assistant|>"
START_OF_TURN_TOKEN = "<|start_of_turn|>"
END_OF_TURN_TOKEN   = "<|end_of_turn|>"

THINK_TOKEN         = "<think>"
END_THINK_TOKEN     = "</think>"
NOTHINK_TOKEN       = "<|nothink|>"

TOOL_CALL_TOKEN     = "<tool_call>"
END_TOOL_CALL_TOKEN = "</tool_call>"
OBSERVATION_TOKEN   = "<|observation|>"

# FIM_PREFIX_TOKEN    = "<|fim_prefix|>"
# FIM_MIDDLE_TOKEN    = "<|fim_middle|>"
# FIM_SUFFIX_TOKEN    = "<|fim_suffix|>"

LANG_HIN_TOKEN = "<lang_hin>"
LANG_TAM_TOKEN = "<lang_tam>"
LANG_TEL_TOKEN = "<lang_tel>"
LANG_KAN_TOKEN = "<lang_kan>"
LANG_MAL_TOKEN = "<lang_mal>"
LANG_MAR_TOKEN = "<lang_mar>"
LANG_GUJ_TOKEN = "<lang_guj>"
LANG_BEN_TOKEN = "<lang_ben>"
LANG_PAN_TOKEN = "<lang_pan>"
LANG_ORY_TOKEN = "<lang_ory>"
LANG_URD_TOKEN = "<lang_urd>"
LANG_NPI_TOKEN = "<lang_npi>"
LANG_PUS_TOKEN = "<lang_pus>"
LANG_SIN_TOKEN = "<lang_sin>"
LANG_MYA_TOKEN = "<lang_mya>"
LANG_FAS_TOKEN = "<lang_fas>"
LANG_ENG_TOKEN = "<lang_eng>"
LANG_DEU_TOKEN = "<lang_deu>"
LANG_FRA_TOKEN = "<lang_fra>"
LANG_RUS_TOKEN = "<lang_rus>"
LANG_CMN_TOKEN = "<lang_cmn>"
LANG_JPN_TOKEN = "<lang_jpn>"
LANG_KOR_TOKEN = "<lang_kor>"


# Stop tokens — pass these to inference engine as generation stop signals
STOP_TOKENS = [EOS_TOKEN, END_OF_TURN_TOKEN]


# ── Sanity check ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Total special tokens : {len(SPECIAL_TOKENS)}")
    print(f"  Foundational       : {len(FOUNDATIONAL)}")
    print(f"  Utility            : {len(UTILITY)}")
    print(f"  Chat               : {len(CHAT)}")
    print(f"  Tool use           : {len(TOOL_USE)}")
    print(f"  Reasoning          : {len(REASONING)}")
    print(f"  Language tags      : {len(LANGUAGE_TAGS)}")
    # print(f"  FIM                : {len(FIM)}")
    print(f"  Reserved           : {len(RESERVED)}")
    print()

    # Verify no duplicates
    assert len(SPECIAL_TOKENS) == len(set(SPECIAL_TOKENS)), "DUPLICATE TOKENS FOUND"
    print("No duplicates — OK")

    # Verify reserved range
    assert RESERVED[0]    == "<|reserved_0|>",    "Reserved start mismatch"
    assert RESERVED[-1]   == "<|reserved_2047|>", "Reserved end mismatch"
    print("Reserved range — OK")

    # Print ID map
    print()
    print("ID   Token")
    print("─" * 40)
    for i, tok in enumerate(SPECIAL_TOKENS):
        if i < 37 or i >= len(SPECIAL_TOKENS) - 2:   # show all non-reserved + last 2
            print(f"{i:<5}{tok}")
        elif i == 37:
            print(f"...  ({len(RESERVED)} reserved slots: IDs 36–1059)")
