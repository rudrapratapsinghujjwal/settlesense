"""Debug: see exact raw Gemini response for a real pipeline record"""
import os, json, re, sys
sys.path.insert(0, '.')
from dotenv import load_dotenv
load_dotenv('.env', override=True)
from openai import OpenAI
from settlesense.config import config
from settlesense.data_generator import main as gen_main
from settlesense.normalization import load_and_normalize_all
from settlesense.matching import run_deterministic_baseline, generate_candidates
from settlesense.evidence import assemble_evidence
from settlesense.pipeline import _hint_category
from settlesense.classifier import SYSTEM_PROMPT, USER_PROMPT_TEMPLATE, _format_evidence_for_prompt, _format_candidates_for_prompt
import logging
logging.basicConfig(level=logging.CRITICAL)

# Generate data
gen_main(seed=42, output_dir=config.data_dir)

# Get one real exception record
normalized = load_and_normalize_all(config.data_dir)
all_txns = normalized["payments"] + normalized["recon"] + normalized["ledger"] + normalized["bank"]
baseline = run_deterministic_baseline(normalized)
unresolved = baseline.unresolved_payment_ids

txn_by_id = {t.txn_id: t for t in all_txns}
txn_id = unresolved[0]
txn = txn_by_id[txn_id]
candidates = generate_candidates(txn, all_txns)
hint_cat = _hint_category(txn, candidates)
evidence = assemble_evidence(hint_cat, txn, candidates, all_txns)

user_msg = USER_PROMPT_TEMPLATE.format(
    record_id=txn_id,
    source=txn.source.value,
    amount=int(abs(txn.amount)),
    date=txn.transaction_date.isoformat() if txn.transaction_date else "unknown",
    hint_category=hint_cat.value,
    evidence_json=_format_evidence_for_prompt(evidence),
    candidates_json=_format_candidates_for_prompt(candidates),
)

print(f"Record: {txn_id} | hint: {hint_cat.value} | candidates: {len(candidates)}")
print(f"User prompt length: {len(user_msg)} chars")
print()

key = config.llm.openai_api_key
client = OpenAI(api_key=key, base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
resp = client.chat.completions.create(
    model=config.llm.model,
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ],
    max_tokens=2048,
    temperature=0.0,
)

raw = resp.choices[0].message.content
finish = resp.choices[0].finish_reason
print(f"Finish reason: {finish}")
print(f"Raw response ({len(raw or '')} chars):")
print(repr(raw))
print()

if raw:
    raw2 = raw.strip()
    if raw2.startswith("```"):
        lines = raw2.split("\n")
        inner = lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        raw2 = "\n".join(inner).strip()
    b1 = raw2.find("{")
    b2 = raw2.rfind("}")
    if b1 != -1 and b2 != -1:
        raw2 = raw2[b1:b2+1]
    raw2 = re.sub(r',\s*([}\]])', r'\1', raw2)
    print(f"After cleanup ({len(raw2)} chars):")
    print(raw2[:500])
    try:
        data = json.loads(raw2)
        print("\nPARSED OK:", list(data.keys()))
    except Exception as e:
        print(f"\nPARSE FAILED: {e}")
