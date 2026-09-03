"""Seed the pipeline from scratch and run evaluation. Outputs real numbers."""
import sys, json, logging
sys.path.insert(0, '.')
logging.basicConfig(level=logging.WARNING)  # Keep output clean

from settlesense.config import config, load_config
print(f"Provider: {config.llm.provider} | Razorpay: {'OK' if config.razorpay.is_configured else 'MISSING'}")

from settlesense.data_generator import main as gen_main
from settlesense.pipeline import run_full_pipeline, run_evaluation
from settlesense.database import initialize_database

# Ensure DB ready
initialize_database(config.db_path)

# Generate data
gen_result = gen_main(seed=config.random_seed, output_dir=config.data_dir)
dist = gen_result['distribution']
print(f"\nData generated: {gen_result['total_ground_truth']} ground truth records")
print(f"Distribution: {dist}")
print(f"Label leakage: {gen_result['label_leakage']}")

# Run pipeline
print("\n--- Running Pipeline (tune split) ---")
s = run_full_pipeline(config, split='tune', save_to_db=True)
print(f"Processed: {s['total_records']} | Clean: {s['clean_records']} ({s['clean_match_rate']:.0%})")
print(f"Exceptions: {s['exception_records']} | Auto: {s['auto_resolved']} | Human: {s['human_review']}")
print(f"Automation: {s['automation_rate']:.0%} | {s['throughput_rps']:.0f} rec/s | {s['total_time_ms']:.0f}ms")

# Evaluate
print("\n--- Evaluation (tune split) ---")
ev = run_evaluation(config, split='tune')
if 'error' in ev:
    print(f"ERROR: {ev['error']}")
else:
    print(f"Accuracy: {ev['overall_accuracy']:.0%} | Automation: {ev['automation_rate']:.0%} | FAR: {ev['false_auto_resolve_rate']:.1%}")
    print(f"Auto-resolved: {ev['auto_resolved']}/{ev['total_records']} | Human: {ev['human_review']}/{ev['total_records']}")
    print("\nPer-category:")
    for cat in json.loads(ev['per_category_json']):
        if cat['support'] > 0:
            print(f"  {cat['category']:25s} P={cat['precision']:.2f} R={cat['recall']:.2f} F1={cat['f1']:.2f} n={cat['support']}")

print("\nPipeline seeding complete. Dashboard: streamlit run app.py")
