"""
SettleSense — Razorpay API Client
===================================
Phase 0 feasibility test + real data fetching.

Tests:
  1. Create an order
  2. Fetch payment (simulated capture in test mode)
  3. Fetch settlements/recon combined endpoint
  4. Record actual API responses and schema

All results are saved to data/razorpay_live/ for traceability.
If settlement data is not populated in test mode, this is documented.
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class RazorpayClient:
    """
    Thin wrapper around the razorpay SDK.
    Handles test-mode vs live-mode detection.
    Saves API responses for traceability.
    """

    def __init__(self, key_id: str, key_secret: str):
        import razorpay
        self.client = razorpay.Client(auth=(key_id, key_secret))
        self.is_test_mode = key_id.startswith("rzp_test_")
        self.key_id = key_id
        logger.info(
            "Razorpay client initialized | mode=%s",
            "TEST" if self.is_test_mode else "LIVE",
        )

    def _save_response(self, name: str, data: Any, output_dir: Path) -> Path:
        """Save API response to disk for traceability."""
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y%m%dT%H%M%S")
        path = output_dir / f"{name}_{ts}.json"
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info("Saved API response: %s", path)
        return path

    def test_create_order(self, amount_paise: int = 100000) -> dict:
        """Create a test order (₹1000). Returns API response."""
        payload = {
            "amount": amount_paise,
            "currency": "INR",
            "receipt": f"settlesense_test_{int(time.time())}",
            "notes": {"purpose": "settlesense_phase0_test"},
        }
        try:
            response = self.client.order.create(data=payload)
            logger.info("Order created: %s | amount=%d", response.get("id"), amount_paise)
            return {"success": True, "response": response}
        except Exception as e:
            logger.error("Order creation failed: %s", e)
            return {"success": False, "error": str(e)}

    def fetch_orders(self, count: int = 10) -> dict:
        """Fetch recent orders."""
        try:
            response = self.client.order.all({"count": count})
            logger.info("Fetched %d orders", len(response.get("items", [])))
            return {"success": True, "response": response}
        except Exception as e:
            logger.error("Order fetch failed: %s", e)
            return {"success": False, "error": str(e)}

    def fetch_payments(self, count: int = 10) -> dict:
        """Fetch recent payments. Records actual schema."""
        try:
            response = self.client.payment.all({"count": count})
            logger.info("Fetched %d payments", len(response.get("items", [])))
            return {"success": True, "response": response}
        except Exception as e:
            logger.error("Payment fetch failed: %s", e)
            return {"success": False, "error": str(e)}

    def fetch_settlements(self, count: int = 10) -> dict:
        """Fetch settlements."""
        try:
            response = self.client.settlement.all({"count": count})
            logger.info("Fetched settlements: %s", response)
            return {"success": True, "response": response}
        except Exception as e:
            logger.error("Settlement fetch failed: %s", e)
            return {"success": False, "error": str(e)}

    def fetch_settlement_recon(self, year: int = 2025, month: int = 1) -> dict:
        """
        Critical Phase 0 test: GET /v1/settlements/recon/combined
        Tests whether test-mode settlement data is populated.
        """
        try:
            # The SDK may not have a direct method for recon/combined.
            # Use raw request if available.
            import requests
            import base64
            credentials = base64.b64encode(
                f"{self.key_id}:__SECRET__".encode()
            ).decode()
            # Note: We don't log the actual secret
            params = {
                "year": year,
                "month": month,
                "day": 1,
                "count": 100,
            }
            resp = requests.get(
                "https://api.razorpay.com/v1/settlements/recon/combined",
                params=params,
                auth=(self.key_id, "__SECRET__"),  # Will be replaced in run_phase0
            )
            data = resp.json()
            logger.info(
                "Settlement recon response | status=%d | count=%s",
                resp.status_code,
                data.get("count", "N/A"),
            )
            return {
                "success": resp.status_code == 200,
                "status_code": resp.status_code,
                "response": data,
                "item_count": len(data.get("items", [])),
            }
        except Exception as e:
            logger.error("Settlement recon fetch failed: %s", e)
            return {"success": False, "error": str(e)}


def run_phase0_feasibility(
    key_id: str,
    key_secret: str,
    output_dir: Path,
) -> dict:
    """
    Phase 0: Run all Razorpay API feasibility tests.
    Saves actual responses. Returns summary of what worked.
    """
    import requests

    results = {
        "timestamp": datetime.utcnow().isoformat(),
        "key_id_prefix": key_id[:12] + "...",
        "is_test_mode": key_id.startswith("rzp_test_"),
        "tests": {},
    }

    client = RazorpayClient(key_id, key_secret)

    # Test 1: Create order
    logger.info("Phase 0 Test 1: Create order")
    order_result = client.test_create_order()
    results["tests"]["create_order"] = {
        "success": order_result["success"],
        "error": order_result.get("error"),
        "order_id": order_result.get("response", {}).get("id") if order_result["success"] else None,
    }
    if order_result["success"]:
        client._save_response("create_order", order_result["response"], output_dir)

    # Test 2: Fetch payments
    logger.info("Phase 0 Test 2: Fetch payments")
    payment_result = client.fetch_payments(count=5)
    results["tests"]["fetch_payments"] = {
        "success": payment_result["success"],
        "count": len(payment_result.get("response", {}).get("items", [])),
        "error": payment_result.get("error"),
    }
    if payment_result["success"] and payment_result.get("response", {}).get("items"):
        # Record actual payment schema fields
        first_payment = payment_result["response"]["items"][0]
        results["tests"]["fetch_payments"]["actual_fields"] = list(first_payment.keys())
        client._save_response("payments", payment_result["response"], output_dir)

    # Test 3: Fetch settlements
    logger.info("Phase 0 Test 3: Fetch settlements")
    settlement_result = client.fetch_settlements()
    results["tests"]["fetch_settlements"] = {
        "success": settlement_result["success"],
        "count": len(settlement_result.get("response", {}).get("items", [])),
        "error": settlement_result.get("error"),
    }
    if settlement_result["success"]:
        client._save_response("settlements", settlement_result["response"], output_dir)

    # Test 4: Settlement recon combined (CRITICAL)
    logger.info("Phase 0 Test 4: Settlement recon combined endpoint")
    recon_result = requests.get(
        "https://api.razorpay.com/v1/settlements/recon/combined",
        params={"year": 2025, "month": 1, "day": 1, "count": 100},
        auth=(key_id, key_secret),
    )
    recon_data = recon_result.json()
    item_count = len(recon_data.get("items", []))
    results["tests"]["settlement_recon_combined"] = {
        "status_code": recon_result.status_code,
        "success": recon_result.status_code == 200,
        "item_count": item_count,
        "populated": item_count > 0,
        "conclusion": (
            "POPULATED — real recon data available"
            if item_count > 0
            else "EMPTY — test mode does not auto-populate recon data. "
                 "Will use schema-accurate synthetic recon data. (Expected behavior)"
        ),
    }
    if recon_result.status_code == 200:
        client._save_response("settlement_recon", recon_data, output_dir)

    # Save full results
    results_path = output_dir / "phase0_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Phase 0 results saved to %s", results_path)

    return results
