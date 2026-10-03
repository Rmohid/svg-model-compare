#!/usr/bin/env python3
"""worktracker-676.4.22.167.2 — call_openrouter must retry at least once
when a non-reasoning-override model comes back with finish_reason=length.

The truncation signal is the error string _openrouter_attempt returns for
finish_reason == "length" (see its first branch). Everything here runs
offline: _openrouter_attempt is stubbed, and the module-level API key is
satisfied from the environment before generate is imported.
"""
import os
import unittest
from unittest import mock

os.environ.setdefault("OPENROUTER_911_API_KEY", "offline-test-key")

import generate


TRUNCATED = "Truncated: hit max_tokens before finishing (reasoning likely ate the budget)"
GOOD_SVG = '<svg xmlns="http://www.w3.org/2000/svg" width="10" height="10">' + ("x" * 300) + "</svg>"


class RetryOnLengthTest(unittest.TestCase):
    def test_non_override_model_retries_once_on_length(self):
        calls = []

        def fake_attempt(name, model_id, has_reasoning_override):
            calls.append(has_reasoning_override)
            if len(calls) == 1:
                return None, 0.1, TRUNCATED, {"tokens": 1}
            return GOOD_SVG, 0.1, None, {"tokens": 2}

        with mock.patch.object(generate, "_openrouter_attempt", fake_attempt):
            name, svg, elapsed, error, usage = generate.call_openrouter(
                "Ox Alpha", "stealth/ox-alpha"
            )

        self.assertGreaterEqual(
            len(calls), 2, "finish_reason=length was not retried (non-override model)"
        )
        self.assertIsNone(error, error)
        self.assertEqual(svg, GOOD_SVG)

    def test_success_first_attempt_is_not_retried(self):
        calls = []

        def fake_attempt(name, model_id, has_reasoning_override):
            calls.append(1)
            return GOOD_SVG, 0.1, None, {"tokens": 1}

        with mock.patch.object(generate, "_openrouter_attempt", fake_attempt):
            name, svg, elapsed, error, usage = generate.call_openrouter(
                "Ox Alpha", "stealth/ox-alpha"
            )

        self.assertEqual(len(calls), 1, "a clean first attempt must not be re-called")
        self.assertIsNone(error, error)


if __name__ == "__main__":
    unittest.main()
