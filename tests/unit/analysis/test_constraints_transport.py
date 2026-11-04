import unittest

import dill

from mvesuvio.util.constraints_transport import deserialize_constraints, serialize_constraints


class TestConstraintsTransport(unittest.TestCase):
    def test_registry_roundtrip(self):
        constraints = ({"type": "eq", "fun": lambda par: par[0] - 2.5 * par[3]},)

        payload = serialize_constraints(constraints)
        restored = deserialize_constraints(payload)

        self.assertIsInstance(payload, str)
        self.assertTrue(payload.startswith("registry:"))
        self.assertIs(restored, constraints)

    def test_legacy_dill_payload(self):
        constraints = ({"type": "eq", "fun": lambda par: par[0] - 2.5 * par[3]},)
        legacy_payload = str(dill.dumps(constraints))

        restored = deserialize_constraints(legacy_payload)

        self.assertEqual(restored[0]["fun"]([3, 0, 0, 1]), 0.5)

    def test_unknown_registry_token_raises(self):
        with self.assertRaisesRegex(RuntimeError, "Constraints token not found in registry: missing-token"):
            deserialize_constraints("registry:missing-token")
