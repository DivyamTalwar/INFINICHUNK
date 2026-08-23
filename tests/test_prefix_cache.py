import unittest

from infinichunk_ext.prefix_cache import PrefixCacheReceipt, common_prefix_length


class PrefixCacheTests(unittest.TestCase):
    def test_common_prefix_length(self):
        self.assertEqual(common_prefix_length([1, 2, 3], [1, 2, 9]), 2)
        self.assertEqual(common_prefix_length([], [1]), 0)
        self.assertEqual(common_prefix_length([1, 2], [1, 2, 3]), 2)

    def test_receipt_counts_only_reusable_followup_prefixes(self):
        receipt = PrefixCacheReceipt()
        receipt.observe(None, [10, 11, 12])
        receipt.observe([10, 11, 12], [10, 11, 99, 100])
        receipt.observe([10, 11, 99, 100], [10, 11, 99, 8])
        self.assertEqual(receipt.turns, 3)
        self.assertEqual(receipt.cacheable_tokens, 5)
        self.assertEqual(receipt.submitted_tokens, 11)
        self.assertAlmostEqual(receipt.eligible_fraction, 5 / 11)


if __name__ == "__main__":
    unittest.main()
