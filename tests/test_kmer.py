import unittest
from src import KMer

class TestKMer(unittest.TestCase):

  def test_basic_functionality(self):
    tokenizer = KMer(kmer=4)
    sequence = "BAACATGTCCTGCATGGCATTAMGTTTGTTGGGGCAGTGCCCGPGATAGCATCAACGCTGCGCTGATTTGCCGTGGCGAGAAAE"
    shredded = tokenizer._shred(sequence)
    self.assertEqual(shredded, ["B", "AACA", "TGTC", "CTGC", "ATGG", "CATT", "A", "M", "GTTT", "GTTG", "GGGC", "AGTG", "CCCG", "P", "GATA", "GCAT", "CAAC", "GCTG", "CGCT", "GATT", "TGCC", "GTGG", "CGAG", "AAA", "E"])

    encoded = tokenizer.encode(sequence)
    self.assertEqual(encoded, [3, 177, 345, 575, 199, 543, 7, 1, 443, 444, 475, 219, 629, 2, 417, 488, 540, 494, 603, 418, 355, 449, 589, 37, 5])

    decoded = tokenizer.decode(encoded)
    self.assertEqual(decoded, "BAACATGTCCTGCATGGCATTAMGTTTGTTGGGGCAGTGCCCGPGATAGCATCAACGCTGCGCTGATTTGCCGTGGCGAGAAAE")

  def test_invalid_input(self):
    tokenizer = KMer(kmer=3)
    with self.assertRaises(ValueError):
      tokenizer.encode("INVALID_SEQ")

  def test_special_tokens(self):
    tokenizer = KMer(kmer=2)
    sequence = "MP"
    encoded = tokenizer.encode(sequence)
    decoded = tokenizer.decode(encoded)
    self.assertEqual(decoded, sequence)

if __name__ == "__main__":
  unittest.main()