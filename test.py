import unittest
from biosaic import tokenizer

class TestKMer(unittest.TestCase):

  def test_basic_functionality(self):
    token = tokenizer(encoding="base_3k")
    sequence = "TCTTACATAGAAAGGAGCGGTATTTGGTATGAATTTATTTGCAACTGACTGCTTGGAAGTTGGCGTACATCTTTCCACGGAAACTATGAAAATACTGGTCAGCCTCTCAGTCATTTCATAAAATCTTGATTTTGTATTACAACAAATTAGGATATTTTCAGTAGAACTGATTGTAAGGCCAGACTGTTGGAATGTAATTCCTTCCCAAACATCTCTCAGGGGCACTTTCCTGAACGGCTGCTGACAGCAGCATTTGAGGACGGTGGGGCGGAGGACATCCTGGGGGGCCTGGCTTCTTGGGAACTGGAGGCTTTGGCCCTTGTCCCACCCCTGCTCCCCTGAGGAGGGAGGCGTGGGGCCCTGGGCTGGCTGCAAGACGTGGAGTGACTGTGGGTCCCCGTGGCCCCTGACATGCTCCCAGGGAACCCAAGAAAAGACTGAGACCCTGTGGTGCCTCCCGCTTTCCATCCGCATTCCATGGCAGGTGAGTCTGATTATTCGAAGGAGGCTGGAGTGTGGGCGGAGGGCAGCGCCAGGTTTCCCAATCAGATTTGCTCAGGGTCCCTCCAGCAGTCCATGCCGCAGAGGCTGTCCCTTGGGGGCCCACGCATCCTAGCCACGGCCTCCTCACGTCCATGCGGGGATTTGCGCCCTGGAAGGAGCCGCCCGGCTGCCTCTCGCCAACATGCAGCACTTCCCTTCCTTTCCATGGAGCACGGTTCCTGTCCCGGGGGTCCATATTGGCCACTGTGGGAGAGAGTCGGGCAGCTGAATTCCCGCAGGTGGGAATGCCAGGGCCCGAGGATGTTGCCCCTGTCCTGAAGGCTGTCGCCCGATCGCTCTATCCAAGGCTGCCCTGGGGCAGCGTCACCTGGGGGTCCTGCGGGGGCTTCTCAGCACAGCATCCAGCACTGCCACCTAGTGTGTTCCCGTCACGTCTCCTCCCCCCGCCTGCACCAGGCACCAGAGACCCGGATGCCAAGGCCTGTCAGCTTCCTCAATGGGAAACTTTTCTTCAGTGAACAAAGCTCTGTTTTATA"
    tokenized = token.tokenize(sequence)
    actual_tokenized = ['TCT', 'CTT', 'TTA', 'TAC', 'ACA', 'CAT', 'ATA', 'TAG', 'AGA', 'GAA', 'AAA', 'AAG', 'AGG', 'GGA', 'GAG', 'AGC', 'GCG', 'CGG', 'GGT', 'GTA', 'TAT', 'ATT', 'TTT', 'TTG', 'TGG', 'GGT', 'GTA', 'TAT', 'ATG', 'TGA',
                        'GAA', 'AAT', 'ATT', 'TTT', 'TTA', 'TAT', 'ATT', 'TTT', 'TTG', 'TGC', 'GCA', 'CAA', 'AAC', 'ACT', 'CTG', 'TGA', 'GAC', 'ACT', 'CTG', 'TGC', 'GCT', 'CTT', 'TTG', 'TGG', 'GGA', 'GAA', 'AAG', 'AGT', 'GTT', 'TTG',
                        'TGG', 'GGC', 'GCG', 'CGT', 'GTA', 'TAC', 'ACA', 'CAT', 'ATC', 'TCT', 'CTT', 'TTT', 'TTC', 'TCC', 'CCA', 'CAC', 'ACG', 'CGG', 'GGA', 'GAA', 'AAA', 'AAC', 'ACT', 'CTA', 'TAT', 'ATG', 'TGA', 'GAA', 'AAA', 'AAA',
                        'AAT', 'ATA', 'TAC', 'ACT', 'CTG', 'TGG', 'GGT', 'GTC', 'TCA', 'CAG', 'AGC', 'GCC', 'CCT', 'CTC', 'TCT', 'CTC', 'TCA', 'CAG', 'AGT', 'GTC', 'TCA', 'CAT', 'ATT', 'TTT', 'TTC', 'TCA', 'CAT', 'ATA', 'TAA', 'AAA',
                        'AAA', 'AAT', 'ATC', 'TCT', 'CTT', 'TTG', 'TGA', 'GAT', 'ATT', 'TTT', 'TTT', 'TTG', 'TGT', 'GTA', 'TAT', 'ATT', 'TTA', 'TAC', 'ACA', 'CAA', 'AAC', 'ACA', 'CAA', 'AAA', 'AAT', 'ATT', 'TTA', 'TAG', 'AGG', 'GGA',
                        'GAT', 'ATA', 'TAT', 'ATT', 'TTT', 'TTT', 'TTC', 'TCA', 'CAG', 'AGT', 'GTA', 'TAG', 'AGA', 'GAA', 'AAC', 'ACT', 'CTG', 'TGA', 'GAT', 'ATT', 'TTG', 'TGT', 'GTA', 'TAA', 'AAG', 'AGG', 'GGC', 'GCC', 'CCA', 'CAG',
                        'AGA', 'GAC', 'ACT', 'CTG', 'TGT', 'GTT', 'TTG', 'TGG', 'GGA', 'GAA', 'AAT', 'ATG', 'TGT', 'GTA', 'TAA', 'AAT', 'ATT', 'TTC', 'TCC', 'CCT', 'CTT', 'TTC', 'TCC', 'CCC', 'CCA', 'CAA', 'AAA', 'AAC', 'ACA', 'CAT',
                        'ATC', 'TCT', 'CTC', 'TCT', 'CTC', 'TCA', 'CAG', 'AGG', 'GGG', 'GGG', 'GGC', 'GCA', 'CAC', 'ACT', 'CTT', 'TTT', 'TTC', 'TCC', 'CCT', 'CTG', 'TGA', 'GAA', 'AAC', 'ACG', 'CGG', 'GGC', 'GCT', 'CTG', 'TGC', 'GCT',
                        'CTG', 'TGA', 'GAC', 'ACA', 'CAG', 'AGC', 'GCA', 'CAG', 'AGC', 'GCA', 'CAT', 'ATT', 'TTT', 'TTG', 'TGA', 'GAG', 'AGG', 'GGA', 'GAC', 'ACG', 'CGG', 'GGT', 'GTG', 'TGG', 'GGG', 'GGG', 'GGC', 'GCG', 'CGG', 'GGA',
                        'GAG', 'AGG', 'GGA', 'GAC', 'ACA', 'CAT', 'ATC', 'TCC', 'CCT', 'CTG', 'TGG', 'GGG', 'GGG', 'GGG', 'GGG', 'GGC', 'GCC', 'CCT', 'CTG', 'TGG', 'GGC', 'GCT', 'CTT', 'TTC', 'TCT', 'CTT', 'TTG', 'TGG', 'GGG', 'GGA',
                        'GAA', 'AAC', 'ACT', 'CTG', 'TGG', 'GGA', 'GAG', 'AGG', 'GGC', 'GCT', 'CTT', 'TTT', 'TTG', 'TGG', 'GGC', 'GCC', 'CCC', 'CCT', 'CTT', 'TTG', 'TGT', 'GTC', 'TCC', 'CCC', 'CCA', 'CAC', 'ACC', 'CCC', 'CCC', 'CCT',
                        'CTG', 'TGC', 'GCT', 'CTC', 'TCC', 'CCC', 'CCC', 'CCT', 'CTG', 'TGA', 'GAG', 'AGG', 'GGA', 'GAG', 'AGG', 'GGG', 'GGA', 'GAG', 'AGG', 'GGC', 'GCG', 'CGT', 'GTG', 'TGG', 'GGG', 'GGG', 'GGC', 'GCC', 'CCC', 'CCT',
                        'CTG', 'TGG', 'GGG', 'GGC', 'GCT', 'CTG', 'TGG', 'GGC', 'GCT', 'CTG', 'TGC', 'GCA', 'CAA', 'AAG', 'AGA', 'GAC', 'ACG', 'CGT', 'GTG', 'TGG', 'GGA', 'GAG', 'AGT', 'GTG', 'TGA', 'GAC', 'ACT', 'CTG', 'TGT', 'GTG',
                        'TGG', 'GGG', 'GGT', 'GTC', 'TCC', 'CCC', 'CCC', 'CCG', 'CGT', 'GTG', 'TGG', 'GGC', 'GCC', 'CCC', 'CCC', 'CCT', 'CTG', 'TGA', 'GAC', 'ACA', 'CAT', 'ATG', 'TGC', 'GCT', 'CTC', 'TCC', 'CCC', 'CCA', 'CAG', 'AGG',
                        'GGG', 'GGA', 'GAA', 'AAC', 'ACC', 'CCC', 'CCA', 'CAA', 'AAG', 'AGA', 'GAA', 'AAA', 'AAA', 'AAG', 'AGA', 'GAC', 'ACT', 'CTG', 'TGA', 'GAG', 'AGA', 'GAC', 'ACC', 'CCC', 'CCT', 'CTG', 'TGT', 'GTG', 'TGG', 'GGT',
                        'GTG', 'TGC', 'GCC', 'CCT', 'CTC', 'TCC', 'CCC', 'CCG', 'CGC', 'GCT', 'CTT', 'TTT', 'TTC', 'TCC', 'CCA', 'CAT', 'ATC', 'TCC', 'CCG', 'CGC', 'GCA', 'CAT', 'ATT', 'TTC', 'TCC', 'CCA', 'CAT', 'ATG', 'TGG', 'GGC',
                        'GCA', 'CAG', 'AGG', 'GGT', 'GTG', 'TGA', 'GAG', 'AGT', 'GTC', 'TCT', 'CTG', 'TGA', 'GAT', 'ATT', 'TTA', 'TAT', 'ATT', 'TTC', 'TCG', 'CGA', 'GAA', 'AAG', 'AGG', 'GGA', 'GAG', 'AGG', 'GGC', 'GCT', 'CTG', 'TGG',
                        'GGA', 'GAG', 'AGT', 'GTG', 'TGT', 'GTG', 'TGG', 'GGG', 'GGC', 'GCG', 'CGG', 'GGA', 'GAG', 'AGG', 'GGG', 'GGC', 'GCA', 'CAG', 'AGC', 'GCG', 'CGC', 'GCC', 'CCA', 'CAG', 'AGG', 'GGT', 'GTT', 'TTT', 'TTC', 'TCC',
                        'CCC', 'CCA', 'CAA', 'AAT', 'ATC', 'TCA', 'CAG', 'AGA', 'GAT', 'ATT', 'TTT', 'TTG', 'TGC', 'GCT', 'CTC', 'TCA', 'CAG', 'AGG', 'GGG', 'GGT', 'GTC', 'TCC', 'CCC', 'CCT', 'CTC', 'TCC', 'CCA', 'CAG', 'AGC', 'GCA',
                        'CAG', 'AGT', 'GTC', 'TCC', 'CCA', 'CAT', 'ATG', 'TGC', 'GCC', 'CCG', 'CGC', 'GCA', 'CAG', 'AGA', 'GAG', 'AGG', 'GGC', 'GCT', 'CTG', 'TGT', 'GTC', 'TCC', 'CCC', 'CCT', 'CTT', 'TTG', 'TGG', 'GGG', 'GGG', 'GGG',
                        'GGC', 'GCC', 'CCC', 'CCA', 'CAC', 'ACG', 'CGC', 'GCA', 'CAT', 'ATC', 'TCC', 'CCT', 'CTA', 'TAG', 'AGC', 'GCC', 'CCA', 'CAC', 'ACG', 'CGG', 'GGC', 'GCC', 'CCT', 'CTC', 'TCC', 'CCT', 'CTC', 'TCA', 'CAC', 'ACG',
                        'CGT', 'GTC', 'TCC', 'CCA', 'CAT', 'ATG', 'TGC', 'GCG', 'CGG', 'GGG', 'GGG', 'GGA', 'GAT', 'ATT', 'TTT', 'TTG', 'TGC', 'GCG', 'CGC', 'GCC', 'CCC', 'CCT', 'CTG', 'TGG', 'GGA', 'GAA', 'AAG', 'AGG', 'GGA', 'GAG',
                        'AGC', 'GCC', 'CCG', 'CGC', 'GCC', 'CCC', 'CCG', 'CGG', 'GGC', 'GCT', 'CTG', 'TGC', 'GCC', 'CCT', 'CTC', 'TCT', 'CTC', 'TCG', 'CGC', 'GCC', 'CCA', 'CAA', 'AAC', 'ACA', 'CAT', 'ATG', 'TGC', 'GCA', 'CAG', 'AGC',
                        'GCA', 'CAC', 'ACT', 'CTT', 'TTC', 'TCC', 'CCC', 'CCT', 'CTT', 'TTC', 'TCC', 'CCT', 'CTT', 'TTT', 'TTC', 'TCC', 'CCA', 'CAT', 'ATG', 'TGG', 'GGA', 'GAG', 'AGC', 'GCA', 'CAC', 'ACG', 'CGG', 'GGT', 'GTT', 'TTC',
                        'TCC', 'CCT', 'CTG', 'TGT', 'GTC', 'TCC', 'CCC', 'CCG', 'CGG', 'GGG', 'GGG', 'GGG', 'GGT', 'GTC', 'TCC', 'CCA', 'CAT', 'ATA', 'TAT', 'ATT', 'TTG', 'TGG', 'GGC', 'GCC', 'CCA', 'CAC', 'ACT', 'CTG', 'TGT', 'GTG',
                        'TGG', 'GGG', 'GGA', 'GAG', 'AGA', 'GAG', 'AGA', 'GAG', 'AGT', 'GTC', 'TCG', 'CGG', 'GGG', 'GGC', 'GCA', 'CAG', 'AGC', 'GCT', 'CTG', 'TGA', 'GAA', 'AAT', 'ATT', 'TTC', 'TCC', 'CCC', 'CCG', 'CGC', 'GCA', 'CAG',
                        'AGG', 'GGT', 'GTG', 'TGG', 'GGG', 'GGA', 'GAA', 'AAT', 'ATG', 'TGC', 'GCC', 'CCA', 'CAG', 'AGG', 'GGG', 'GGC', 'GCC', 'CCC', 'CCG', 'CGA', 'GAG', 'AGG', 'GGA', 'GAT', 'ATG', 'TGT', 'GTT', 'TTG', 'TGC', 'GCC',
                        'CCC', 'CCC', 'CCT', 'CTG', 'TGT', 'GTC', 'TCC', 'CCT', 'CTG', 'TGA', 'GAA', 'AAG', 'AGG', 'GGC', 'GCT', 'CTG', 'TGT', 'GTC', 'TCG', 'CGC', 'GCC', 'CCC', 'CCG', 'CGA', 'GAT', 'ATC', 'TCG', 'CGC', 'GCT', 'CTC',
                        'TCT', 'CTA', 'TAT', 'ATC', 'TCC', 'CCA', 'CAA', 'AAG', 'AGG', 'GGC', 'GCT', 'CTG', 'TGC', 'GCC', 'CCC', 'CCT', 'CTG', 'TGG', 'GGG', 'GGG', 'GGC', 'GCA', 'CAG', 'AGC', 'GCG', 'CGT', 'GTC', 'TCA', 'CAC', 'ACC',
                        'CCT', 'CTG', 'TGG', 'GGG', 'GGG', 'GGG', 'GGT', 'GTC', 'TCC', 'CCT', 'CTG', 'TGC', 'GCG', 'CGG', 'GGG', 'GGG', 'GGG', 'GGC', 'GCT', 'CTT', 'TTC', 'TCT', 'CTC', 'TCA', 'CAG', 'AGC', 'GCA', 'CAC', 'ACA', 'CAG',
                        'AGC', 'GCA', 'CAT', 'ATC', 'TCC', 'CCA', 'CAG', 'AGC', 'GCA', 'CAC', 'ACT', 'CTG', 'TGC', 'GCC', 'CCA', 'CAC', 'ACC', 'CCT', 'CTA', 'TAG', 'AGT', 'GTG', 'TGT', 'GTG', 'TGT', 'GTT', 'TTC', 'TCC', 'CCC', 'CCG',
                        'CGT', 'GTC', 'TCA', 'CAC', 'ACG', 'CGT', 'GTC', 'TCT', 'CTC', 'TCC', 'CCT', 'CTC', 'TCC', 'CCC', 'CCC', 'CCC', 'CCC', 'CCG', 'CGC', 'GCC', 'CCT', 'CTG', 'TGC', 'GCA', 'CAC', 'ACC', 'CCA', 'CAG', 'AGG', 'GGC',
                        'GCA', 'CAC', 'ACC', 'CCA', 'CAG', 'AGA', 'GAG', 'AGA', 'GAC', 'ACC', 'CCC', 'CCG', 'CGG', 'GGA', 'GAT', 'ATG', 'TGC', 'GCC', 'CCA', 'CAA', 'AAG', 'AGG', 'GGC', 'GCC', 'CCT', 'CTG', 'TGT', 'GTC', 'TCA', 'CAG',
                        'AGC', 'GCT', 'CTT', 'TTC', 'TCC', 'CCT', 'CTC', 'TCA', 'CAA', 'AAT', 'ATG', 'TGG', 'GGG', 'GGA', 'GAA', 'AAA', 'AAC', 'ACT', 'CTT', 'TTT', 'TTT', 'TTC', 'TCT', 'CTT', 'TTC', 'TCA', 'CAG', 'AGT', 'GTG', 'TGA',
                        'GAA', 'AAC', 'ACA', 'CAA', 'AAA', 'AAG', 'AGC', 'GCT', 'CTC', 'TCT', 'CTG', 'TGT', 'GTT', 'TTT', 'TTT', 'TTA', 'TAT', 'ATA']
    self.assertEqual(tokenized, actual_tokenized)

    encoded = token.encode(sequence)
    actual_encoded = [75, 51, 80, 69, 24, 39, 32, 70, 28, 52, 20, 22, 30, 60, 54, 29, 58, 46, 63, 64, 71, 35, 83, 82, 78, 63, 64, 71, 34, 76, 52, 23, 35, 83, 80, 71, 35, 83, 82, 77, 56, 36, 21, 27, 50, 76, 53, 27, 50, 77, 59, 51, 82,
                      78, 60, 52, 22, 31, 67, 82, 78, 61, 58, 47, 64, 69, 24, 39, 33, 75, 51, 83, 81, 73, 40, 37, 26, 46, 60, 52, 20, 21, 27, 48, 71, 34, 76, 52, 20, 20, 23, 32, 69, 27, 50, 78, 63, 65, 72, 38, 29, 57, 43, 49, 75, 49,
                      72, 38, 31, 65, 72, 39, 35, 83, 81, 72, 39, 32, 68, 20, 20, 23, 33, 75, 51, 82, 76, 55, 35, 83, 83, 82, 79, 64, 71, 35, 80, 69, 24, 36, 21, 24, 36, 20, 23, 35, 80, 70, 30, 60, 55, 32, 71, 35, 83, 83, 81, 72, 38,
                      31, 64, 70, 28, 52, 21, 27, 50, 76, 55, 35, 82, 79, 64, 68, 22, 30, 61, 57, 40, 38, 28, 53, 27, 50, 79, 67, 82, 78, 60, 52, 23, 34, 79, 64, 68, 23, 35, 81, 73, 43, 51, 81, 73, 41, 40, 36, 20, 21, 24, 39, 33, 75,
                      49, 75, 49, 72, 38, 30, 62, 62, 61, 56, 37, 27, 51, 83, 81, 73, 43, 50, 76, 52, 21, 26, 46, 61, 59, 50, 77, 59, 50, 76, 53, 24, 38, 29, 56, 38, 29, 56, 39, 35, 83, 82, 76, 54, 30, 60, 53, 26, 46, 63, 66, 78, 62,
                      62, 61, 58, 46, 60, 54, 30, 60, 53, 24, 39, 33, 73, 43, 50, 78, 62, 62, 62, 62, 61, 57, 43, 50, 78, 61, 59, 51, 81, 75, 51, 82, 78, 62, 60, 52, 21, 27, 50, 78, 60, 54, 30, 61, 59, 51, 83, 82, 78, 61, 57, 41, 43,
                      51, 82, 79, 65, 73, 41, 40, 37, 25, 41, 41, 43, 50, 77, 59, 49, 73, 41, 41, 43, 50, 76, 54, 30, 60, 54, 30, 62, 60, 54, 30, 61, 58, 47, 66, 78, 62, 62, 61, 57, 41, 43, 50, 78, 62, 61, 59, 50, 78, 61, 59, 50, 77,
                      56, 36, 22, 28, 53, 26, 47, 66, 78, 60, 54, 31, 66, 76, 53, 27, 50, 79, 66, 78, 62, 63, 65, 73, 41, 41, 42, 47, 66, 78, 61, 57, 41, 41, 43, 50, 76, 53, 24, 39, 34, 77, 59, 49, 73, 41, 40, 38, 30, 62, 60, 52, 21,
                      25, 41, 40, 36, 22, 28, 52, 20, 20, 22, 28, 53, 27, 50, 76, 54, 28, 53, 25, 41, 43, 50, 79, 66, 78, 63, 66, 77, 57, 43, 49, 73, 41, 42, 45, 59, 51, 83, 81, 73, 40, 39, 33, 73, 42, 45, 56, 39, 35, 81, 73, 40, 39,
                      34, 78, 61, 56, 38, 30, 63, 66, 76, 54, 31, 65, 75, 50, 76, 55, 35, 80, 71, 35, 81, 74, 44, 52, 22, 30, 60, 54, 30, 61, 59, 50, 78, 60, 54, 31, 66, 79, 66, 78, 62, 61, 58, 46, 60, 54, 30, 62, 61, 56, 38, 29, 58,
                      45, 57, 40, 38, 30, 63, 67, 83, 81, 73, 41, 40, 36, 23, 33, 72, 38, 28, 55, 35, 83, 82, 77, 59, 49, 72, 38, 30, 62, 63, 65, 73, 41, 43, 49, 73, 40, 38, 29, 56, 38, 31, 65, 73, 40, 39, 34, 77, 57, 42, 45, 56, 38,
                      28, 54, 30, 61, 59, 50, 79, 65, 73, 41, 43, 51, 82, 78, 62, 62, 62, 61, 57, 41, 40, 37, 26, 45, 56, 39, 33, 73, 43, 48, 70, 29, 57, 40, 37, 26, 46, 61, 57, 43, 49, 73, 43, 49, 72, 37, 26, 47, 65, 73, 40, 39, 34,
                      77, 58, 46, 62, 62, 60, 55, 35, 83, 82, 77, 58, 45, 57, 41, 43, 50, 78, 60, 52, 22, 30, 60, 54, 29, 57, 42, 45, 57, 41, 42, 46, 61, 59, 50, 77, 57, 43, 49, 75, 49, 74, 45, 57, 40, 36, 21, 24, 39, 34, 77, 56, 38,
                      29, 56, 37, 27, 51, 81, 73, 41, 43, 51, 81, 73, 43, 51, 83, 81, 73, 40, 39, 34, 78, 60, 54, 29, 56, 37, 26, 46, 63, 67, 81, 73, 43, 50, 79, 65, 73, 41, 42, 46, 62, 62, 62, 63, 65, 73, 40, 39, 32, 71, 35, 82, 78,
                      61, 57, 40, 37, 27, 50, 79, 66, 78, 62, 60, 54, 28, 54, 28, 54, 31, 65, 74, 46, 62, 61, 56, 38, 29, 59, 50, 76, 52, 23, 35, 81, 73, 41, 42, 45, 56, 38, 30, 63, 66, 78, 62, 60, 52, 23, 34, 77, 57, 40, 38, 30, 62,
                      61, 57, 41, 42, 44, 54, 30, 60, 55, 34, 79, 67, 82, 77, 57, 41, 41, 43, 50, 79, 65, 73, 43, 50, 76, 52, 22, 30, 61, 59, 50, 79, 65, 74, 45, 57, 41, 42, 44, 55, 33, 74, 45, 59, 49, 75, 48, 71, 33, 73, 40, 36, 22,
                      30, 61, 59, 50, 77, 57, 41, 43, 50, 78, 62, 62, 61, 56, 38, 29, 58, 47, 65, 72, 37, 25, 43, 50, 78, 62, 62, 62, 63, 65, 73, 43, 50, 77, 58, 46, 62, 62, 62, 61, 59, 51, 81, 75, 49, 72, 38, 29, 56, 37, 24, 38, 29,
                      56, 39, 33, 73, 40, 38, 29, 56, 37, 27, 50, 77, 57, 40, 37, 25, 43, 48, 70, 31, 66, 79, 66, 79, 67, 81, 73, 41, 42, 47, 65, 72, 37, 26, 47, 65, 75, 49, 73, 43, 49, 73, 41, 41, 41, 41, 42, 45, 57, 43, 50, 77, 56,
                      37, 25, 40, 38, 30, 61, 56, 37, 25, 40, 38, 28, 54, 28, 53, 25, 41, 42, 46, 60, 55, 34, 77, 57, 40, 36, 22, 30, 61, 57, 43, 50, 79, 65, 72, 38, 29, 59, 51, 81, 73, 43, 49, 72, 36, 23, 34, 78, 62, 60, 52, 20, 21,
                      27, 51, 83, 83, 81, 75, 51, 81, 72, 38, 31, 66, 76, 52, 21, 24, 36, 20, 22, 29, 59, 49, 75, 50, 79, 67, 83, 83, 80, 71, 32]
    self.assertEqual(encoded, actual_encoded)

    decoded = token.decode(encoded)
    actual_decoded = "TCTTACATAGAAAGGAGCGGTATTTGGTATGAATTTATTTGCAACTGACTGCTTGGAAGTTGGCGTACATCTTTCCACGGAAACTATGAAAATACTGGTCAGCCTCTCAGTCATTTCATAAAATCTTGATTTTGTATTACAACAAATTAGGATATTTTCAGTAGAACTGATTGTAAGGCCAGACTGTTGGAATGTAATTCCTTCCCAAACATCTCTCAGGGGCACTTTCCTGAACGGCTGCTGACAGCAGCATTTGAGGACGGTGGGGCGGAGGACATCCTGGGGGGCCTGGCTTCTTGGGAACTGGAGGCTTTGGCCCTTGTCCCACCCCTGCTCCCCTGAGGAGGGAGGCGTGGGGCCCTGGGCTGGCTGCAAGACGTGGAGTGACTGTGGGTCCCCGTGGCCCCTGACATGCTCCCAGGGAACCCAAGAAAAGACTGAGACCCTGTGGTGCCTCCCGCTTTCCATCCGCATTCCATGGCAGGTGAGTCTGATTATTCGAAGGAGGCTGGAGTGTGGGCGGAGGGCAGCGCCAGGTTTCCCAATCAGATTTGCTCAGGGTCCCTCCAGCAGTCCATGCCGCAGAGGCTGTCCCTTGGGGGCCCACGCATCCTAGCCACGGCCTCCTCACGTCCATGCGGGGATTTGCGCCCTGGAAGGAGCCGCCCGGCTGCCTCTCGCCAACATGCAGCACTTCCCTTCCTTTCCATGGAGCACGGTTCCTGTCCCGGGGGTCCATATTGGCCACTGTGGGAGAGAGTCGGGCAGCTGAATTCCCGCAGGTGGGAATGCCAGGGCCCGAGGATGTTGCCCCTGTCCTGAAGGCTGTCGCCCGATCGCTCTATCCAAGGCTGCCCTGGGGCAGCGTCACCTGGGGGTCCTGCGGGGGCTTCTCAGCACAGCATCCAGCACTGCCACCTAGTGTGTTCCCGTCACGTCTCCTCCCCCCGCCTGCACCAGGCACCAGAGACCCGGATGCCAAGGCCTGTCAGCTTCCTCAATGGGAAACTTTTCTTCAGTGAACAAAGCTCTGTTTTATA"
    self.assertEqual(decoded, actual_decoded)

  def test_invalid_input(self):
    token = tokenizer(encoding="base_3k")
    with self.assertRaises(ValueError):
      token.encode("AGCTXXAGC")


if __name__ == "__main__":
  unittest.main()