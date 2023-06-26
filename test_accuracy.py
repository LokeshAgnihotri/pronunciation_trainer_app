import unittest
from accuracy import *


class MyTestCase(unittest.TestCase):
    def test_word_errors(self):
        word_error_rate, len = word_errors("Hello World", "Hello World")
        self.assertEqual(word_error_rate, 0)
        self.assertEqual(len, 2)

    def test_word_errors2(self):
        word_error_rate, len = word_errors("Hello World", "Hello")
        self.assertEqual(word_error_rate, 1)
        self.assertEqual(len, 2)

    def test_word_errors3(self):
        word_error_rate, len = word_errors("Hello World", "Hello World!")
        self.assertEqual(word_error_rate, 1)
        self.assertEqual(len, 2)

    def test_word_errors4(self):
        word_error_rate, len = word_errors("Hello World", "Hello World")
        self.assertEqual(word_error_rate, 0)
        self.assertEqual(len, 2)

    def test_word_errors5(self):
        word_error_rate, len = word_errors("", "Hello World")
        self.assertEqual(word_error_rate, 2)
        self.assertEqual(len, 1)

    def test_word_errors6(self):
        word_error_rate, len = word_errors("Hello World", "")
        self.assertEqual(word_error_rate, 2)
        self.assertEqual(len, 2)

    def test_char_errors(self):
        char_error_rate, len = char_errors("Hello World", "Hello World")
        self.assertEqual(char_error_rate, 0)
        self.assertEqual(len, 11)

    def test_char_errors2(self):
        char_error_rate, len = char_errors("Hello World", "Hello")
        self.assertEqual(char_error_rate, 6)
        self.assertEqual(len, 11)

    def test_char_errors3(self):
        char_error_rate, len = char_errors("Hello World", "Hello World!")
        self.assertEqual(char_error_rate, 1)
        self.assertEqual(len, 11)

    def test_char_errors4(self):
        char_error_rate, len = char_errors("Hello World", "Hello World")
        self.assertEqual(char_error_rate, 0)
        self.assertEqual(len, 11)

    def test_char_errors5(self):
        char_error_rate, len = char_errors("", "Hello World")
        self.assertEqual(char_error_rate, 11)
        self.assertEqual(len, 0)

    def test_char_errors6(self):
        char_error_rate, len = char_errors("Hello World", "")
        self.assertEqual(char_error_rate, 11)
        self.assertEqual(len, 11)

    def test_wer(self):
        word_error_rate = wer("Hello World", "Hello World")
        self.assertEqual(word_error_rate, 0)

    def test_wer2(self):
        word_error_rate = wer("Hello World", "Hello")
        self.assertEqual(0.5, word_error_rate)

    def test_wer3(self):
        word_error_rate = wer("Hello World", "Hello World!")
        self.assertEqual(0.5, word_error_rate)

    def test_wer4(self):
        word_error_rate = wer("Hello World", "Hello")
        self.assertEqual(0.5, word_error_rate)

    def test_wer5(self):
        word_error_rate = wer("", "Hello World")
        self.assertEqual(2, word_error_rate)

    def test_wer6(self):
        word_error_rate = wer("Hello World", "")
        self.assertEqual(1, word_error_rate)

    def test_cer(self):
        char_error_rate = cer("Hello World", "Hello World")
        self.assertEqual(char_error_rate, 0)

    def test_cer2(self):
        char_error_rate = cer("Hello World", "Hello")
        char_error_rate = np.round(char_error_rate, 2)
        self.assertEqual(.55, char_error_rate)

    def test_cer3(self):
        char_error_rate = cer("Hello World", "Hello World!")
        char_error_rate = np.round(char_error_rate, 2)
        self.assertEqual(.09, char_error_rate)

    def test_cer4(self):
        char_error_rate = cer("Hello World", "Hello")
        char_error_rate = np.round(char_error_rate, 2)
        self.assertEqual(0.55, char_error_rate)

    def test_cer5(self):
        with self.assertRaises(ValueError) as context:
            char_error_rate = cer("", "Hello World")
        self.assertTrue('Length of reference should be greater than 0.' in str(context.exception))

    def test_cer6(self):
        char_error_rate = cer("Hello World", "")
        char_error_rate = np.round(char_error_rate, 2)
        self.assertEqual(1, char_error_rate)

    def test_word_error(self):
        word_error_rate, len = word_errors("Hello World", "Bye World")
        self.assertEqual(word_error_rate, 1)
        self.assertEqual(len, 2)


if __name__ == '__main__':
    unittest.main()
