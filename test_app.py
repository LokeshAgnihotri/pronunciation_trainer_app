import unittest

import eng_to_ipa

from app import app
from flask import jsonify


class AppTestCase(unittest.TestCase):
    def setUp(self):
        self.ctx = app.app_context()
        self.ctx.push()
        self.client = app.test_client()

    def tearDown(self):
        self.ctx.pop()

    def test_home(self):
        response = self.client.get("/")
        assert response.status_code == 200

    def test_getAudio_file(self):
        response = self.client.get("/reference_recordings/lemon_is_a_fruit")
        assert response.status_code == 200
        assert response.headers["Content-Type"] == "audio/mpeg"

    def test_random_word(self):
        response = self.client.get("/next_word")
        assert response.status_code == 200
        assert response.json["random_word"] is not None
        assert response.json["random_word_ipa"] is not None


if __name__ == "__main__":
    unittest.main()
