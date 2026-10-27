from email.message import Message
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError
import unittest

from rl_pm.http_client import get_text, retry_seconds


class BackoffTests(unittest.TestCase):
    def test_retry_after_seconds(self):
        self.assertEqual(retry_seconds('600', 300), 600)
        self.assertEqual(retry_seconds('invalid', 300), 300)

    def test_rate_limit_retries_same_request_after_wait(self):
        headers = Message()
        headers['Retry-After'] = '120'
        limited = HTTPError('https://example.com', 429, 'limited', headers, None)
        response = MagicMock()
        response.__enter__.return_value.read.return_value = b'prices'
        with patch('rl_pm.http_client.urlopen', side_effect=[limited, response]) as request:
            with patch('rl_pm.http_client.time.sleep') as sleep:
                self.assertEqual(get_text('https://example.com', 'test'), 'prices')
                self.assertEqual(request.call_count, 2)
                self.assertEqual(sum(call.args[0] for call in sleep.call_args_list), 120)

    def test_forbidden_is_not_retried(self):
        error = HTTPError('https://example.com', 403, 'forbidden', Message(), None)
        with patch('rl_pm.http_client.urlopen', side_effect=error) as request:
            with self.assertRaises(HTTPError):
                get_text('https://example.com', 'test')
            self.assertEqual(request.call_count, 1)
