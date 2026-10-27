import unittest
from rl_pm.scrape_prices import parse_history


class ScrapeTests(unittest.TestCase):
    def test_public_table_with_nested_cells(self):
        html = '<title>Retired Co</title><table><tr>' + ''.join(
            f'<th>{x}</th>' for x in ['Date', 'Open', 'High', 'Low', 'Close', 'Adj. Close', 'Change', 'Volume'])
        html += '</tr><tr>' + ''.join(f'<td><span>{x}</span></td>' for x in
            ['May 20, 2025', '24.63', '24.69', '24.63', '24.66', '24.66', '0.08%', '50,704,421']) + '</tr></table>'
        rows, title = parse_history(html)
        self.assertEqual(rows[0]['date'], '2025-05-20')
        self.assertEqual(rows[0]['volume'], 50704421)
        self.assertEqual(title, 'Retired Co')

    def test_block_or_login_page_is_not_price_data(self):
        with self.assertRaises(ValueError):
            parse_history('<html>Please sign in</html>')
