import unittest

from leaguesync.util import *

class TestUtil(unittest.TestCase):
    def test_pathinterp(self):
        v = path_interp('/api/v2/desk/people/:person_id/visits', person_id=10, updated='foo')
        self.assertEqual(v[0], '/api/v2/desk/people/10/visits')
        self.assertEqual(v[1], {'updated': 'foo'})



if __name__ == '__main__':
    unittest.main()
