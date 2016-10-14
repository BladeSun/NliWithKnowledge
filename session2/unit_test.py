__author__ = 'Suneburg'

import unittest
from data_iterator import TextIterator
from dam import *

class MyTestCase(unittest.TestCase):
    def setUp(self):
        base_path = '../data/'
        source_path = base_path + 'dev_h.tok'
        target_path = base_path + 'dev_t.tok'
        label_path = base_path + 'dev_label.tok'
        dict_path = base_path + 'snli_dict.pkl'
        self.testiter = TextIterator(source=source_path, target=target_path, label=label_path, all_dict=dict_path, batch_size=4)

    def test_iter(self):
        m = self.testiter.__iter__()
        s,t,l = m.next()
        print s
        print t
        print l
        #self.assertEqual(True, False)

    def test_prepare_data(self):
        m = self.testiter.__iter__()
        s,t,l = m.next()
        x, x_mask, y, y_mask, l = prepare_data(s, t, l)
        print x 
        print x_mask
        print y
        print y_mask
        print l


if __name__ == '__main__':
    #unittest.main()
    suite = unittest.TestSuite()
    suite.addTest(MyTestCase("test_prepare_data"))
    runner = unittest.TextTestRunner()
    runner.run(suite)
