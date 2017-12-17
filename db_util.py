#coding=utf8
'''
    utils for operating with sqlite3
'''
import sqlite3
import time

from StringIO import StringIO

dir_ = 'data/yelp/'

class DB(object):

    def __init__(self, db_filename, is_from_dump=False):
        if is_from_dump:
            self.load_from_dump(db_filename)
        else:
            self.conn = sqlite3.connect(db_filename)
            self.cur = self.conn.cursor()

    def load_from_dump(self, db_filename):
        start = time.time()
        conn1 = sqlite3.connect(db_filename)
        self.conn = sqlite3.connect(':memory:')
        self.cur = self.conn.cursor()
        #self.cur.executescript(''.join([l.strip() for l in open(db_filename, 'r').readlines()]))
        sql = ''.join(l for l in conn1.iterdump())
        self.cur.executescript(sql)
        #self.cur.executemany(sql,[])
        self.conn.commit()
        self.conn.row_factory = sqlite3.Row
        print 'finish loading db from %s, cost %smin' % (db_filename, (time.time() - start) / 60.0)
        test_sql = 'select * from relation where tail_id = 20062 and rel_id = 4 limit 10'
        print self.get_list_by_sql(sql)

    def insert_by_sql(self, sql, values):
        self.cur.executemany(sql, values)
        self.conn.commit()
        print 'insert %s records' % len(values)

    def batch_insert_by_sql(self, sql, values, batch_size=10000):
        if batch_size > len(values):
            self.insert_by_sql(sql,values)
        else:
            num = len(values) / batch_size
            for n in range(num+1):
                l = n * batch_size
                r = (n+1) * batch_size
                self.insert_by_sql(sql,values[l:r])
                print 'batch insert from %s to %s, save %s records' % (l, r, len(values[l:r]))
        print 'finish batch inserting %s records' % len(values)

    def get_list_by_sql(self, sql):
        self.cur.execute(sql)
        return self.cur.fetchall()

    def get_list_by_paras(self,sql,paras):
        self.cur.execute(sql,paras)
        return self.cur.fetchall()

    def dump_db(self, wfilename):
        '''
            dump db into file, for later loaded as in-memory one
        '''
        fw = open(dir_ + wfilename,'w+')
        for line in self.conn.iterdump():
            fw.write('%s\n' % line)
        self.conn.close()
        fw.close()

if __name__ == '__main__':
    db_filename = dir_ + 'yelp2016_hin.db'
    dump_filename = dir_ + 'yelp2016_hin.dump'
    db = DB(db_filename, is_from_dump=True)
