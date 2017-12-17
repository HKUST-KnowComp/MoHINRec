#coding=utf8
'''
    utils for str processing
'''

def unicode2str(str_):
    if isinstance(str_, unicode):
        return str_.encode('utf8')
    return str_

def str2unicode(str_):
    if isinstance(str_, str):
        return str_.decode('utf8')
    return str_

if __name__ == '__main__':
    main()
