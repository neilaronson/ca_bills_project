import re
from sys import argv

def check_line(line, p):
    if re.search(p, line):
        return True
    else:
        return False

def main():
    filename = argv[1]
    p = re.compile(r'changing\s.*\sbased\son\sfile\s.*\.lob')
    with open(filename) as f:
        for i, line in enumerate(f):
            if not check_line(line, p):
                print "problem with line {}, which starts with {}...".format(i,line[:10])

if __name__ == '__main__':
    main()
