def read_from_demo_txt_file(path):
    def read_until(f,symbol):
        line = read_without_nl(f)
        if line is None:
            return None 
        if line == symbol:
            return '' 
        res = line+'\n'
        while True:
            line = read_without_nl(f)
            assert line is not None
            if line == symbol:
                res = res[:-1]
                break
            res+=line+'\n'
        return res

    def read_without_nl(f):
        l = f.readline()
        if len(l)>0  :
            l = l.rstrip()
        else:
            return None
        return l

    def readline_with_assert(f,symbol):
        line = read_without_nl(f)
        assert line is not None and symbol == line
        return line

    l = []
    with open(path,"r",encoding='utf-8') as f:
        while True:
            q = read_until(f,'>>>')
            if q is None:
                break
            passage = read_until(f,'<<<')
            assert passage is not None
            l.append({'question':q,'passage':passage})
    return l
