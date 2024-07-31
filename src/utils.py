
def to_pure_python(dct, np_list=['weights', 'biases', 'upper_boundary', 'lower_boundary']):
    """
    takes a dict object and converts all numpy object to pure python object
    :param dct:
    :param np_list:
    :return:
    """

    dct2 = dict()
    for k in dct.keys():
        a = dct[k]
        print(type(a))
        if type(a) is np.ndarray:
            print("processing ", k)
            a = a.tolist()
            pass
        dct2[k] = a
        pass
    k = 'weights'
    dct2[k] = [a.tolist() for a in dct2[k]]
    k = 'biases'
    dct2[k] = [a.tolist() for a in dct2[k]]
    return dct2