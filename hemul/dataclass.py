class Myarr:
    """Custom vector dataset.
        Addtion, multiplication, (rotation, and conjugation) are supported.

        Separated from scheme module since it's an ad-hoc component. 
    """
    def __init__(self, arr):
        try:
            self._data = [float(a) for a in arr]
        except:
            print(f"{arr} doesn't seem to be valid numeric")

    def __add__(self, o):
        return [dd + oo for dd, oo in zip(self._data, o._data)]

    def __mul__(self, o):
        return [dd * oo for dd, oo in zip(self._data, o._data)]

    """
    def rotate # but, implement here or outside?


    def conjugate
    """