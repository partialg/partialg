import re
import numpy as np
from tqdm import tqdm

class dok:
    ''' 
    Dictionary Of Keys TensOR.
    '''
    def __init__(self, *a, shape=tuple() , tolerance= 10**-9):
        #
        def asdok(a, tolerance = 10**-9):
            ''' '''
            ty = str(type(a))
            if 'dict' in ty or 'dok' in ty: # For dictionaries and doks
                return a
            elif 'array' in ty or 'list' in ty or 'tuple' in ty: # For dense array-like
                d = dict()
                for k, i in np.ndenumerate(a):
                    if i > tolerance:
                        d.update({ k:i })
                return d
        #
        if len(a) != 0 :
            self.d = asdok(a[0], tolerance=tolerance)
            self.shape = shape
        else:
            self.d = dict() 
        self.tol = tolerance
        self.len = len(self.d)

    def __repr__(self):
        return f'dokenstein.dok with shape {self.shape}'

    def __len__(self):
        return len(self.d)
    
    def __str__(self):
        return str(self.d)

    def __iter__(self):
        return iter(self.d)  # Iterate over keys, like a dict

    def __setitem__(self, key, value):
        self.d[key] = value

    def __getitem__(self, key):
        if key in self.d:
            return self.d[key]
        else:
            return 0.

    def __add__(self, other):
        anew = dict()
        if isinstance(other, dok):
            for i in other:
                anew[i] = self.d.get(i, 0) + other[i]
            return dok(anew, shape=self.shape)
        return NotImplemented

    def __radd__(self, other):
        anew = dict()
        if isinstance(other, dok):
            for i in other:
                anew[i] = self.d.get(i, 0) + other[i]
            return dok(anew, shape=self.shape)
        return NotImplemented

    def __sub__(self, other):
        anew = dict()
        if isinstance(other, dok):
            for i in other:
                anew[i] = self.d.get(i, 0) - other[i]
            return dok(anew, shape=self.shape)
        return NotImplemented
    
    def __rsub__(self, other):
        anew = dict()
        if isinstance(other, dok):
            for i in other:
                anew[i] = other[i] - self.d.get(i, 0)
            return dok(anew, shape=self.shape)
        return NotImplemented

    def __mul__(self, scalar):
        anew = dict()
        if isinstance(scalar, (int, float)):  # Check if scalar is an int or float
            for i in self.d:
                anew[i] = self.d[i] * scalar
            return dok(anew, shape=self.shape)
        return NotImplemented

    def __rmul__(self, scalar):
        anew = dict()
        if isinstance(scalar, (int, float)):  # Check if scalar is an int or float
            for i in self.d:
                anew[i] = self.d[i] * scalar
            return dok(anew, shape=self.shape)
        return NotImplemented

    def get_memory(self):
        '''Returns estimated memory used by self, in MB'''
        from sys import getsizeof
        mem = getsizeof(self.d)
        for i in self.d:
            mem += getsizeof(i)*len(self.d)
            mem += getsizeof(self.d[i])*len(self.d)
            break
        return float( np.round(mem / (1024 * 1024), 4) )

    def items(self):
        return self.d.items()

    def keys(self):
        return self.d.keys()

    def values(self):
        return self.d.values()

    def to_array(self, *null, **nulls):
        ar = np.zeros(self.shape)
        for i in self.d:
            ar[i] = self.d[i]
        #
        return ar
    #
    def add(self, *args):
        'Adds doks of same shape.'
        if np.all([adok.shape == self.shape for adok in args]) != True:
            raise Warning('ADD SKIPPED. Shape mismatch.')
            return self.d
        #
        anew = dict()
        for adok in args:
            for i in adok:
                anew[i] = self.d.get(i, 0) + adok[i]
        #
        return dok(anew, shape=self.shape)

    def absolute(self):
        anew = dict()
        for i in self.d:
            anew[i] = abs( self.d[i] )
        return dok(anew, shape=self.shape)
    
    def apply_tolerance(self, tol):
        self.tol = tol
        to_del   = []
        for i in self.d:
            if self.d[i] < tol:
                to_del.append(i)
        #
        for idx in to_del:
            del self.d[idx]

    def apply_absolute_tolerance(self, tol):
        self.tol = tol
        to_del   = []
        for i in self.d:
            if abs(self.d[i]) < tol:
                to_del.append(i)
        #
        for idx in to_del:
            del self.d[idx]
    
    def similarity(self, inp, out, *null, **nulls):
        ''' Computes lists of equal indices (left, right) and transposes unique indices according to out.
        Test:
        inp = 'ijklmjkk' 
        out = 'mli'
        returns positions of indices (left) that are equal to other indices (right)
        '''
        left    = []
        right   = []
        toout   = []
        toout_s = ''
        #
        gone = set()
        for i_, i in enumerate(inp):
            if i not in gone:
                gone = gone.union({i})
                s = [m.start() for m in re.finditer(i, inp )][1:]
                if len(s) != 0 :
                    right.append(s)
                    left.append(i_)
                else: 
                    toout.append(i_)
                    toout_s = toout_s + i
        #
        # Transposing toout based on out
        toout_i    = [out.find( i ) for i in toout_s]
        toout      = np.array(toout)[toout_i]
        #
        return left, right, toout
    #
    def bigif(self, d, left, right, *null, **nulls):
        ''' Checks if d at each index of left is equal to d at each index of right.'''
        out = []
        for l_, l in enumerate(left):
            for r in right[l_]:
                out.append( d[l] == d[r] )
        return np.all(out)
    #
    def einsum(self, command, *B, **kwargs):
        ''' 
        Numpy-inspired einsum, but for doks.
        Warnings: 
            - Any symbol other than ',' and '->' is considered an index, even space.
            - A, B must be tensors as doks.
            - Mismatch between dimensions of commanded indices and actual output shape may generate errors or wrong result.
        1-input example:
            ij->ji
        2-input example:
            ijk,nkpq->ijnqp
        '''
        #
        inp,  out          = command.split('->')
        left, right, toout = self.similarity( inp=inp.replace(',',''),  out=out)
        C                  = dict()
        A = self.d
        #
        A_shape = self.shape
        C_shape = list(A_shape)
        if len(B) != 0:
            C_shape = C_shape + list(B[0].shape)
        #
        for a in tqdm(A):
            lab = list(a)
            if len(B) != 0: # Contractions + transpositions in A*B
                for b in B[0]:
                    lab     = list(a) + list(b)
                    if self.bigif( lab, left, right ) == True:
                        #
                        c_idx    = tuple([ lab[ it ] for it in toout])
                        C[c_idx] = C.get(c_idx, 0) + A[a]*B[0][b]
            #
            else: # Only transpositions in A
                c_idx    = tuple([ lab[ it ] for it in toout])
                C[c_idx] = C.get(c_idx, 0) + A[a]
        #
        C_shape = tuple([ C_shape[ it ] for it in toout])
        C       = dok(C)
        C.shape = C_shape
        #
        return C



'''
# TESTS

AA = np.random.rand(4,4)
d  = dok( AA , shape=AA.shape )
print( (5*d) - d)

len(d)
print(d)


d.apply_tolerance(tol=10**0)


d.keys()


d = d.add(d)


d

d = d.einsum('ijk,ijm->km', d)

d.shape
d.get_memory()

d[10, 10, 2] = 0.

(10,10,2) in d


AA_ = np.einsum('ijk,ijm->km', AA, AA)
AA_ - d.to_array()
'''
