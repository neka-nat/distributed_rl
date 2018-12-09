import numpy as np

class Node(object):
    def __init__(self, value=0.0, left=None, right=None, op=sum):
        self.value = value
        self.left = left
        self.right = right
        self.op = op

    def reset(self, value=0.0):
        self.value = value
        self.left = None
        self.right = None

    def _expand(self):
        if self.left is None and self.right is None:
            self.left = Node(op=self.op)
            self.right = Node(op=self.op)

    def _reduce(self):
        values = []
        if not self.left is None:
            values.append(self.left.value)
        if not self.right is None:
            values.append(self.right.value)
        self.value = self.op(values) if len(values) > 0 else 0.0

    def _write(self, index_left, index_right, key, value):
        if index_right - index_left == 1:
            ret = self.value
            self.reset(value)
        else:
            self._expand()
            index_center = (index_left + index_right) // 2
            if key < index_center:
                ret = self.left._write(index_left, index_center, key, value)
            else:
                ret = self.right._write(index_center, index_right, key, value)
            self._reduce()
        return ret

    def _get(self, index_left, index_right, key):
        if index_right - index_left == 1:
            return self.value
        else:
            index_center = (index_left + index_right) // 2
            if key < index_center:
                return self.left._get(index_left, index_center, key)
            else:
                return self.right._get(index_center, index_right, key)

    def _find(self, index_left, index_right, pos):
        if index_right - index_left == 1:
            return index_left
        else:
            index_center = (index_left + index_right) // 2
            left_value = self.left.value if not self.left is None else 0.0
            if pos < left_value:
                return self.left._find(index_left, index_center, pos)
            else:
                return self.right._find(index_center, index_right, pos - left_value)


class TreeQueue(object):
    def __init__(self, op):
        self.length = 0
        self.op = op

    def __setitem__(self, ix, val):
        self._write(ix, val)

    def __getitem__(self, ix):
        ixl, ixr = self.bounds
        return self.root._get(ixl, ixr, ix)

    def _write(self, ix, val):
        ixl, ixr = self.bounds
        return self.root._write(ixl, ixr, ix, val)

    def append(self, value):
        if self.length == 0:
            self.root = Node(value, op=self.op)
            self.bounds = (0, 1)
            self.length = 1
            return

        ixl, ixr = self.bounds
        root = self.root
        if ixr == self.length:
            self.root = Node(root.value, self.root, Node(op=self.op),
                             op=self.op)
            ixr += ixr - ixl
            self.bounds = (ixl, ixr)
        ret = self._write(self.length, value)
        self.length += 1

    def extend(self, values):
        for v in values:
            self.append(v)

    def popleft(self):
        assert self.length > 0
        ret = self._write(0, 0.0)
        ixl, ixr = self.bounds
        ixl -= 1
        ixr -= 1
        self.length -= 1
        if self.length == 0:
            self.root = None
            self.bounds = None
            return ret

        ixc = (ixl + ixr) // 2
        if ixc == 0:
            ixl = ixc
            self.root = self.root.right
        self.bounds = ixl, ixr
        return ret

    def __len__(self):
        return self.length

class SumTree(TreeQueue):
    def __init__(self):
        super(SumTree, self).__init__(op=sum)

    def prioritized_sample(self, n):
        assert n >= 0
        ixs = []
        vals = []
        if n > 0:
            ixl, ixr = self.bounds
            for _ in range(n):
                ix = self.root._find(ixl, ixr, np.random.uniform(0.0, self.root.value))
                val = self[ix]
                ixs.append(ix)
                vals.append(val)
        return ixs, vals
