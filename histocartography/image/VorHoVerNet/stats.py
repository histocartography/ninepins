import csv
import numpy as np
import matplotlib.pyplot as plt

class mlrun_col:
    def __init__(self, mr, key=None):
        if key is not None:
            self.data = [row[key] for row in mr]
        else:
            self.data = mr

    def __eq__(self, cri):
        return mlrun_index([i for i, val in enumerate(self.data) if val == cri], len(self))

    def startswith(self, cri):
        return mlrun_index([i for i, val in enumerate(self.data) if val.startswith(cri)], len(self))

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return self.data.__iter__()

    def __repr__(self):
        r = ""
        for row in self.data:
            r += row + '\n'
        return r

    def astype(self, func):
        return mlrun_col(list(map(func, self)))

    def tolist(self):
        return self.data

class mlrun_index:
    def __init__(self, indexes, len_):
        self.indexes = indexes
        self.len_ = len_

    def __and__(self, idxs):
        assert len(self) == len(idxs), "indexes should have the same shape"
        return mlrun_index([i for i in self.indexes if i in idxs], len(self))

    def __or__(self, idxs):
        assert len(self) == len(idxs), "indexes should have the same shape"
        m = max(self.indexes[-1], idxs[-1])
        return mlrun_index([i for i in range(m) if (i in self.indexs) or (i in idxs)], len(self))

    def __getitem__(self, ind):
        return self.indexes[ind]

    def __in__(self, ele):
        return ele in self.indexes

    def __iter__(self):
        return self.indexes.__iter__()

    def __len__(self):
        return self.len_

    def __repr__(self):
        return self.indexes.__repr__()

class mlruns:
    def __init__(self, filename=None):
        if filename is not None:
            with open(filename, newline='') as csvfile:
                self.data = list(csv.DictReader(csvfile))

    @staticmethod
    def fromrows(rows):
        m = mlruns()
        m.data = rows
        return m

    def all(self):
        return self.data

    def __getitem__(self, col):
        if isinstance(col, str):
            try:
                return mlrun_col(self, col)
            except KeyError:
                raise KeyError("unknown key: {}".format(col))
        if isinstance(col, mlrun_index):
            assert len(self) == len(col), "indexes should have the same shape as mlruns"
            return self.fromrows([self.data[i] for i in col])

    def __len__(self):
        return len(self.data)

    def __repr__(self):
        if len(self) == 0: return ""
        r = ""
        keys = self.data[0].keys()
        max_col = 5
        short = len(keys) > max_col
        for i, key in enumerate(keys):
            if short and i == max_col:
                r += '...'
                break
            else:
                r += key + '\t'
        r += "\n"
        for row in self.data:
            for i, val in enumerate(row.values()):
                if short and i == max_col:
                    r += '...'
                    break
                else:
                    if len(val) > 20:
                        val = val[:20] + "..."
                    r += val + '\t'
            r += "\n"
        r += "\n"
        return r

    def __iter__(self):
        return self.data.__iter__()

def stats(vals):
    vals = vals.astype(float).tolist()
    avg = np.mean(vals)
    std = np.std(vals)
    return {'mean': avg, 'std': std}

def stats_str(s):
    return "mean: {:.4f}, std: {:.4f}".format(s['mean'], s['std'])

def extract_metrics(m, d):
    for eps in range(11):
        eps = eps * 2e-01
        eps_str = 'm_eps{:.1f}'.format(eps) if eps != 0 else 'm_eps0_'
        rows = m[(m['ckpt-filename'].startswith(eps_str)) & (m['Status'] == 'FINISHED') & (m['version'] == '5')]
        # find model with certain epsilon value
        if len(rows) == 5: # if all 5 runs are finished
            s_DICE = stats(rows['average_DICE'])
            s_DQ = stats(rows['average_DQ'])
            # print("{:.1f}".format(eps), "DICE:", stats_str(s_DICE), "DQ:", stats_str(s_DQ))
            d["{:.1f}".format(eps)] = {"DICE": s_DICE, "DQ": s_DQ}

def test():
    # m = mlruns("mixed_pseu.csv")
    m1 = mlruns("runs.csv")
    # m2 = mlruns("runs (1).csv")
    m3 = mlruns("runs (3).csv")
    
    # rows = m[(m['ckpt-filename'].startswith('m_PseuRate1.0')) & (m['Status'] == 'FINISHED') & (m['version'] == '5')]

    # metrics = {"0.0": {"DICE": stats(rows['average_DICE']), "DQ": stats(rows['average_DQ'])}}
    metrics = {}
    extract_metrics(m1, metrics)
    # extract_metrics(m2, metrics)
    extract_metrics(m3, metrics)
    print(metrics.keys())
    lbls = []
    ys1 = []
    yss1 = []
    ys2 = []
    yss2 = []
    for eps in range(11):
        eps = eps * 2e-01
        eps = "{:.1f}".format(eps)
        lbls.append(eps)
        if eps not in metrics: continue
        stat = metrics[eps]
        ys1.append(stat['DICE']['mean'])
        yss1.append(stat['DICE']['std'])
        ys2.append(stat['DQ']['mean'])
        yss2.append(stat['DQ']['std'])
        # print(eps, "DICE:", stats_str(stat['DICE']), "DQ:", stats_str(stat['DQ']))
    plt.grid(linestyle='--')
    plt.errorbar(range(11), ys1, yerr=yss1, ecolor='red', barsabove=True, capsize=2)
    plt.ylim(0.6, 0.8)
    plt.xticks(range(11), lbls)
    plt.savefig('test1.png')
    plt.clf()
    plt.grid(linestyle='--')
    plt.errorbar(range(11), ys2, yerr=yss2, ecolor='red', barsabove=True, capsize=2)
    plt.ylim(0.3, 0.5)
    plt.xticks(range(11), lbls)
    plt.savefig('test2.png')

if __name__ == "__main__":
    test()