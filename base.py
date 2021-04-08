import pandas as pd
import pingouin as pg
import numpy as np


def anova_onoff(on, off, subjects, columns):
    off = pd.DataFrame(data=np.insert(off,
                                      0,
                                      np.arange(len(subjects)),
                                      axis=1),
                       columns=columns[:-1])
    off = pd.melt(off,
                  id_vars=['sub'],
                  value_vars=columns[1:-1],
                  var_name='block',
                  value_name='RT')
    off.insert(1, 'Triplet', np.zeros(len(off)))
    on = pd.DataFrame(data=np.insert(on,
                                     0,
                                     np.arange(len(subjects)),
                                     axis=1),
                      columns=columns)
    on = pd.melt(on,
                 id_vars=['sub'],
                 value_vars=columns[1:],
                 var_name='block',
                 value_name='RT')
    on.insert(1, 'Triplet', np.ones(len(on)))
    anova_onoff = pd.concat([on, off])
    aov_stats = pg.rm_anova(data=anova_onoff, dv='RT',
                            within=['block', 'Triplet'], subject='sub')
    return aov_stats
