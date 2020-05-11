import csv
import numpy as np
import matplotlib.pyplot as plt

from histocartography.image.VorHoVerNet.stats import mlruns, stats, stats_str

def extract_metrics(m, d, start_with, ver):
    global ms
    for eps in range(11):
        eps = eps / 10.0
        rows = m[(m['ckpt-filename'].startswith(f'{start_with}{eps:.1f}')) & (m['Status'] == 'FINISHED') & (m['version'] == str(ver))]
        # print('len of rows and eps:', len(rows), eps)
        # if len(rows) == 5:
        # s_m1 = stats(rows[f'average_{ms[0]}'])
        # s_m2 = stats(rows[f'average_{ms[1]}'])
        # print("{:.1f}".format(eps), f"{ms[0]}:", stats_str(s_m1), "ms[1]:", stats_str(s_m2))
        # d["{:.1f}".format(eps)] = {f"{ms[0]}": stats(rows[f'average_{ms[0]}']), f"{ms[1]}": stats(rows[f'average_{ms[1]}'])}
        d["{:.1f}".format(eps)] = {metric: stats(rows['average_' + metric]) for metric in ms}

    # rows = m[(m['ckpt-filename'].startswith(f'{start_with}')) & (m['Status'] == 'FINISHED') & (m['version'] == str(ver))]
    # # if len(rows) == 5:
    # s_DICE = stats(rows['average_DICE'])
    # s_DQ_point = stats(rows['average_DQ_point'])
    # # print("{:.1f}".format(eps), "DICE:", stats_str(s_DICE), "DQ_point:", stats_str(s_DQ_point))
    # d[f"{start_with}"] = {"DICE": s_DICE, "DQ_point": s_DQ_point}

def autolabel(rects, ax, xpos='center'):
    """
    Attach a text label above each bar in *rects*, displaying its height.

    *xpos* indicates which side to place the text w.r.t. the center of
    the bar. It can be one of the following {'center', 'right', 'left'}.
    """

    ha = {'center': 'center', 'right': 'left', 'left': 'right'}
    offset = {'center': 0, 'right': 1, 'left': -1}

    for rect in rects:
        height = rect.get_height()
        ax.annotate('{:.4f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(offset[xpos]*3, 3),  # use 3 points offset
                    textcoords="offset points",  # in both directions
                    ha=ha[xpos], va='bottom')

def main(filenames, ver):
    global ms
    print(f'version is {ver}')
    metrics1 = {}
    metrics2 = {}
    # metrics3 = {}
    # for fn in filenames:
    extract_metrics(mlruns(filenames[0]), metrics1, start_with='m_PseuRate', ver=ver)
    extract_metrics(mlruns(filenames[1]), metrics2, start_with='m_dot_PseuRate', ver=ver)
    # print(metrics1)
    # extract_metrics(mlruns(filenames[0]), metrics1, start_with='m_CoNSeP', ver=ver)
    # extract_metrics(mlruns(filenames[1]), metrics2, start_with='m_PseuRate1.0', ver=ver)
    # extract_metrics(mlruns(filenames[0]), metrics3, start_with='m_CoNuSeg', ver=ver)

    # for eps in range(11):
    #     eps = eps / 10.0 
    #     eps = "{:.1f}".format(eps)
    #     if eps not in metrics: continue
    #     stat = metrics[eps]
    #     print(eps, "DICE:", stats_str(stat['DICE']), "DQ:", stats_str(stat['DQ']))
    
    # v1 = [metrics1['m_CoNSeP'][t][k] for k in ['mean',  'std'] for t in ['DICE', 'DQ_point'] ]
    # v2 = [metrics2['m_PseuRate1.0'][t][k] for k in ['mean',  'std'] for t in ['DICE', 'DQ_point'] ]
    # v3 = [metrics3['m_CoNuSeg'][t][k] for k in ['mean',  'std'] for t in ['DICE', 'DQ_point'] ]

    # print(v1, '\n', v2, '\n', v3)
    
    # ind = np.arange(2)  # the x locations for the groups
    # width = 0.2  # the width of the bars

    # fig, ax = plt.subplots()
    # rects1 = ax.bar(ind - width, v1[:2], width, yerr=v1[2:],
    #                 label='CoNSeP')
    # rects2 = ax.bar(ind, v2[:2], width, yerr=v2[2:],
    #                 label='MoNuSeg')
    # rects3 = ax.bar(ind + width, v3[:2], width, yerr=v3[2:],
    #                 label='CoNuSeg')

    # autolabel(rects1, ax, "left")
    # autolabel(rects2, ax, "right")
    # autolabel(rects3, ax, "right")

    # # Add some text for labels, title and custom x-axis tick labels, etc.
    # # ax.set_ylabel('Scores')
    # ax.set_title('DICE and DQpoint')
    # ax.set_xticks(ind)
    # ax.set_xticklabels(('DICE', 'DQpoint'))
    # ax.legend()
    # fig.tight_layout()
    # plt.savefig('3datasets.png')

    # return

    lbls1 = []
    ys1 = [[] for _ in ms]
    yss1 = [[] for _ in ms]
    # ys1 = []
    # yss1 = []
    # ys2 = []
    # yss2 = []
    for eps in range(11):
        eps = eps / 10.0
        eps = "{:.1f}".format(eps)
        lbls1.append(eps)
        if eps not in metrics1: continue
        stat = metrics1[eps]
        # ys1.append(stat[ms[0]]['mean'])
        # yss1.append(stat[ms[0]]['std'])
        # ys2.append(stat[ms[1]]['mean'])
        # yss2.append(stat[ms[1]]['std'])
        for i, metric in enumerate(ms):
            ys1[i].append(stat[metric]['mean'])
            yss1[i].append(stat[metric]['std'])
        # print(eps, f"{ms[0]}:", stats_str(stat[ms[0]]), f"{ms[1]}:", stats_str(stat[ms[1]]))
    

    lbls2 = []
    ys2 = [[] for _ in ms]
    yss2 = [[] for _ in ms]
    # ys3 = []
    # yss3 = []
    # ys4 = []
    # yss4 = []
    for eps in range(11):
        eps = eps / 10.0
        eps = "{:.1f}".format(eps)
        lbls2.append(eps)
        if eps not in metrics2: continue
        stat = metrics2[eps]
        # ys3.append(stat[ms[0]]['mean'])
        # yss3.append(stat[ms[0]]['std'])
        # ys4.append(stat[ms[1]]['mean'])
        # yss4.append(stat[ms[1]]['std'])
        for i, metric in enumerate(ms):
            ys2[i].append(stat[metric]['mean'])
            yss2[i].append(stat[metric]['std'])
        # print(eps, f"{ms[0]}:", stats_str(stat[ms[0]]), f"{ms[1]}:", stats_str(stat[ms[1]]))

    fig, axes = plt.subplots(len(ms)//2, 2, figsize=(9, 4), dpi=800) # 10, 6; 9, 4
    plt.setp(axes, xticks=range(11), xticklabels=lbls1)
    xvalues = np.array([i for i in range(len(ys1[0]))])
    # ylims = [(0.68, 0.76), (0.45, 0.75), (0.35, 0.6), (0.3, 0.6)]
    ylims = [(0.7, 0.82), (0.74, 0.86)]

    for i, ax in enumerate(axes.ravel()):
        ax.grid(linestyle='--')
        ax.errorbar(xvalues-0.1, ys1[i], yerr=yss1[i], color='green', ecolor='red', barsabove=True, capsize=2, label='HoVer-Net', fmt='-*')
        ax.errorbar(xvalues+0.1, ys2[i], yerr=yss2[i], ecolor='red', barsabove=True, capsize=2, label='HoVer-Net + Dot Branch', fmt='--.')
        ax.set_ylim(*ylims[i])
        ax.legend()
        ax.set_title(ms[i].replace("_", "") + " vs PseudoRate")

    # plt.legend()
    plt.tight_layout()
    plt.savefig(f"{'-'.join(ms).replace('_', '')}.png")

    return

    fig, (plt1, plt2) = plt.subplots(1, 2, figsize=(5, 4), dpi=1200) # , sharey=True
    plt.setp((plt1, plt2), xticks=range(len(ys1)), xticklabels=lbls1)
    xvalues = np.array([i for i in range(len(ys1))])
    plt1.grid(linestyle='--')
    plt1.errorbar(xvalues-0.1, ys1, yerr=yss1, color='green', ecolor='red', barsabove=True, capsize=2, label='HoVer-Net', fmt='-*')
    plt1.errorbar(xvalues+0.1, ys3, yerr=yss3, ecolor='red', barsabove=True, capsize=2, label='HoVer-Net + Dot Branch', fmt='--.')
    # plt.ylim(0.6, 0.8)
    # plt.xticks(range(len(ys1)), lbls1)
    # plt.xlabel('PseudoRate')
    # plt.ylabel(f'{ms[0]}')
    plt1.set_title(f'{ms[0]} vs PseudoRate')
    plt1.legend()
    # plt.savefig(f'{ms[0]}_dot.png')
    # plt.clf()
    plt2.grid(linestyle='--')
    plt2.errorbar(xvalues-0.1, ys2, yerr=yss2, color='green', ecolor='red', barsabove=True, capsize=2, label='HoVerNet', fmt='-*')
    plt2.errorbar(xvalues+0.1, ys4, yerr=yss4, ecolor='red', barsabove=True, capsize=2, label='HoVerNet + Dot Branch', fmt='--.')
    # plt.ylim(0.3, 0.5)
    # plt.xlabel('PseudoRate')
    # plt.ylabel('DQpoint')
    # plt.xticks(range(len(ys1)), lbls1)
    plt2.set_title(f'{ms[1]} vs PseudoRate')
    plt2.legend()
    plt.tight_layout()
    # plt.savefig('DQp_dot.png')
    plt.savefig(f"{ms[0].replace('_', '')}and{ms[1].replace('_', '')}.png")

if __name__ == "__main__":
    import sys
    ver = sys.argv[1]
    # ms = ['SQ', 'DQ', 'AJI', 'PQ']
    ms = ['DICE', 'DQ_point']
    main(['mixed_pseu.csv', 'mixed_pseudorate_dot.csv'], ver=ver)
    # main(['i_CoNuSeg_CoNSeP.csv', 'mixed_pseu.csv'], ver=ver)
