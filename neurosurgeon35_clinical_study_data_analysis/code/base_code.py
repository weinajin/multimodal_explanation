import numpy as np
import pandas as pd
import seaborn as sns
import researchpy as rp
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.text import Annotation
from matplotlib.transforms import Affine2D
from sklearn.metrics import precision_score, recall_score, classification_report, RocCurveDisplay, roc_curve, auc, \
    precision_recall_curve, accuracy_score, f1_score, roc_auc_score, recall_score, precision_score, confusion_matrix, \
matthews_corrcoef, auc
from scipy.stats import pearsonr
sns.set_style("whitegrid")

def get_drwise_perf_metric(valid_data, xai25, perf_metric = accuracy_score, pos_label = None, complete_response_only=False):
    gt = xai25['gt']
    condition_dict = {'dr': 'What grade of glioma would you predict', 'AI': 'After viewing AI’s suggestion',
                      'XAI': 'After viewing AI’s explanation'}
    condition_acc = dict()
    id_dfs = []
    num_mri = []

    for k, c in condition_dict.items():
        id_acc = dict()
        id_num = dict()
        answers = valid_data.filter(regex=c)
        accs = []
        for row in answers.itertuples():
            answer = pd.DataFrame(row).values.flatten().tolist()[1:]
            pred_valid = []
            gt_valid = []
            drID = []
            for idx, a in enumerate(answer):
                if not np.isnan(a):
                    pred_valid.append(a)
                    gt_valid.append(gt[idx])
            if complete_response_only:
                if len(pred_valid) != 25:
                    continue
            if pos_label is not None:
                perf = perf_metric(gt_valid, pred_valid, pos_label = pos_label, average = 'binary')
            else:
                perf = perf_metric(gt_valid, pred_valid)
            accs.append(perf)
            id_acc[row.Index] = [perf]
            id_num[row.Index] = len(pred_valid)
        condition_acc[k] = accs
        df = pd.DataFrame.from_dict(id_acc, orient='index', columns=[k])
        numb_df = pd.DataFrame.from_dict(id_num, orient='index', columns=[k])
        id_dfs.append(df)
        num_mri.append(numb_df)
    id_dfs = pd.concat(id_dfs, axis=1)
    num_mri = pd.concat(num_mri, axis = 1)
    return id_dfs, num_mri



def get_metrics(result_long, results, label=1, roc_annotate=0.8, pr_annotate=0.7, prc_ylim=0, arrow='all'):
    # arrow options:
    # 1. all - plot the two arrows, from dr -> dr+AI -> dr+XAI;
    # 2. dr - plot the one arrow, from dr -> dr+AI
    # 3. xai - plot the one arrow, from dr+AI -> dr+XAI
    plt.rcParams.update({'font.size': 20})

    gt = results['gt']
    prob = results['softmax_{}'.format(label)]
    prob_mtx = np.array([results['softmax_0'], results['softmax_1']]).transpose()
    gt_one_hot = np.zeros(prob_mtx.shape)
    for i, g in enumerate(gt):
        gt_one_hot[i, g] = 1

    pred = results['pred']
    accuracy = accuracy_score(gt, pred)

    cm = confusion_matrix(gt, pred)
    precision, recall, thresholds = precision_recall_curve(gt, prob,
                                                           pos_label=label)  # for BRATS_GBM task, Grade II/III =0 is the minority class
    # calculate precision-recall AUC
    prauc = auc(recall, precision)


    auroc = roc_auc_score(gt_one_hot, prob_mtx, average=None)
    f1 = f1_score(gt, pred, average=None)
    rc = recall_score(gt, pred, average=None)
    print("sensitivity: {}, specificity: {}".format(rc[label], rc[1 - label]))
    print(classification_report(gt, pred))
    print(auroc)

    # plot doctor precision-recall dot with arrows
    dr_pr = {drID: [] for drID in
             result_long['Respondent ID'].unique()}  # key - drID, values - [dralone, ai, xai] precision, recall
    dr_rc = {drID: [] for drID in result_long['Respondent ID'].unique()}
    dr_fpr = {drID: [] for drID in result_long['Respondent ID'].unique()}
    dr_tpr = {drID: [] for drID in result_long['Respondent ID'].unique()}
    for drID in result_long['Respondent ID'].unique():
        subdf = result_long[result_long['Respondent ID'] == drID]
        for cndt in ["drAlone", "drAI", "drXAI"]:

            if np.isnan(np.array(subdf[cndt].tolist())).any():  # todo
                mask = np.isnan(np.array(subdf[cndt].tolist()))
                gt_25 = np.ma.array(subdf['gt'].tolist(), mask=mask).compressed()
                pred = np.ma.array(subdf[cndt].tolist(), mask=mask).compressed()
            else:
                gt_25 = subdf['gt'].tolist()
                pred = subdf[cndt].tolist()
            dr_pr[drID].append(precision_score(gt_25, pred, pos_label=label))
            dr_rc[drID].append(recall_score(gt_25, pred, pos_label=label))
            if len(confusion_matrix(gt_25, pred).ravel()) < 4:
                continue
            if label == 1:
                tn, fp, fn, tp = confusion_matrix(gt_25, pred).ravel()
            else:
                tp, fn, fp, tn = confusion_matrix(gt_25, pred).ravel()
            dr_fpr[drID].append(fp / (fp + tn))
            dr_tpr[drID].append(tp / (tp + fn))
        #     print("dr_pr", dr_pr)
    #     print('dr_rc', dr_rc)
    #     print('dr_fpr', dr_fpr)
    #     print('dr_tpr', dr_tpr)

    line_color = ['green', 'orange']
    class_name = ['Grade II/III', 'GBM']
    idcolor = list(sns.color_palette("nipy_spectral", 35))

    id_colors = {drID: idcolor[i] for i, drID in enumerate(result_long['Respondent ID'].unique())}

    # roc curve
    fpr, tpr, threshold = roc_curve(gt, prob, pos_label=label)
    auroc = auc(fpr, tpr)
    print('auroc', auroc)

    figwidth = 8
    mutateSize =25

    ns_probs = [label for _ in range(len(gt))]
    ns_auc = roc_auc_score(gt, ns_probs)
    ns_fpr, ns_tpr, _ = roc_curve(gt, ns_probs, pos_label=label)
    fig = plt.figure(figsize=(figwidth, figwidth))
    ax = fig.add_subplot(111)
    p1, = ax.plot(fpr, tpr, marker='.', color='gray', linewidth = 3, label='AI = {:.2f}'.format(auroc))
    p2, = ax.plot(ns_fpr, ns_tpr, linestyle='--', color='gray', label='Baseline = {:.2f}'.format(ns_auc))
    for i, drID in enumerate(result_long['Respondent ID'].unique()):
        if len(dr_fpr[drID]) >= 3 and len(dr_tpr[drID]) >= 3:
            print(drID, dr_fpr[drID], dr_tpr[drID])
            if arrow == 'all':
                arrowplot(ax, dr_fpr[drID], dr_tpr[drID], nArrs=1, mutateSize=mutateSize, color=id_colors[drID])
                plt.plot(dr_fpr[drID], dr_tpr[drID], '+', color=id_colors[drID])
            elif arrow == 'dr':
                arrowplot(ax, dr_fpr[drID][:2], dr_tpr[drID][:2], nArrs=1, mutateSize=mutateSize, color=id_colors[drID])
                plt.plot(dr_fpr[drID][:2], dr_tpr[drID][:2], '+', color=id_colors[drID])
            elif arrow == 'xai':
                arrowplot(ax, dr_fpr[drID][1:], dr_tpr[drID][1:], nArrs=1, mutateSize=mutateSize, color=id_colors[drID])
                plt.plot(dr_fpr[drID][1:], dr_tpr[drID][1:], '+', color=id_colors[drID])                

    # axis labels
    plt.xlabel('1- Specificity')
    plt.ylabel('Sensitivity')
    # show the legend
    line_annotate('AI AUC = {:.2f}'.format(auroc), p1, roc_annotate)
    line_annotate('Baseline AUC = {:.2f}'.format(ns_auc), p2, 0.5)
    plt.title("{}: ROC".format(class_name[label]))
    # show the plot
#     fig.savefig('../reporting/roc_{}.svg'.format(label), bbox_inches='tight')

    plt.show()

    # pr curve
    ns_probs = [label for _ in range(len(gt))]
    bl_precision, bl_recall, thresholds = precision_recall_curve(gt, ns_probs,
                                                                 pos_label=label)  # for BRATS_GBM task, Grade II/III =0 is the minority class
    # calculate precision-recall AUC
    bl_prauc = auc(bl_recall, bl_precision)

    # plot
    fig = plt.figure(figsize=(figwidth, figwidth))
    ax1 = fig.add_subplot(111)
    p1, = plt.plot(recall, precision, color='gray', linewidth = 3, label='AI = {:.2f}'.format(prauc))
    p2, = plt.plot(bl_recall, bl_precision, linestyle='--', color='gray', label='Baseline = {:.2f}'.format(bl_prauc))
    for i, drID in enumerate(result_long['Respondent ID'].unique()):
        if len(dr_rc[drID]) >= 3 and len(dr_pr[drID]) >= 3 and sum(dr_rc[drID]) + sum(dr_pr[drID]) >0:
            print('pr-rc',drID, dr_rc[drID], dr_pr[drID])
            if arrow == 'all':
                arrowplot(ax1, dr_rc[drID], dr_pr[drID], nArrs=1, mutateSize=mutateSize, color=id_colors[drID])
                plt.plot(dr_rc[drID], dr_pr[drID], '+', color=id_colors[drID])
            elif arrow == 'dr':
                arrowplot(ax1, dr_rc[drID][:2], dr_pr[drID][:2], nArrs=1, mutateSize=mutateSize, color=id_colors[drID])
                plt.plot(dr_rc[drID][:2], dr_pr[drID][:2], '+', color=id_colors[drID])
            elif arrow == 'xai':
                arrowplot(ax1, dr_rc[drID][1:], dr_pr[drID][1:], nArrs=1, mutateSize=mutateSize, color=id_colors[drID])
                plt.plot(dr_rc[drID][1:], dr_pr[drID][1:], '+', color=id_colors[drID])

    axes = plt.gca()
    plt.xlabel('Recall (Sensitivity)')
    plt.ylabel('Precision')
    line_annotate('AI AUC = {:.2f}'.format(prauc), p1, pr_annotate)
    line_annotate('Baseline AUC = {:.2f}'.format(bl_prauc), p2, 0.5)

    plt.title("{}: Precision-Recall Curve".format(class_name[label]))

    # show the plot
    plt.show()
#     fig.savefig('../reporting/precision_recall_{}.svg'.format(label), bbox_inches='tight')

#     print(f1, cm, prauc)
    return fig, ax, ax1


### ROC change curve vis


def arrowplot(axes, x, y, nArrs=30, mutateSize=10, color='gray', markerStyle='o'):
    '''arrowplot : plots arrows along a path on a set of axes
        axes   :  the axes the path will be plotted on
        x      :  list of x coordinates of points defining path
        y      :  list of y coordinates of points defining path
        nArrs  :  Number of arrows that will be drawn along the path
        mutateSize :  Size parameter for arrows
        color  :  color of the edge and face of the arrow head
        markerStyle : Symbol

        Bugs: If a path is straight vertical, the matplotlab FanceArrowPatch bombs out.
          My kludge is to test for a vertical path, and perturb the second x value
          by 0.1 pixel. The original x & y arrays are not changed

        MHuster 2016, based on code by
    '''
    # recast the data into numpy arrays
    x = np.array(x, dtype='f')
    y = np.array(y, dtype='f')
    nPts = len(x)

    # Plot the points first to set up the display coordinates
    axes.plot(x, y, markerStyle, ms=5, color=color)

    # get inverse coord transform
    inv = axes.transData.inverted()

    # transform x & y into display coordinates
    # Variable with a 'D' at the end are in display coordinates
    xyDisp = np.array(axes.transData.transform(list(zip(x, y))))
    #     print(xyDisp)
    xD = xyDisp[:, 0]
    yD = xyDisp[:, 1]

    # drD is the distance spanned between pairs of points
    # in display coordinates
    dxD = xD[1:] - xD[:-1]
    dyD = yD[1:] - yD[:-1]
    drD = np.sqrt(dxD ** 2 + dyD ** 2)
    if not np.any(drD):
        #         print(drD, 'dist')
        return

    # Compensating for matplotlib bug
    dxD[np.where(dxD == 0.0)] = 0.1

    # rtotS is the total path length
    rtotD = np.sum(drD)

    # based on nArrs, set the nominal arrow spacing
    arrSpaceD = rtotD / nArrs

    # Loop over the path segments
    iSeg = 0
    while iSeg < nPts - 1:
        # Figure out how many arrows in this segment.
        # Plot at least one.
        nArrSeg = max(1, int(drD[iSeg] / arrSpaceD + 0.5))
        xArr = (dxD[iSeg]) / nArrSeg  # x size of each arrow
        segSlope = dyD[iSeg] / dxD[iSeg]
        # Get display coordinates of first arrow in segment
        xBeg = xD[iSeg]
        xEnd = xBeg + xArr
        yBeg = yD[iSeg]
        yEnd = yBeg + segSlope * xArr
        # Now loop over the arrows in this segment
        for iArr in range(nArrSeg):
            # Transform the oints back to data coordinates
            xyData = inv.transform(((xBeg, yBeg), (xEnd, yEnd)))
            # Use a patch to draw the arrow
            # I draw the arrows with an alpha of 0.5
            p = patches.FancyArrowPatch(
                xyData[0], xyData[1],
                arrowstyle=patches.ArrowStyle("Fancy", head_length=.6, head_width=.5, tail_width=0.1),
                mutation_scale=mutateSize,
                color=color, alpha=0.5)
            axes.add_patch(p)
            # Increment to the next arrow
            xBeg = xEnd
            xEnd += xArr
            yBeg = yEnd
            yEnd += segSlope * xArr
        # Increment segment number
        iSeg += 1

class LineAnnotation(Annotation):
    """A sloped annotation to *line* at position *x* with *text*
    Optionally an arrow pointing from the text to the graph at *x* can be drawn.
    Usage
    -----
    fig, ax = subplots()
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    ax.add_artist(LineAnnotation("text", line, 1.5))
    """

    def __init__(
            self, text, line, x, xytext=(0, 5), textcoords="offset points", **kwargs
    ):
        """Annotate the point at *x* of the graph *line* with text *text*.

        By default, the text is displayed with the same rotation as the slope of the
        graph at a relative position *xytext* above it (perpendicularly above).

        An arrow pointing from the text to the annotated point *xy* can
        be added by defining *arrowprops*.

        Parameters
        ----------
        text : str
            The text of the annotation.
        line : Line2D
            Matplotlib line object to annotate
        x : float
            The point *x* to annotate. y is calculated from the points on the line.
        xytext : (float, float), default: (0, 5)
            The position *(x, y)* relative to the point *x* on the *line* to place the
            text at. The coordinate system is determined by *textcoords*.
        **kwargs
            Additional keyword arguments are passed on to `Annotation`.

        See also
        --------
        `Annotation`
        `line_annotate`
        """
        assert textcoords.startswith(
            "offset "
        ), "*textcoords* must be 'offset points' or 'offset pixels'"

        self.line = line
        self.xytext = xytext

        # Determine points of line immediately to the left and right of x
        xs, ys = line.get_data()

        def neighbours(x, xs, ys, try_invert=True):
            inds, = np.where((xs <= x)[:-1] & (xs > x)[1:])
            if len(inds) == 0:
                assert try_invert, "line must cross x"
                return neighbours(x, xs[::-1], ys[::-1], try_invert=False)

            i = inds[0]
            return np.asarray([(xs[i], ys[i]), (xs[i + 1], ys[i + 1])])

        self.neighbours = n1, n2 = neighbours(x, xs, ys)

        # Calculate y by interpolating neighbouring points
        y = n1[1] + ((x - n1[0]) * (n2[1] - n1[1]) / (n2[0] - n1[0]))

        kwargs = {
            "horizontalalignment": "center",
            "rotation_mode": "anchor",
            **kwargs,
        }
        super().__init__(text, (x, y), xytext=xytext, textcoords=textcoords, **kwargs)

    def get_rotation(self):
        """Determines angle of the slope of the neighbours in display coordinate system
        """
        transData = self.line.get_transform()
        dx, dy = np.diff(transData.transform(self.neighbours), axis=0).squeeze()
        return np.rad2deg(np.arctan2(dy, dx))

    def update_positions(self, renderer):
        """Updates relative position of annotation text
        Note
        ----
        Called during annotation `draw` call
        """
        xytext = Affine2D().rotate_deg(self.get_rotation()).transform(self.xytext)
        self.set_position(xytext)
        super().update_positions(renderer)


def line_annotate(text, line, x, *args, **kwargs):
    """Add a sloped annotation to *line* at position *x* with *text*

    Optionally an arrow pointing from the text to the graph at *x* can be drawn.

    Usage
    -----
    x = linspace(0, 2*pi)
    line, = ax.plot(x, sin(x))
    line_annotate("sin(x)", line, 1.5)

    See also
    --------
    `LineAnnotation`
    `plt.annotate`
    """
    ax = line.axes
    a = LineAnnotation(text, line, x, *args, **kwargs)
    if "clip_on" in kwargs:
        a.set_clip_path(ax.patch)
    ax.add_artist(a)
    return a