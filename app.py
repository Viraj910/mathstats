from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)

def pearson_correlation(x, y):
    n = len(x)
    sx = sum(x)
    sy = sum(y)
    sxy = sum(xi * yi for xi, yi in zip(x, y))
    sxs = sum(xi ** 2 for xi in x)
    sys = sum(yi ** 2 for yi in y)

    numerator = (n * sxy) - (sx * sy)
    denominator = (((n * sxs) - (sx ** 2)) * ((n * sys) - (sy ** 2))) ** 0.5

    if denominator == 0:
        return None
    else:
        return numerator / denominator

def spearman_rank_correlation(x, y):
    n = len(x)
    rank_x = {val: rank for rank, val in enumerate(sorted(set(x)), 1)}
    rank_y = {val: rank for rank, val in enumerate(sorted(set(y)), 1)}
    d = [(rank_x[xi] - rank_y[yi]) for xi, yi in zip(x, y)]
    d2 = [di ** 2 for di in d]
    totalD = sum(d)
    totalD2 = sum(d2)
    r = 1 - (6 * totalD2) / (n * (n ** 2 - 1))
    return r, d, d2, totalD, totalD2

def avgrank(x):
    n = len(x)
    sorted_x = sorted(x)
    rx = [sorted_x.index(xi) + 1 for xi in x]
    seen = set()
    copyrx = rx.copy()
    for i in range(n):
        if rx[i] in seen:
            continue
        seen.add(rx[i])
        count = rx.count(rx[i])
        if count > 1:
            avg = sum(range(rx[i], rx[i] + count)) / count
            for j in range(n):
                if rx[j] == rx[i]:
                    copyrx[j] = avg
    return copyrx

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/pearson', methods=['GET', 'POST'])
def pearson():
    if request.method == 'POST':
        try:
            x = list(map(float, request.form.get('x').split(',')))
            y = list(map(float, request.form.get('y').split(',')))

            if len(x) != len(y):
                raise ValueError("The lengths of X and Y must be equal.")

            # Calculate Pearson correlation coefficient
            pearson_result = pearson_correlation(x, y)

            # Prepare data for the table
            xy = [xi * yi for xi, yi in zip(x, y)]
            xs = [xi ** 2 for xi in x]
            ys = [yi ** 2 for yi in y]

            # Zip the data together
            data = list(zip(x, y, xy, xs, ys))

            # Calculate sums for each column
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(xy)
            sum_xs = sum(xs)
            sum_ys = sum(ys)

            return render_template('pearson.html', pearson=pearson_result, data=data,
                                   sum_x=sum_x, sum_y=sum_y, sum_xy=sum_xy, sum_xs=sum_xs, sum_ys=sum_ys)

        except ValueError as e:
            return render_template('pearson.html', error=str(e), pearson=None, data=None,
                                   sum_x=None, sum_y=None, sum_xy=None, sum_xs=None, sum_ys=None)

    return render_template('pearson.html', pearson=None, data=None,
                           sum_x=None, sum_y=None, sum_xy=None, sum_xs=None, sum_ys=None)

@app.route('/spearman', methods=['GET', 'POST'])
def spearman():
    if request.method == 'POST':
        try:
            x = list(map(float, request.form.get('xValues').split(',')))
            y = list(map(float, request.form.get('yValues').split(',')))
            choice = int(request.form.get('choice'))

            if len(x) != len(y):
                raise ValueError("The lengths of X and Y must be equal.")

            if choice == 1:
                arx = avgrank(x)
                ary = avgrank(y)
                d = [arx[i] - ary[i] for i in range(len(x))]
                d2 = [di ** 2 for di in d]
                totalD2 = sum(d2)
                r = 1 - (6 * totalD2) / (len(x) * (len(x) ** 2 - 1))
                data = list(zip(x, y, d, d2))
                return render_template('spearman.html', result=r, data=data, totalD=sum(d), totalD2=totalD2)

            elif choice == 2:
                avgrx = avgrank(x)
                avgry = avgrank(y)
                d = [avgrx[i] - avgry[i] for i in range(len(x))]
                d2 = [di ** 2 for di in d]
                totalD2 = sum(d2)
                r = 1 - (6 * totalD2) / (len(x) * (len(x) ** 2 - 1))
                data = list(zip(x, y, avgrx, avgry, d, d2))
                return render_template('spearman.html', result=r, data=data, totalD=sum(d), totalD2=totalD2)

        except ValueError as e:
            return render_template('spearman.html', error=str(e), result=None, data=None, totalD=None, totalD2=None)

    return render_template('spearman.html', result=None, data=None, totalD=None, totalD2=None)

@app.route('/linearreg', methods=['GET', 'POST'])
def linearreg():
    if request.method == 'POST':
        try:
            x = list(map(float, request.form.get('xValues').split(',')))
            y = list(map(float, request.form.get('yValues').split(',')))

            if len(x) != len(y):
                raise ValueError("The lengths of X and Y must be equal.")

            a = x[0]
            b = y[0]
            dx = [xi - a for xi in x]
            dy = [yi - b for yi in y]
            dx2 = [dxi ** 2 for dxi in dx]
            dy2 = [dyi ** 2 for dyi in dy]
            dxdy = [dxi * dyi for dxi, dyi in zip(dx, dy)]
            sdx = sum(dx)
            sdy = sum(dy)
            sdx2 = sum(dx2)
            sdy2 = sum(dy2)
            sdxdy = sum(dxdy)
            byx = (sdxdy - (sdx * sdy / len(x))) / (sdx2 - (sdx ** 2 / len(x)))
            bxy = (sdxdy - (sdx * sdy / len(x))) / (sdy2 - (sdy ** 2 / len(x)))
            r = (byx * bxy) ** 0.5

            data = list(zip(x, y, dx, dy, dx2, dy2, dxdy))

            return render_template('linearreg.html', byx=byx, bxy=bxy, r=r, data=data,
                                   sdx=sdx, sdy=sdy, sdx2=sdx2, sdy2=sdy2, sdxdy=sdxdy)

        except ValueError as e:
            return render_template('linearreg.html', error=str(e), byx=None, bxy=None, r=None, data=None,
                                   sdx=None, sdy=None, sdx2=None, sdy2=None, sdxdy=None)

    return render_template('linearreg.html', byx=None, bxy=None, r=None, data=None,
                           sdx=None, sdy=None, sdx2=None, sdy2=None, sdxdy=None)

@app.route('/curvefitting', methods=['GET', 'POST'])
def curvefitting():
    if request.method == 'POST':
        try:
            x = list(map(float, request.form.get('xValues').split(',')))
            y = list(map(float, request.form.get('yValues').split(',')))
            choice = int(request.form.get('choice'))

            if len(x) != len(y):
                raise ValueError("The lengths of X and Y must be equal.")

            if choice == 1:
                x2 = [xi ** 2 for xi in x]
                xy = [xi * yi for xi, yi in zip(x, y)]
                sx2 = sum(x2)
                sxy = sum(xy)
                sx = sum(x)
                sy = sum(y)
                data = list(zip(x, y, x2, xy))

                return render_template('curvefitting.html', choice=choice, data=data,
                                       sx=sx, sy=sy, sx2=sx2, sxy=sxy)

            elif choice == 2:
                mid = len(x) // 2
                a = x[mid]
                b = y[mid]
                X = [xi - a for xi in x]
                Y = [yi - b for yi in y]
                Xs = [Xi ** 2 for Xi in X]
                Xc = [Xi ** 3 for Xi in X]
                Xf = [Xi ** 4 for Xi in X]
                XY = [Xi * Yi for Xi, Yi in zip(X, Y)]
                Xsy = [(Xi ** 2) * Yi for Xi, Yi in zip(X, Y)]
                data = list(zip(x, y, X, Y, Xs, Xc, Xf, XY, Xsy))

                return render_template('curvefitting.html', choice=choice, data=data,
                                       sum_X=sum(X), sum_Y=sum(Y), sum_Xs=sum(Xs), sum_Xc=sum(Xc),
                                       sum_Xf=sum(Xf), sum_XY=sum(XY), sum_Xsy=sum(Xsy))

            elif choice == 3:
                Y = [np.log(yi) for yi in y]
                x2 = [xi ** 2 for xi in x]
                xY = [xi * Yi for xi, Yi in zip(x, Y)]
                data = list(zip(x, y, Y, x2, xY))

                return render_template('curvefitting.html', choice=choice, data=data,
                                       sum_x=sum(x), sum_y=sum(y), sum_Y=sum(Y), sum_x2=sum(x2), sum_xY=sum(xY))

        except ValueError as e:
            return render_template('curvefitting.html', error=str(e), choice=None, data=None,
                                   sum_x=None, sum_y=None, sum_Y=None, sum_x2=None, sum_xY=None)

    return render_template('curvefitting.html', choice=None, data=None,
                           sum_x=None, sum_y=None, sum_Y=None, sum_x2=None, sum_xY=None)

if __name__ == '__main__':
    app.run(debug=True)