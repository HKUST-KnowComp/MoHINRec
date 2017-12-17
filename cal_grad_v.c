#include <stdio.h>
#include <stdlib.h>

void cal_grad_v(double * ep, double* xp, double * fp, double* tp, int xn, double *rp) {
    int p = 0;
    int k = 0;

    double res = 0.0;
    for (p = 0; p < xn; p++) {
        res += ep[p] * (xp[p] * fp[p] - tp[p]);
    }
    res = 2.0 * res;
    rp[0] = res;
    return;
}
