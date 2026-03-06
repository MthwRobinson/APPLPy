"""KSRV distribution."""


from sympy import (
    Symbol,
    factorial,
    floor,
    integrate,
    simplify,
    symbols,
)

from ...rv import RV, RVError

x, y, z, t, v = symbols("x y z t v")


class KSRV(RV):
    """
    Procedure Name: KSRV
    Purpose: Creates an instance of the kolmogoroff-smirnov distribution
    Arguments:  1. n: a positive integer parameter
    Output:     1. A kolmogoroff-smirnov random variable
    """

    def __init__(self, n=Symbol("n", positive=True, integer=True)):
        if not isinstance(n, Symbol):
            if n <= 0:
                if not isinstance(n, int):
                    err_string = "n must be a positive integer"
                    raise RVError(err_string)
        # Phase 1
        N = n
        m = floor(3 * N / 2) + (N % 2) - 1
        vv = list(range(m + 1))
        vvalue = []
        for i in range(len(vv)):
            vvalue.append(0)
        vv = dict(list(zip(vv, vvalue)))
        vv[0] = 0
        g = 1 / (2 * N)
        mm = 0
        for i in range(1, N):
            mm += 1
            vv[mm] = i * g
        for j in range(2 * floor(N / 2) + 1, 2 * N, 2):
            mm += 1
            vv[mm] = j * g
        # Phase 2
        # Generate the c array
        cidx = []
        cval = []
        for k in range(1, m + 1):
            cidx.append(k)
            cval.append((vv[k - 1] + vv[k]) / 2)
        c = dict(list(zip(cidx, cval)))
        # Generate the x array
        xidx = []
        xval = []
        for k in range(1, N + 1):
            xidx.append(k)
            xval.append((2 * k - 1) * g)
        x = dict(list(zip(xidx, xval)))
        # Generate an NxN A array
        aidx = list(range(1, N + 1))
        aval = []
        for i in aidx:
            aval.append(0)
        arow = dict(list(zip(aidx, aval)))
        A = dict(list(zip(aidx, aval)))
        for i in aidx:
            A[i] = arow
        # Insert values into the A array
        for i in range(2, N + 1):
            for j in range(1, i):
                A[i][j] = 0
        for k in range(1, m + 1):
            for i in range(1, N + 1):
                for j in range(i, N + 1):
                    A[i][j] = 0
            z = max(floor(N * c[k] - 1 / 2), 0)
            seg_len = min(floor(2 * N * c[k]) + 1, N)
            for i in range(1, N + 1):
                for j in range(max(i, z + 1), min(N, i + seg_len - 1) + 1):
                    A[i][j] = 1
        # Create a 1xm P array
        Pidx = []
        Pval = []
        for i in range(1, m + 1):
            Pidx.append(i)
            Pval.append(0)
        P = dict(list(zip(Pidx, Pval)))
        # Create an NxN F array
        fidx = list(range(1, N + 1))
        fval = []
        for i in fidx:
            fval.append(0)
        frow = dict(list(zip(fidx, fval)))
        F = dict(list(zip(fidx, fval)))
        for i in fidx:
            F[i] = frow
        # Create an NxN V array
        vidx = list(range(1, N + 1))
        vval = []
        for i in vidx:
            vval.append(0)
        vrow = dict(list(zip(vidx, vval)))
        V = dict(list(zip(vidx, vval)))
        for i in vidx:
            V[i] = vrow
        # Create a list of indexed u variables
        varstring = "u:" + str(N + 1)
        u = symbols(varstring)
        for k in range(2, m + 1):
            z = int(max(floor(N * c[k] - 1 / 2), 0))
            seg_len = int(min(floor(2 * N * c[k]) + 1, N))
            F[N][N] = integrate(1, (u[N], x[N] - v, 1))
            V[N][N] = integrate(1, (u[N], u[N - 1], 1))
            for i in range(N - 1, 1, -1):
                if i + seg_len > N:
                    S = 0
                else:
                    S = F[i + 1][i + seg_len]
                if i + seg_len > N + 1:
                    F[i][N] = integrate(V[i + 1][N], (u[i], x[N] - v, floor(x[i] + c[k])))
                    V[i][N] = integrate(V[i + 1][N], (u[i], u[i - 1], floor(x[i] + c[k])))
                if i + seg_len == N + 1:
                    F[i][N] = integrate(V[i + 1][N], (u[i], x[N] - v, x[i] + v))
                if i + seg_len < N + 1:
                    F[i][i + seg_len - 1] = integrate(
                        V[i + 1][i + seg_len - 1] + S, (u[i], x[N] - v, x[i] + v)
                    )
                S += F[i + 1][min(i + seg_len - 1, N)]
                for j in range(min(N - 1, i + seg_len - 2), max(i + 1 - 1, z + 2 - 1), -1):
                    F[i][j] = integrate(V[i + 1][j] + S, (u[i], x[j] - v, x[j + 1] - v))
                    V[i][j] = integrate(V[i + 1][j] + S, (u[i], u[i - 1], x[j + 1] - v))
                    S += F[i + 1][j]
                if z + 1 <= i:
                    V[i][i] = integrate(S, (u[i], u[i - 1], x[i + 1] - v))
                if z + 1 > i:
                    V[i][z + 1] = integrate(V[i + 1][z + 1] + S, (u[i], u[i - 1], x[z + 2] - v))
                if z + 1 < i:
                    F[i][i] = integrate(S, (u[i], x[i] - v, x[i + 1] - v))
            if seg_len == N:
                S = 0
                F[1][N] = integrate(V[2][N], (u[1], x[N] - v, x[1] + v))
            else:
                S = F[2][seg_len + 1]
            if seg_len < N:
                F[1][seg_len] = integrate(V[2][seg_len] + S, (u[1], x[seg_len] - v, x[1] + v))
            S += F[2][j]
            for j in range(min(N - 1, i + seg_len - 2), max(i, z + 1), -1):
                F[1][j] = integrate(
                    V[2][j] + S, (u[1], (x[j] - v) * (floor(x[j] - c[k]) + 1), x[j + 1] - v)
                )
                S += F[2][j]
            if z == 0:
                F[1][1] = integrate(S, (u[1], 0, x[2] - v))
            P[k] = 0
            for j in range(z + 1, seg_len + 1):
                P[k] += F[1][j]
            P[k] = factorial(N) * P[k]
        # Create the support and function list for the KSRV
        KSspt = []
        KSCDF = []
        x = Symbol("x")
        for i in range(0, m + 1):
            KSspt.append(vv[i] + 1 / (2 * N))
        for i in range(1, m + 1):
            func = P[i]
            if isinstance(func, (int, float)):
                ksfunc = func
            else:
                ksfunc = func.subs(v, (x - 1 / (2 * N)))
            KSCDF.append(simplify(ksfunc))
        # Remove redundant elements from the list
        KSCDF2 = []
        KSspt2 = []
        KSspt2.append(KSspt[0])
        KSCDF2.append(KSCDF[0])
        for i in range(1, len(KSCDF)):
            if KSCDF[i] != KSCDF[i - 1]:
                KSCDF2.append(KSCDF[i])
                KSspt2.append(KSspt[i])
        KSspt2.append(KSspt[-1])
        X_dummy = RV(KSCDF, KSspt, ["continuous", "cdf"])
        self.func = X_dummy.func
        self.support = X_dummy.support
        self.ftype = X_dummy.ftype
        self.cache = {}


__all__ = ["KSRV"]
