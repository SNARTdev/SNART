
import logging
import copy
import numpy as np
import lmfit
import uncertainties
from uncertainties import ufloat


class SNFit(object):

    def __init__(self, sed_data, sed_pre_list, ribbons, impostors=[],
                 a1_guess=-1, a2_guess=2.5, s_guess=-0.01, a1_vary=True,
                 a2_vary=True, s_vary=True, asym_lim=False,
                 two_slope=False, log=False):

        poly_a1, poly_a2, sed_list, SEDs_w_1_pt = self.check_onesided(
            ribbons, sed_pre_list, impostors)

        self.sed_data = sed_data
        self.ribbons = ribbons
        self.a1_guess = a1_guess
        self.a2_guess = a2_guess
        self.s_guess = s_guess
        self.a1_vary = a1_vary
        self.a2_vary = a2_vary
        self.s_vary = s_vary
        self.poly_a1 = poly_a1
        self.poly_a2 = poly_a2
        self.sed_list = sed_list
        self.SEDs_w_1_pt = SEDs_w_1_pt
        self.impostors = impostors
        self.asym_lim = asym_lim
        self.two_slope = two_slope
        self.log = log

        logging.info("Running fit")
        self.result = self.run_fit()

    def check_onesided(self, ribbons, sed_pre_list, impostors=[]):
        #these are the SED numbers that need their own polyfit
        poly_a2, poly_a1 = [], []
        SEDs_w_1_pt = []
        for sed in [s for s in sed_pre_list if s not in impostors]:
            flux = ribbons[sed].chi_flux
            unc = ribbons[sed].chi_unc
            a = len(flux)
            if a == 1:
                SEDs_w_1_pt.append(sed)
            else:
                b = (np.diff(flux + unc) > 0).sum()
                if b == a - 1:
                    poly_a2.append(sed)
                elif b == 0:
                    poly_a1.append(sed)
        sed_list = [s for s in sed_pre_list if s not in SEDs_w_1_pt]

        logging.debug("These SEDs have just one data point: %s", SEDs_w_1_pt)
        logging.debug("The original list was: %s", sed_pre_list)
        logging.debug("Now it is: %s", sed_list)

        return poly_a1, poly_a2, sed_list, SEDs_w_1_pt

    def run_fit(self):

        params = lmfit.Parameters()
        params.add('alpha1', self.a1_guess, vary=self.a1_vary, max=-0.01)
        params.add('alpha2', self.a2_guess, vary=self.a2_vary, min=0.01)
        params.add('s', self.s_guess, min=-6,max=-0.01, vary=self.s_vary)

        flux = {i: self.ribbons[i].chi_flux_log if self.log else
                   self.ribbons[i].chi_flux for i in self.sed_list}

        onesided = self.poly_a1 + self.poly_a2
        twosided = list(set(self.sed_list) - set(onesided))
        for i in onesided:
            b_guess = flux[i].max() if i in self.poly_a1 else flux[i].min()
            params.add('b{}'.format(i), b_guess)
        for i in twosided:
            freqlog = self.ribbons[i].chi_freq_log
            bfrlog = freqlog[np.argmax(flux[i])]
            bfr_guess = bfrlog if self.log else np.power(10, bfrlog)
            params.add('brk_flux{}'.format(i), flux[i].max())
            params.add('brk_freq{}'.format(i), bfr_guess)

        mini = lmfit.Minimizer(
            joint_residual, params=params,
            fcn_kws=dict(asym_lim=self.asym_lim, log=self.log),
            fcn_args=(self.poly_a1, self.poly_a2, self.sed_list, self.ribbons),
            nan_policy='propagate', scale_covar=True, calc_covar=True)
        result = mini.minimize(method='leastsq')

        return SNFitResult(result, poly_a1=self.poly_a1, log=self.log)

    @classmethod
    def from_config(cls, cp, data, display=False):

        df = data.df if display else data.df_obs
        ribbons = data.ribbons if display else data.ribbons_obs

        args = dict(a1_guess=-1, a2_guess=2.5, s_guess=-0.01,
                    a1_vary=True, a2_vary=True, s_vary=True)

        # get param guesses
        if cp.has_option("fit", "alpha1-guess"):
            args["a1_guess"] = float(cp.get("fit", "alpha1-guess"))
        if cp.has_option("fit", "alpha2-guess"):
            args["a2_guess"] = float(cp.get("fit", "alpha2-guess"))
        if cp.has_option("fit", "s-guess"):
            args["s_guess"] = float(cp.get("fit", "s-guess"))

        # check for params held constant
        if cp.has_option("fit", "fixed-alpha1"):
            args["a1_vary"] = False
            args["a1_guess"] = float(cp.get("fit", "fixed-alpha1"))
        if cp.has_option("fit", "fixed-alpha2"):
            args["a2_vary"] = False
            args["a2_guess"] = float(cp.get("fit", "fixed-alpha2"))
        if cp.has_option("fit", "fixed-s"):
            args["s_vary"] = False
            args["s_guess"] = float(cp.get("fit", "fixed-s"))

        if cp.has_option("fit", "impostors"):
            impostors = [int(i) for i in cp.get("fit", "impostors").split(',')]
            args["impostors"] = impostors

        if cp.has_option("fit", "use-asymptotic-limit"):
            args["asym_lim"] = True

        if cp.has_option("fit", "two-slope"):
            args["two_slope"] = True

        if cp.has_option("fit", "log-scale"):
            args["log"] = True

        logging.debug("Fit args:")
        for k, v in args.items():
            logging.debug("%s: %s", k, v)

        return cls(df, data.sed_pre_list, ribbons, **args)


class SNFitResult(object):
    def __init__(self, lmfitresult, poly_a1=None, log=False):
        self.phys_data = None
        self.phys_fits = None
        self.phys_data_equi = None
        self.poly_a1 = poly_a1
        self.log = log
        self.obs = lmfitresult
        self.var_names = copy.deepcopy(lmfitresult.var_names)
        self.linked_variables = {}

        self.covar_scaled = None
        logging.info("Scaling covariance matrix for p")
        self._scale_covar_for_p()

        self.disp = None
        logging.info("Scaling fit to display units")
        self._scale_to_display()
        if not self.log:
            self.covar_disp = False
            logging.info("Scaling covariance matrix to display units")
            self._scale_covar_to_display()

    def _scale_covar_for_p(self):
        if self.covar_scaled is not None:
            logging.warn("Covariance matrix has already been scaled for p!")
            return
        # will scale a copy of covar; original remains the same
        self.covar_scaled = copy.deepcopy(self.obs.covar)
        # if alpha1 or alpha2 do not exist, add them to the list of varnames
        # and put zeroes in the scaled covar matrix
        if 'alpha1' not in self.var_names:
            self.var_names.insert(0,'alpha1')
            add_col = np.zeros(len(self.var_names) - 1)
            add_row = np.zeros(len(self.var_names))
            hold_matrix = np.column_stack((add_col, self.covar_scaled))
            self.covar_scaled = np.vstack((add_row, hold_matrix))
        if 'alpha2' not in self.var_names:
            self.var_names.insert(1, 'alpha2')
            add_col = np.zeros(len(self.var_names) - 1)
            add_row = np.zeros(len(self.var_names))
            first_col = self.covar_scaled[:, 0]
            rest_cols = self.covar_scaled[:, 1:]
            hold_matrix = np.column_stack((add_col, rest_cols))
            self.covar_scaled = np.column_stack((first_col, hold_matrix))

            first_row = self.covar_scaled[0, :]
            rest_rows = self.covar_scaled[1:, :]
            hold_matrix = np.vstack((add_row, rest_rows))
            self.covar_scaled = np.vstack((first_row, hold_matrix))

        # derivative of alpha1 with respect to p
        # NOTE: this will probably change during free-free absorption!
        da1dp = 1. / 2
        alpha1_loc = list(self.var_names).index('alpha1')
        self.covar_scaled[:, alpha1_loc] *= (1. / da1dp)
        self.covar_scaled[alpha1_loc, :] *= (1. / da1dp)

        nomvals = [self.obs.params[var].value for var in self.var_names]
        nomvals[alpha1_loc] = self.solve_p(self.obs.params['alpha1'].value)
        self.var_names[alpha1_loc] = "p"

        corr_vars = uncertainties.correlated_values(nomvals, self.covar_scaled)
        self.linked_variables = dict(zip(self.var_names, corr_vars))
        return

    def _scale_covar_to_display(self):
        if self.covar_disp is True:
            logging.warn("Covariance has already been scaled to display "
                         "units!")
            return
        for i, var in enumerate(self.disp.var_names):
            s = self._get_scale(var)
            self.disp.covar[:, i] *= s
            self.disp.covar[i, :] *= s
        self.covar_disp = True
        return

    def _scale_to_display(self):
        if self.disp is not None:
            logging.warn("Fit result has already been scaled!")
            return
        self.disp = copy.deepcopy(self.obs)
        for var in self.disp.var_names:
            scale = self._get_scale(var)
            if self.log:
                self.disp.params[var].value += np.log10(scale)
            else:
                self.disp.params[var].value *= scale
                self.disp.params[var].stderr *= scale
        return

    def _get_scale(self, var):
        """Calculate the linear-space scaling factor to go from observational
        to display units (Jy and 5GHz to uJy and GHz).
        """
        if 'brk_freq' in var:
            scale = 5.
        elif 'brk_flux' in var:
            scale = 1e6
        elif var.startswith('b'):
            a1 = self.obs.params['alpha1'].value
            a2 = self.obs.params['alpha2'].value
            sed = int(var[1:])
            m = a1 if sed in self.poly_a1 else a2
            scale = 1e6 * 5. ** -m
        else:
            scale = 1.
        return scale

    @staticmethod
    def solve_p(alpha1):
        return -2. * alpha1 + 1.


def joint_residual(pars, poly_a1, poly_a2, sed_list, ribbons, asym_lim=True,
                  log=False):
    fit_params = pars.valuesdict()
    a_1 = fit_params['alpha1']
    a_2 = fit_params['alpha2']
    s_s = fit_params['s']

    l_freq = {sed: ribbons[sed].chi_freq_log for sed in sed_list}
    l_flux = np.array([ribbons[sed].chi_flux_log
                       for sed in sed_list], dtype=object)
    l_unc = np.array([ribbons[sed].chi_unc_log
                      for sed in sed_list], dtype=object)

    l_model_flux = []
    for sed in sed_list:
        if sed in poly_a1 or sed in poly_a2:
            # one-sided SED, model as line
            slope = a_1 if sed in poly_a1 else a_2
            fval = line(l_freq[sed], slope, fit_params['b{}'.format(sed)],
                        log=log)
        else:
            # full SED, model as broken powerlaw
            b_fl = fit_params['brk_flux{}'.format(sed)]
            b_fr = fit_params['brk_freq{}'.format(sed)]
            if asym_lim:
                fval = asymp_intercept(l_freq[sed], b_fl, b_fr, a_1, a_2, s_s,
                                       log=log)
            else:
                freqratio = peak_freq(a_1, a_2, s_s)
                fluxratio = peak_flux(a_1, a_2, s_s)
                fval = bpl_ratios_notime(l_freq[sed], b_fl, b_fr, a_1, a_2, s_s,
                                         fluxratio, freqratio, log=log)
        l_model_flux.append(fval)
    residual = (l_flux - np.array(l_model_flux, dtype=object)) / l_unc

    return np.concatenate(residual)


def fake_freq(low_pow, high_pow):
    return np.logspace(low_pow, high_pow, 100)


def peak_flux(a1, a2, sss):
    G = peak_freq(a1, a2, sss) ** -1.
    rr = 1. / sss
    first = -sss * np.log10(2)
    second = sss * np.log10(G ** (a1 * rr) + G ** (a2 * rr))
    #returns F0/F in linear space
    #where F is true peak
    return np.power(10, first + second) ** -1.


def peak_freq(a1, a2, sss):
    rr = 1 / sss
    beta2 = a2 * rr - 1.
    beta1 = a1 * rr - 1.
    numerator = -a1 / a2
    power = (beta1 - beta2) ** -1.
    #returns f0/f (in linear space)
    #where f is true peak
    return np.power(numerator, power)


def line(f, m, b_prev, log=False):
    b = b_prev if log else np.log10(b_prev)
    return m * f + b
wrapped_line = uncertainties.wrap(line)


def limit(f, breakfreq, breakflux, alpha, s, asym_lim=True):
    fr = np.power(10, f)
    rr = 1. / s
    res = breakflux * (fr / breakfreq) ** alpha
    if asym_lim:
        res *= 2 ** -s
    return np.log10(res)


def limit1(f, a1, a2, s, breakflux, f0, asym_lim=True):
    fr = np.power(10, f)
    rr = 1. / s
    res = breakflux * (fr / f0) ** a1
    if asym_lim:
        res *= 2 ** -s
    return np.log10(res)


def limit2(f, f0, breakflux, a1, a2, s, asym_lim=True):
    fr = np.power(10, f)
    rr = 1. / s
    res = breakflux * (fr / f0) ** a2
    if asym_lim:
        res *= 2 ** -s
    return np.log10(res)


def bpl_ratios_notime(f, breakflux, breakfreq, alpha1, alpha2, s,
                      fluxratio, freqratio, log=False):
    bfl = np.power(10, breakflux) if log else breakflux
    bfr = np.power(10, breakfreq) if log else breakfreq
    Fp0 = np.log10(fluxratio * bfl)
    Fb0 = np.log10(freqratio * bfr)
    rr = 1. / s
    fdiff = f - Fb0
    t1 = np.power(10, fdiff * alpha1 * rr)
    t2 = np.power(10, fdiff * alpha2 * rr)
    return Fp0 + s * (-np.log10(2) + np.log10(t1 + t2))


def asymp_intercept(f, breakflux, breakfreq, alpha1, alpha2, s, log=False):
    """Calculate the log flux value at which the two slopes of the broken
    powerlaw intersect.
    """
    Fp0 = breakflux if log else np.log10(breakflux)
    Fb0 = breakfreq if log else np.log10(breakfreq)
    rr = 1 / s
    fdiff = f - Fb0
    t1 = np.power(10, fdiff * alpha1 * rr)
    t2 = np.power(10, fdiff * alpha2 * rr)
    return Fp0 + s * np.log10(t1 + t2)


def fit_line_with_unc(xdata, ydata, yuncdata, xlog=False, ylog=False):
    x = xdata if xlog else np.log10(xdata)
    y = ydata if ylog else np.log10(ydata)
    y_unc = yuncdata if ylog else np.log10(np.e) * yuncdata / ydata
    params = lmfit.Parameters()
    params.add('m', -1)
    params.add('b', 5)
    mini = lmfit.Minimizer(line_for_minimizer, params=params,
                           fcn_args=(x, y, y_unc), nan_policy='propagate',
                           scale_covar=False, calc_covar=True)
    result = mini.minimize(method='leastsq')

    m = result.params['m']
    b = result.params['b']

    q = ufloat(m.value, m.stderr)
    q_intercept = ufloat(b.value, b.stderr)

    midx = result.var_names.index('m')
    bidx = result.var_names.index('b')

    sig_mm = result.covar[midx, midx]
    sig_bb = result.covar[bidx, bidx]
    sig_bm = result.covar[midx, bidx]

    plot_x = np.log10(np.logspace(min(x), max(x), 100))
    plot_y = q.n * plot_x + q_intercept.n
    plot_y_unc = np.sqrt(sig_mm*plot_x**2 + sig_bb + 2*plot_x*sig_bm)
    plot_lbl = "{:.2f}$\pm${:.2f}".format(q.n, q.s)
    plot_dict = dict(x=plot_x, y=plot_y, y_unc=plot_y_unc, lbl=plot_lbl)

    return q, q_intercept, plot_dict


def minimize_slope(params, xdata, ydata, yuncdata, xlog=False, ylog=False):
    x = xdata if xlog else np.log10(xdata)
    y = ydata if ylog else np.log10(ydata)
    yunc = yuncdata if ylog else np.log10(np.e) * yuncdata / ydata
    fit_params = params.valuesdict()
    model = BPL(x, fit_params['break_y'], fit_params['break_x'],
                fit_params['slope1'], fit_params['slope2'], fit_params['s'],
                xlog=xlog, ylog=ylog)
    return (y - model) / yunc


def line_for_minimizer(pars, x, y, yerr):
    fit_params = pars.valuesdict()
    m = fit_params['m']
    b = fit_params['b']
    return (y - (x * m + b)) / yerr


def BPL(x_range_log, break_y, break_x, slope1, slope2, s,
        xlog=False, ylog=False):
    break_y_log = break_y if ylog else np.log10(break_y)
    break_x_log = break_x if xlog else np.log10(break_x)

    rr = 1 / s
    xdiff = x_range_log - break_x_log
    t1 = np.power(10, xdiff * slope1 * rr)
    t2 = np.power(10, xdiff * slope2 * rr)
    return break_y_log + s * np.log10(t1 + t2)


def BPL_shell_integrand(x_range, break_y, break_x, slope1, slope2, s,
                        xlog=False, ylog=False):
    x_range_log = np.log10(x_range)
    log_bpl = BPL(x_range_log, break_y, break_x, slope1, slope2, s,
                  xlog=xlog, ylog=ylog)
    log_int = log_bpl + (2 * x_range_log) + np.log10(4. * np.pi)
    return np.power(10, log_int)


def BPL_slope_fit(xdata, ydata, yuncdata, break_x_fixed=None,
                  break_y_fixed=None, s_fixed=None, xlog=False, ylog=False):

    if break_x_fixed == None:
        break_x_guess = np.median(xdata)
        break_x_vary = True
    else:
        break_x_guess = break_x_fixed
        break_x_vary = False

    if break_y_fixed == None:
        break_y_guess = np.median(ydata)
        break_y_vary = True
    else:
        break_y_guess = break_y_fixed
        break_y_vary = False

    if s_fixed == None:
        s_guess = np.median(ydata)
        s_vary = True
    else:
        s_guess = s_fixed
        s_vary = False

    params = lmfit.Parameters()
    params.add('slope1', 0.75)
    params.add('slope2', 1.1)
    params.add('break_x', value=break_x_guess, vary=break_x_vary)
    params.add('break_y', value=break_y_guess, vary=break_y_vary)
    params.add('s', min=-0.005, max=0.005, value=s_guess, vary=s_vary)

    mini = lmfit.Minimizer(minimize_slope, params=params,
                           fcn_args=(xdata, ydata, yuncdata),
                           fcn_kws=dict(xlog=xlog, ylog=ylog),
                           nan_policy='propagate',
                           scale_covar=True, calc_covar=True)
    result = mini.minimize(method='leastsq')

    return result


def LinearUnc(result, sed, freq, slope_name, log=False):
    if slope_name not in result.var_names:
        raise ValueError("%s not found", slope_name)

    m = result.params[slope_name]
    b = result.params['b{}'.format(sed)]
    mindex = result.var_names.index(slope_name)
    bindex = result.var_names.index('b{}'.format(sed))

    sigMsq = result.covar[mindex, mindex]
    sigBsq = result.covar[bindex, bindex]
    sigBM = result.covar[mindex, bindex]

    x = np.log10(freq) if log else freq
    dfdm = x if log else b * x ** m * np.log(x)
    dfdb = 1. if log else x ** m

    return np.sqrt(dfdm ** 2 * sigMsq + dfdb ** 2 * sigBsq
                   + 2 * dfdm * dfdb * sigBM)


def bestfitunc(xx, cov):
    x = np.array(xx)
    sigmasquared = x ** 2 * cov[0][0] + cov[1][1] + 2 * x * cov[0][1]
    sigma = np.sqrt(sigmasquared)
    return sigma


