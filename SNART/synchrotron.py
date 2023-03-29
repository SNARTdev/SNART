
import logging
import numpy as np
import pandas as pd
from math import gamma
from scipy.integrate import quad
import lmfit
import uncertainties
from uncertainties import ufloat
from SNART.fitting import (wrapped_line, fit_line_with_unc, BPL,
                           BPL_shell_integrand, BPL_slope_fit)


c1 = 6.27e18
chargeElectron = 4.8032068e-10 #esu = g^1/2 cm^3/2 s^-1
massElectron = 9.1093897e-28 #grams
speedLight = 2.99792458e10 #cm/s
thompsonCross = 6.6e-25 #cm**2
fFactor = 0.5 #1 to 0.5
nuFactor = 5e9 # Hz to 5GHz
distFactor = 3.086e24 #cm in a Mpc
fluxFactor = 1e-23 #cgs to Jy
timeFactor = 86400 #days to seconds
energyLower_nom = massElectron * (speedLight ** 2)
massProton = 1.6726e-24
muE = 1.0


def c5(p):
    pterm = (p + 7. / 3) / (p + 1.)
    gterm = gamma((3. * p - 1.) / 12.) * gamma((3. * p + 7.) / 12.)
    const = chargeElectron ** 3. / (massElectron * speedLight ** 2.)
    return np.sqrt(3) / (16. * np.pi) * const * pterm * gterm


def c6(p):
    pterm = p + 10. / 3
    gterm = gamma((3. * p + 2.) / 12.) * gamma((3. * p + 10.) / 12.)
    const = chargeElectron * massElectron ** 5. * speedLight ** 10.
    return np.sqrt(3) * np.pi / 72. * const * pterm * gterm


def magFieldObs(nu, theta,  p, epsilonE, epsilonB, filling, energyLower,
                fluxPeak, distance):
    d = distance * distFactor
    fl = fluxPeak * fluxFactor
    f = filling * fFactor
    eps_r = epsilonE / epsilonB
    coeff = nu * nuFactor / (2. * c1)
    num = c5(p) * np.sin(theta) ** ((-2. * p - 5.) / 2) * np.pi ** 3.
    num *= 36. * energyLower ** (4. - 2. * p)
    denom = d ** 2. * fl * c6(p) ** 3. * f ** 2. * eps_r ** 2. * (p - 2.) ** 2.
    return coeff * (num / denom) ** (2. / (2. * p + 13.))
wrapped_magfield = uncertainties.wrap(magFieldObs)


def logmagFieldObs(lognu, theta,  p, epsilonE, epsilonB, filling, energyLower,
                   logfluxPeak, distance):
    logcf = lognu + np.log10(nuFactor) - np.log10(2. * c1)

    num = c5(p) * np.sin(theta) ** ((-2. * p - 5.) / 2)
    num *= np.pi ** 3. * 36. * energyLower ** (2. * (2. - p))
    lognum = np.log10(num)

    logarg = distance * distFactor * filling * fFactor
    logarg *= epsilonE / epsilonB * (p - 2.)
    logdenom = 2. * np.log10(logarg) + 3. * np.log10(c6(p))
    logdenom += logfluxPeak + np.log10(fluxFactor)
    return logcf + 2. / (2. * p + 13.) * (lognum - logdenom)
wrapped_logmagfield = uncertainties.wrap(logmagFieldObs)


def rForwardObs(nu, theta,  p, epsilonE, epsilonB, filling, energyLower,
                 fluxPeak, distance):
    expfac = 1. / (13. + 2. * p)

    coeff0 = distFactor ** ((12. + 2. * p) * expfac)
    coeff1 = 1. / (nu * nuFactor) * 2. ** (1. + expfac) * 3. ** expfac * c1
    coeff1 *= c5(p) ** ((-6. - p) * expfac) * c6(p) ** ((5. + p) * expfac)

    num = distance ** (12. + 2. * p) * energyLower ** (2. - p) * epsilonB
    num *= (fluxFactor * fluxPeak) ** (6. + p) * np.pi ** (-5. - p)
    num *= np.sin(theta) ** 2.

    denom = epsilonE * fFactor * filling * (-2. + p)
    return coeff0 * coeff1 * (num / denom) ** expfac
wrapped_radius = uncertainties.wrap(rForwardObs)


def logrForwardObs(lognu, theta,  p, epsilonE, epsilonB, filling, energyLower,
                    logfluxPeak, distance):
    pterm = 1. / (13. + 2. * p)

    logcf1 = (12. + 2. * p) * pterm * np.log10(distFactor)

    logcf2 = -lognu - np.log10(nuFactor) + pterm * np.log10(3)
    logcf2 += (1. + pterm) * np.log10(2) + np.log10(c1)
    logcf2 += ((-6. - p) * pterm) * np.log10(c5(p))
    logcf2 += ((5. + p) * pterm) * np.log10(c6(p))

    lognum = (12. + 2. * p) * np.log10(distance) + 2. * np.log10(np.sin(theta))
    lognum += (2. - p) * np.log10(energyLower) + np.log10(epsilonB)
    lognum += (6. + p) * (logfluxPeak + np.log10(fluxFactor))
    lognum += (-5. - p) * np.log10(np.pi)

    logdenom = np.log10(epsilonE * filling * fFactor * (p - 2.))

    return logcf1 + logcf2 + pterm * (lognum - logdenom)
wrapped_logradius = uncertainties.wrap(logrForwardObs)


def totalEnergyObs(nu, theta, p, epsilonE, epsilonB, filling, energyLower,
                   fluxPeak, distance):
    r = wrapped_radius(nu, theta,  p, epsilonE, epsilonB, filling,
                       energyLower, fluxPeak, distance)
    b = wrapped_magfield(nu, theta,  p, epsilonE, epsilonB, filling,
                         energyLower, fluxPeak, distance)
    return 1. / epsilonB * filling * fFactor * 1. / 6 * r ** 3. * b ** 2.
wrapped_energy = uncertainties.wrap(totalEnergyObs)


def logtotalEnergyObs(lognu, theta, p, epsilonE, epsilonB, filling, energyLower,
                      logfluxPeak, distance):
    logr = wrapped_logradius(lognu, theta,  p, epsilonE, epsilonB, filling,
                             energyLower, logfluxPeak, distance)
    logb = wrapped_logmagfield(lognu, theta,  p, epsilonE, epsilonB, filling,
                               energyLower, logfluxPeak, distance)
    logcf = np.log10(1. / epsilonB * filling * fFactor * 1. / 6.)
    return logcf + 3. * logr + 2. * logb
wrapped_logenergy = uncertainties.wrap(logtotalEnergyObs)


def velocityObs(nu, theta,  p, epsilonE, epsilonB, filling, energyLower,
                fluxPeak, distance, t):
    r = wrapped_radius(nu, theta,  p, epsilonE, epsilonB, filling,
                       energyLower, fluxPeak, distance)
    return r / (t * timeFactor)
wrapped_velocity = uncertainties.wrap(velocityObs)


def logvelocityObs(lognu, theta,  p, epsilonE, epsilonB, filling, energyLower,
                   logfluxPeak, distance, t):
    logr = wrapped_logradius(lognu, theta,  p, epsilonE, epsilonB, filling,
                             energyLower, logfluxPeak, distance)
    return logr - np.log10(t * timeFactor)
wrapped_logvelocity = uncertainties.wrap(logvelocityObs)


def electronNumDenObs(nu, theta,  p, epsilonE, epsilonB, filling, energyLower,
                      t, i, distance, fluxPeak, q):
    b = wrapped_magfield(nu, theta,  p, epsilonE, epsilonB, filling,
                         energyLower, fluxPeak, distance)
    v = wrapped_velocity(nu, theta,  p, epsilonE, epsilonB, filling,
                         energyLower, fluxPeak, distance, t)
    ne = (2. / (i + 1.)) ** -1. * 1. / (epsilonB * massProton)
    ne *= b ** 2. / (8. * np.pi) * (q * v) ** -2. / muE
    return ne
wrapped_ne = uncertainties.wrap(electronNumDenObs)


def logelectronNumDenObs(lognu, theta,  p, epsilonE, epsilonB, filling,
                         energyLower, t, i, distance, logfluxPeak, q):
    logb = wrapped_logmagfield(lognu, theta,  p, epsilonE, epsilonB, filling,
                               energyLower, logfluxPeak, distance)
    logv = wrapped_logvelocity(lognu, theta,  p, epsilonE, epsilonB, filling,
                               energyLower, logfluxPeak, distance, t)
    ans = -np.log10(2. / (i + 1.)) + np.log10(1. / epsilonB)
    ans += 2. * logb - np.log10(massProton * 8 * np.pi * muE)
    ans -= 2. * (np.log10(q) + logv)
    return ans
wrapped_logne = uncertainties.wrap(logelectronNumDenObs)


def massLossObs(nu, theta,  p, epsilonE, epsilonB, filling, energyLower, i,
                fluxPeak, distance, q, t):
    massLossFactor = 1e-4 * 1.989e33 * 1. / 3.154e7
    velocityFactor = 1e3 * 1e5
    n_e = wrapped_ne(nu, theta,  p, epsilonE, epsilonB, filling, energyLower,
                     t, i, distance, fluxPeak, q)
    r = wrapped_radius(nu, theta,  p, epsilonE, epsilonB, filling,
                       energyLower, fluxPeak, distance)
    coeff = muE * massProton * 4. * np.pi * r ** 2.
    return coeff * n_e / massLossFactor * velocityFactor
wrapped_mdotv = uncertainties.wrap(massLossObs)


def logmassLossObs(lognu, theta,  p, epsilonE, epsilonB, filling, energyLower,
                   i, logfluxPeak, distance, q, t):
    massLossFactor = 1e-4 * 1.989e33 * 1. / 3.154e7
    velocityFactor = 1e3 * 1e5
    logne = wrapped_logne(lognu, theta,  p, epsilonE, epsilonB, filling,
                          energyLower, t, i, distance, logfluxPeak, q)
    logr = wrapped_logradius(lognu, theta,  p, epsilonE, epsilonB, filling,
                             energyLower, logfluxPeak, distance)
    logcf1 = np.log10(muE * massProton * 4 * np.pi * velocityFactor)
    logcf2 = np.log10(massLossFactor)
    return logcf1 + 2. * logr + logne - logcf2
wrapped_logmdotv = uncertainties.wrap(logmassLossObs)


def densityObs(nu, theta,  p, epsilonE, epsilonB, filling, energyLower,
               fluxPeak, distance, t, q, i):
    u = wrapped_energy(nu, theta, p, epsilonE, epsilonB, filling,
                       energyLower, fluxPeak, distance)
    r = wrapped_radius(nu, theta,  p, epsilonE, epsilonB, filling,
                       energyLower, fluxPeak, distance)
    v = wrapped_velocity(nu, theta,  p, epsilonE, epsilonB, filling,
                         energyLower, fluxPeak, distance, t)

    vol = 4. / 3 * fFactor * filling * np.pi * r ** 3. * (2. / (i + 1.))
    energy_density = u / vol
    return energy_density * (q * v) ** -2.
wrapped_density = uncertainties.wrap(densityObs)


def logdensityObs(lognu, theta,  p, epsilonE, epsilonB, filling, energyLower,
                  logfluxPeak, distance, t, q, i):
    logu = wrapped_logenergy(lognu, theta, p, epsilonE, epsilonB, filling,
                             energyLower, logfluxPeak, distance)
    logr = wrapped_logradius(lognu, theta,  p, epsilonE, epsilonB, filling,
                             energyLower, logfluxPeak, distance)

    logdenom = np.log10(4./3 * filling * fFactor * np.pi * 2. / (i + 1.))
    logdenom += 3. * logr

    log_energy_density = logu - logdenom

    logv = wrapped_logvelocity(lognu, theta,  p, epsilonE, epsilonB, filling,
                               energyLower, logfluxPeak, distance, t)
    return log_energy_density - 2. * (np.log10(q) + logv)
wrapped_logdensity = uncertainties.wrap(logdensityObs)


def internal_pl(rt_slope, denr_slope):
    """Calculates the powerlaw index of the inner density (shocked material)
    """
    n = (-3 + denr_slope * rt_slope) / (rt_slope - 1)
    logging.debug("The powerlaw density of the shocked medium is %s", n)
    return n


def synch_cooling_freq(nu, theta,  p, epsilonE, epsilonB, filling,
                       energyLower, t, i, distance, fluxPeak, q):
    b = wrapped_magfield(nu, theta,  p, epsilonE, epsilonB, filling,
                         energyLower, fluxPeak, distance)
    num = 18. * np.pi * massElectron * speedLight * chargeElectron
    den = thompsonCross ** 2. * b ** 3. * (t * timeFactor) ** 2.
    return 1e-9 * num / den
wrapped_synch_cooling_freq = uncertainties.wrap(synch_cooling_freq)


def log_synch_cooling_freq(lognu, theta,  p, epsilonE, epsilonB, filling,
                           energyLower, t, i, distance, logfluxPeak, q):
    logb = wrapped_logmagfield(lognu, theta,  p, epsilonE, epsilonB, filling,
                               energyLower, logfluxPeak, distance)

    lognum = np.log10(18 * np.pi * massElectron * (speedLight * chargeElectron))

    logden = 2. * np.log10(thompsonCross) + 3. * logb
    logden += 2. * np.log10(t * timeFactor)
    return lognum - logden - 9.
wrapped_log_synch_cooling_freq = uncertainties.wrap(log_synch_cooling_freq)


def calculate_physical_params(data, fitter, opts):
    phys_data, phys_fits = create_phys_param_df(data, fitter, opts)
    fitter.result.phys_data = phys_data
    fitter.result.phys_fits = phys_fits
    if opts.equipartition:
        phys_data_equi, _ = create_phys_param_df(data, fitter, opts, equi=True)
        fitter.result.phys_data_equi = phys_data_equi
    return


def create_phys_param_df(data, fitter, opts, equi=False):
    res = fitter.result
    p = res.linked_variables['p']
    alpha1 = ufloat(res.obs.params['alpha1'].value,
                    res.obs.params['alpha1'].stderr)
    eB = 1. / 3 if equi else opts.epsilonB
    eE = 1. / 3 if equi else opts.epsilonE
    epsilonB = ufloat(eB, 0.)
    epsilonE = ufloat(eE, 0.)
    filling = ufloat(opts.fill_nom, 0.)
    i = ufloat(opts.i_nom, 0.)
    distance = ufloat(data.distance, 0.)
    theta = ufloat(opts.theta_nom,0.)
    energyLower = ufloat(energyLower_nom, 0.)

    # Initialize new dataframe
    d = {'SED': fitter.sed_list}
    new_df = pd.DataFrame(data=d, columns=['SED'])

    vars_with_unc = [res.linked_variables[var] for var in
                     res.var_names]
    # Add B, U, R
    df_wo_qsnnmd = put_bur_in_df(
        fitter.sed_list, res.linked_variables, fitter.poly_a1,
        fitter.poly_a2, alpha1, new_df, data.df_obs,
        theta, epsilonE, epsilonB, filling, energyLower, distance,
        log=fitter.log)

    # check that we have more than one SED peak in order to calculate
    # s and q and other physical parameters
    dates = np.array([
        w for w,l in zip(df_wo_qsnnmd['date'], df_wo_qsnnmd['limit'])
        if l != 'limit']) # all the non limit dates
    if len(dates) <= 1:
        return df_wo_qsnnmd

    df_wo_snnmd, q_fits = put_q_in_df(
        df_wo_qsnnmd, fitter.sed_list, two_slope=False, log=fitter.log)

    df_wo_sn = put_nmd_in_df(fitter.sed_list, fitter.poly_a1,
                             fitter.poly_a2, df_wo_snnmd, theta, p,
                             epsilonE, epsilonB, filling, energyLower,
                             distance, i, log=fitter.log)

    df_wo_n, s_fits = put_s_in_df(df_wo_sn, two_slope=fitter.two_slope,
                                  log=fitter.log)


    df_all_phys_params = put_n_in_df(fitter.sed_list, df_wo_n)


    return df_all_phys_params, {**q_fits, **s_fits}


def put_bur_in_df(sed_list, linked_vars, poly_a1, poly_a2, saved_alpha1,
                  df_add_bur, data_df, theta, epsilonE, epsilonB, filling,
                  energyLower, distance, log=False):
    magfunc = wrapped_logmagfield if log else wrapped_magfield
    energyfunc = wrapped_logenergy if log else wrapped_energy
    radfunc = wrapped_logradius if log else wrapped_radius
    p = linked_vars["p"]
    for sed in sed_list:
        date = np.median(data_df.loc[(data_df['SED Number']==sed), 'Date'])
        df_add_bur.loc[(df_add_bur['SED']==sed),'date'] = date
        if (sed not in poly_a1) and (sed not in poly_a2):
            nu = linked_vars["brk_freq{}".format(sed)]
            fluxPeak = linked_vars["brk_flux{}".format(sed)]

            mag = magfunc(nu, theta, p, epsilonE, epsilonB, filling,
                          energyLower, fluxPeak, distance)
            u = energyfunc(nu, theta, p, epsilonE, epsilonB, filling,
                           energyLower, fluxPeak, distance)
            radius = radfunc(nu, theta, p, epsilonE, epsilonB, filling,
                             energyLower, fluxPeak, distance)

            df_add_bur.loc[(df_add_bur['SED']==sed),'nu pk'] = nu
            df_add_bur.loc[(df_add_bur['SED']==sed),'flux pk'] = fluxPeak
            df_add_bur.loc[(df_add_bur['SED']==sed),'b'] = mag
            df_add_bur.loc[(df_add_bur['SED']==sed),'u'] = u
            df_add_bur.loc[(df_add_bur['SED']==sed),'r'] = radius

        if sed in poly_a1: # this means to look in the thin SPLs
            # this will provide an upper limit on B, a lower limit on R,
            # and a lower limit on U.
            min_freq = min((data_df.loc[(data_df["SED Number"]==sed),
                                        "Frequency"]).to_numpy())
            calc_max_flux = wrapped_line(np.log10(min_freq), saved_alpha1,
                                         linked_vars["b{}".format(sed)],
                                         log=log)
            calc_max_flux = np.power(10, calc_max_flux)
            max_flux = calc_max_flux.nominal_value + calc_max_flux.std_dev

            if log:
                min_freq = np.log10(min_freq)
                max_flux = np.log10(max_flux)

            mag_lim = magfunc(min_freq, theta.n, p.n, epsilonE.n, epsilonB.n,
                              filling.n, energyLower.n, max_flux, distance.n)
            u_lim = energyfunc(min_freq, theta.n, p.n, epsilonE.n,
                               epsilonB.n, filling.n, energyLower.n,
                               max_flux, distance.n)
            r_lim = radfunc(min_freq, theta.n, p.n, epsilonE.n, epsilonB.n,
                            filling.n, energyLower.n, max_flux, distance.n)

            df_add_bur.loc[(df_add_bur['SED']==sed), 'nu pk'] = min_freq
            df_add_bur.loc[(df_add_bur['SED']==sed), 'flux pk'] = max_flux
            df_add_bur.loc[(df_add_bur['SED']==sed), 'b'] = mag_lim
            df_add_bur.loc[(df_add_bur['SED']==sed), 'u'] = u_lim
            df_add_bur.loc[(df_add_bur['SED']==sed), 'r'] = r_lim
            df_add_bur.loc[(df_add_bur['SED']==sed), 'limit'] = 'limit'

        if sed in poly_a2:  # this means to look in the thick SPLs
            # this will provide no limits on any physical parameter.
            max_freq = max((data_df.loc[(data_df["SED Number"]==sed),
                                        "Frequency"]).to_numpy())
            calc_max_flux = wrapped_line(np.log10(max_freq),
                                         linked_vars["alpha2"],
                                         linked_vars["b{}".format(sed)],
                                         log=log)
            calc_max_flux = np.power(10, calc_max_flux)
            max_flux = calc_max_flux.nominal_value + calc_max_flux.std_dev

            if log:
                max_freq = np.log10(max_freq)
                max_flux = np.log10(max_flux)

            df_add_bur.loc[(df_add_bur['SED']==sed),'nu pk'] = max_freq
            df_add_bur.loc[(df_add_bur['SED']==sed),'flux pk'] = max_flux
            df_add_bur.loc[(df_add_bur['SED']==sed),'limit'] = 'limit'

    return df_add_bur


def put_q_in_df(df_add_q, sed_list, two_slope=False, log=False):
    mask = df_add_q['limit'].to_numpy() != 'limit'
    dates = df_add_q.loc[mask, 'date'].to_numpy()
    r = np.array([w.n for w in df_add_q.loc[mask, 'r']])
    r_unc = np.array([w.std_dev for w in df_add_q.loc[mask, 'r']])
    b = np.array([w.n for w in df_add_q.loc[mask, 'b']])
    b_unc = np.array([w.std_dev for w in df_add_q.loc[mask, 'b']])
    well_sampled_dates = np.logspace(
        np.log10(min(dates)), np.log10(max(dates)), 100)

    if len(dates) <= 1:
        q = ufloat(0.88, 0)
        df_add_q['q'] = q
        logging.debug("There are either one or zero peaks in this set of SEDs."
                      " Setting (Chevalier assumption) arbitrary q=%s", q)
        return df_add_q, r_ax, mag_ax

    if two_slope:
        logx = np.log10(well_sampled_dates)
        bpl_pars = ['break_y', 'break_x', 'slope1', 'slope2', 's']

        q_fit = BPL_slope_fit(dates, r, r_unc, s_fixed=0.01,
                              break_x_fixed=140.14, ylog=log)
        logging.info("q report: \n%s", lmfit.fit_report(q_fit))

        # save R vs t arrays for plotting
        yrt = BPL(logx, *[q_fit[par].value for par in bpl_pars], ylog=log)
        rt_dict = dict(x=logx, y=yrt)

        # NOTE: are these slopes in the right order? Other first and
        # second values have flipped slope numbers (e.g. b_slope below)
        q1 = ufloat(q_fit.params['slope1'].value, q_fit.params['slope1'].stderr)
        q2 = ufloat(q_fit.params['slope2'].value, q_fit.params['slope2'].stderr)
        qbx = q_fit.params['break_x'].value

        df_add_q.loc[(df_add_q['date'] <= qbx) & mask,'q'] = q1
        df_add_q.loc[(df_add_q['date'] > qbx) & mask,'q'] = q2

        b_slope_fit = BPL_slope_fit(dates, b, b_unc, s_fixed=-0.01, ylog=log)
        logging.debug("b slope report: \n%s", lmfit.fit_report(b_slope_fit))

        # save B vs t arrays for plotting
        ybt = BPL(logx, *[b_slope_fit.params[par].value for par in bpl_pars],
                  ylog=log)
        bt_dict = dict(x=logx, y=ybt)

        b_slope = ufloat(b_slope_fit.params['slope2'].value,
                         b_slope_fit.params['slope2'].stderr)
        second_b_slope = ufloat(b_slope_fit.params['slope1'].value,
                                b_slope_fit.params['slope1'].stderr)
        bbx = b_slope_fit.params['break_x'].value

        df_add_q.loc[(df_add_q['date'] <= bbx) & mask,'b slope'] = b_slope
        df_add_q.loc[(df_add_q['date'] > bbx) & mask,'b slope'] = second_b_slope

        for sed in sed_list:
            if np.array(df_add_q.loc[(df_add_q['SED'] == sed), 'limit']) == 'limit':
                q_prev = np.array(df_add_q.loc[(df_add_q['SED'] == (sed-1)), 'q'])
                df_add_q.loc[(df_add_q['SED'] == sed), 'q'] = q_prev

    else:
        q, qint, rt_dict = fit_line_with_unc(dates, r, r_unc, ylog=log)
        _, _, bt_dict = fit_line_with_unc(dates, b, b_unc, ylog=log)
        df_add_q['q'] = q
        logging.debug("R(t) and B(t) have been fit to a single line, "
                      "rather than a broken powerlaw.")

    return df_add_q, dict(r_vs_t=rt_dict, b_vs_t=bt_dict)


def put_s_in_df(df_add_s, two_slope=False, log=False):
    mask = df_add_s['limit'].to_numpy() != 'limit'
    dates = df_add_s.loc[mask, 'date'].to_numpy()
    r = np.array([w.n for w in df_add_s.loc[mask, 'r']])
    den = np.array([w.n for w in df_add_s.loc[mask, 'den']])
    den_unc = np.array([w.std_dev for w in df_add_s.loc[mask, 'den']])
    b = np.array([w.n for w in df_add_s.loc[mask, 'b']])
    b_unc = np.array([w.std_dev for w in df_add_s.loc[mask, 'b']])
    rmin = min(r) if log else np.log10(min(r))
    rmax = max(r) if log else np.log10(max(r))
    well_sampled_r = np.logspace(rmin, rmax, 100)

    gToSolarMass = 1.989e33

    if len(dates) <= 1:
        s = ufloat(-2, 0)
        df_add_s['s profile'] = s
        logging.debug("There are either one or zero peaks in this set of SEDs."
                      " Setting (constant density) arbitrary s profile=%s", s)
        return df_add_s

    if two_slope:
        logx = np.log10(well_sampled_r)
        bpl_pars = ['break_y', 'break_x', 'slope1', 'slope2', 's']

        s_slope_fit = BPL_slope_fit(r, den, den_unc, s_fixed=-0.01,
                                    xlog=log, ylog=log)
        logging.debug("s slope report: \n%s",
                      lmfit.fit_report(s_slope_fit, show_correl=False))

        # save den vs R arrays for plotting
        ydenr = BPL(logx, *[s_slope_fit.params[par].value for par in bpl_pars],
                    xlog=log, ylog=log)
        denr_dict = dict(x=logx, y=ydenr)

        sss = ufloat(s_slope_fit.params['slope2'].value,
                     s_slope_fit.params['slope2'].stderr)
        second_sss = ufloat(s_slope_fit.params['slope1'].value,
                            s_slope_fit.params['slope1'].stderr)
        sbx = s_slope_fit.params['break_x'].value

        df_add_s.loc[(df_add_s['r'] <= sbx) & mask, 's profile'] = sss
        # NOTE should this be masked also?
        df_add_s.loc[(df_add_s['r'] > sbx), 's profile'] = second_sss

        br_slope_fit = BPL_slope_fit(r, b, b_unc, s_fixed=-0.01,
                                     xlog=log, ylog=log)

        logging.debug("br slope report: \n%s",
                      lmfit.fit_report(br_slope_fit, show_correl=False))

        # save B vs R arrays for plotting
        ybr = BPL(logx, *[br_slope_fit.params[par].value for par in bpl_pars],
                  xlog=log, ylog=log)
        br_dict = dict(x=logx, y=ybr)

        br_slope = ufloat(br_slope_fit.params['slope2'].value,
                          br_slope_fit.params['slope2'].stderr)
        second_br_slope = ufloat(br_slope_fit.params['slope1'].value,
                                 br_slope_fit.params['slope1'].stderr)
        brbx = br_slope_fit.params['break_x'].value

        df_add_s.loc[(df_add_s['r'] <= brbx) & mask,'b vs r profile'] = br_slope
        # NOTE should this be masked also?
        df_add_s.loc[(df_add_s['r'] > brbx),'b vs r profile'] = second_br_slope

        integral = quad(BPL_shell_integrand, min(well_sampled_r),
                        max(well_sampled_r),
                        args=(s_slope_fit.params['break_y'].value, sbx,
                              s_slope_fit.params['slope1'].value,
                              s_slope_fit.params['slope2'].value,
                              s_slope_fit.params['s'].value,
                              log, log),
                        points=[sbx])

        df_add_s['integrated mass'] = ufloat(*integral) / gToSolarMass

        if (sss < 3) or (second_sss < 3):
            logging.debug("WARNING: you have an ambient power law < 3, "
                          "cannot use self similar solutions from Chevalier "
                          "1982")

    else:
        sss, bbb, denr_dict = fit_line_with_unc(r, den, den_unc,
                                                xlog=log, ylog=log)
        rtmin = np.power(10, min(r)) if log else min(r)
        rtmax = np.power(10, max(r)) if log else max(r)
        rterm = (rtmax ** (sss.n + 3) - rtmin ** (sss.n + 3)) / (sss.n + 3)
        integral = 10 ** bbb.n * 4 * np.pi * rterm  # analytic solution
        if (sss < 3):
            logging.debug("WARNING: you have an ambient power law < 3, "
                          "cannot use self similar solutions from Chevalier "
                          "1982")

        df_add_s['s profile'] = sss
        df_add_s['integrated mass'] = integral / gToSolarMass

        _, _, br_dict = fit_line_with_unc(r, b, b_unc, xlog=log, ylog=log)

    return df_add_s, dict(den_vs_r=denr_dict, b_vs_r=br_dict)


def put_nmd_in_df(sed_list, poly_a1, poly_a2, df_add_nmd, theta, p, epsilonE,
                  epsilonB, filling, energyLower, distance, i, log=False):
    nefunc = wrapped_logne if log else wrapped_ne
    mdotvfunc = wrapped_logmdotv if log else wrapped_mdotv
    denfunc = wrapped_logdensity if log else wrapped_density
    vfunc = wrapped_logvelocity if log else wrapped_velocity
    csyncfunc = wrapped_log_synch_cooling_freq if log else \
                wrapped_synch_cooling_freq
    for sed in sed_list:
        q = np.array(df_add_nmd.loc[df_add_nmd['SED']==sed,'q'])[0]
        day = ufloat(df_add_nmd.loc[df_add_nmd['SED']==sed,'date'], 0)

        if (sed not in poly_a1) and (sed not in poly_a2):
            nu = np.array(df_add_nmd.loc[df_add_nmd['SED']==sed,
                                         'nu pk'])[0]
            fluxPeak = np.array(df_add_nmd.loc[df_add_nmd['SED']==sed,
                                               'flux pk'])[0]

            n_e = nefunc(nu, theta,  p, epsilonE, epsilonB, filling,
                         energyLower, day, i, distance, fluxPeak, q)
            m_v = mdotvfunc(nu, theta,  p, epsilonE, epsilonB, filling,
                            energyLower, i, fluxPeak, distance, q, day)
            density = denfunc(nu, theta,  p, epsilonE, epsilonB, filling,
                              energyLower, fluxPeak, distance, day, q,i)
            v = vfunc(nu, theta,  p, epsilonE, epsilonB, filling, energyLower,
                      fluxPeak, distance, day)

            df_add_nmd.loc[(df_add_nmd['SED']==sed),'n_e'] = n_e
            df_add_nmd.loc[(df_add_nmd['SED']==sed),'mv'] = m_v
            df_add_nmd.loc[(df_add_nmd['SED']==sed),'v'] = v
            df_add_nmd.loc[(df_add_nmd['SED']==sed),'den'] = density

            cool_sync = csyncfunc(nu, theta,  p, epsilonE, epsilonB, filling,
                                     energyLower, day, i, distance, fluxPeak, q)
            logging.debug("Synchrotron cooling takes place in sed %s at %s "
                          "%sGHz", sed, cool_sync, "(log) " if log else "")
            df_add_nmd.loc[(df_add_nmd['SED']==sed),'coolingSync'] = cool_sync

        if sed in poly_a1: #this means to look in the thin SPLs
            min_freq = np.array(df_add_nmd.loc[(df_add_nmd['SED']==sed),
                                               'nu pk'])
            max_flux = np.array(df_add_nmd.loc[(df_add_nmd['SED']==sed),
                                               'flux pk'])
            n_e_lim = nefunc(min_freq, theta.n, p.n, epsilonE.n, epsilonB.n,
                             filling.n, energyLower.n, day.n, i.n, distance.n,
                             max_flux, q.n)
            m_v_lim = mdotvfunc(min_freq, theta.n, p.n, epsilonE.n, epsilonB.n,
                                filling.n, energyLower.n, i.n, max_flux,
                                distance.n, q.n, day.n)
            v_lim = vfunc(min_freq, theta.n, p.n, epsilonE.n, epsilonB.n,
                          filling.n, energyLower.n, max_flux,  distance.n,
                          day.n)

            df_add_nmd.loc[(df_add_nmd['SED']==sed), 'n_e'] = n_e_lim
            df_add_nmd.loc[(df_add_nmd['SED']==sed), 'mv'] = m_v_lim
            df_add_nmd.loc[(df_add_nmd['SED']==sed), 'v'] = v_lim

    return df_add_nmd


def put_n_in_df(sed_list, df_add_n):
    for sed in sed_list:
        sss = df_add_n.loc[df_add_n['SED']==sed,'s profile']
        q = df_add_n.loc[df_add_n['SED']==sed,'q']
        df_add_n.loc[df_add_n['SED']==sed,'n profile'] = internal_pl(q, -1*sss)
    return df_add_n


def const_m_loss(r, mdot, vw):
    c = 1.989e21 / 3.15
    den = 4. * np.pi * vw * r ** 2.
    return c * mdot / den


def r_to_t(r):
    return r - np.log10(1e8 * 86400)


def t_to_r(t):
    return t + np.log10(1e8 * 86400)


