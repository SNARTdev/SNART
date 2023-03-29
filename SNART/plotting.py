import os
import logging
import numpy as np
import seaborn as sns
from uncertainties import ufloat
from SNART.fitting import (peak_flux, peak_freq, fake_freq, bpl_ratios_notime,
                           limit2, asymp_intercept, LinearUnc, limit)
from SNART.SED_from_B_R import wrapped_prop_b_and_r, wrapped_flux, wrapped_nu
from SNART.synchrotron import r_to_t, t_to_r, const_m_loss, speedLight
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib.lines import Line2D
matplotlib.rcParams['font.sans-serif'] = 'League Gothic'
matplotlib.rcParams['font.family'] = 'sans-serif'


def make_all_plots(data, fitter, opts, propagate_days=None):
    # how sad that this is a performance improvement...
    dummyfig = plt.figure()

    if opts.plot_quicklook:
        logging.info("Making quicklook plot")
        make_quicklook_plot(data, opts.output_dir)

    logging.info("Plotting SEDs")
    plot_seds(data, fitter, opts.output_dir, theta_nom=opts.theta_nom,
              epsilonE=opts.epsilonE, epsilonB=opts.epsilonB,
              fill_nom=opts.fill_nom, propagate_days=propagate_days)

    logging.info("Plotting physical parameters")
    plot_physical_params(fitter, data.name, opts.output_dir,
                         opts.epsilonE, opts.epsilonB)
    plt.close(dummyfig)
    return


def baseplot(xdata, ydata, title, xlabel, ylabel, axname=False, figname=False,
             new=False, alpha=0.5, color='black', grid=False, marker='o',
             xerror=None, yerror=None, legend_title=False, legend_font=5,
             legend_cols=1, ptlabel=None, xscale='linear', yscale='linear',
             rtrn=False, savename=False, show=False):

    if new:
        figname, axname = plt.subplots()

    axname.errorbar(xdata, ydata, yerr=yerror, xerr=xerror, fmt=marker,
                    alpha=alpha, label=ptlabel, color=color, capsize=5)
    axname.set(title=title, ylabel=ylabel, xlabel=xlabel,
               xscale=xscale, yscale=yscale)

    if ptlabel:
        axname.legend(title=legend_title, fontsize=legend_font,
                      ncol=legend_cols)

    if grid:
        axname.grid(color='grey')

    if savename:
        figname.savefig('{}.pdf'.format(savename))

    if show:
        plt.show()

    if rtrn:
        return axname, figname
    else:
        plt.close(figname)
    return


def plot_seds(data, fitter, output_dir, propagate_days=None,
              theta_nom=np.pi/2, fill_nom=1.0, epsilonE=0.1, epsilonB=0.01):
    ls = '-'
    ms = 'D'

    df = data.df
    res = fitter.result.disp
    log = fitter.log
    palette = sns.husl_palette(len(df["Flux Density"]) / 2)
    fig_all, ax_all = plt.subplots()
    plotpath = os.path.join(output_dir, data.name, 'Individual_SEDs')
    os.makedirs(plotpath, exist_ok=True)

    fit_alpha1 = res.params['alpha1'].value
    fit_alpha2 = res.params['alpha2'].value
    fit_s = res.params['s'].value

    fluxratio = peak_flux(fit_alpha1, fit_alpha2, fit_s)
    freqratio = peak_freq(fit_alpha1, fit_alpha2, fit_s)

    legend1_elements = []
    full_SED_dates=[]

    for sed in fitter.sed_list:
        logging.info("Plotting SED %s", sed)
        fig_indv, ax_indv = plt.subplots()

        date = round(np.median(df.loc[df["SED Number"]==sed,"Date"]), 2)
        freq = (df.loc[(df["SED Number"]==sed),"Frequency"]).to_numpy()
        flux = (df.loc[(df["SED Number"]==sed),"Flux Density"]).to_numpy()
        unc = (df.loc[(df["SED Number"]==sed),"Uncertainty"]).to_numpy()
        fvals = fake_freq(np.log10(min(freq)), np.log10(max(freq)))
        fvals_turn = fake_freq(np.log10(min(freq/2)), np.log10(max(freq*2)))
        flogvals = np.log10(fvals)
        flogvals_turn = np.log10(fvals_turn)

        ax_indv.errorbar(freq, flux, yerr=unc, fmt='o',
                         alpha=0.3, color=palette[int(sed)])
        ax_indv.set(xscale='log', yscale='log')
        legend1_elements.append(Line2D([0], [0], marker='o', color='w',
                                alpha=0.3, label=date, markersize=8,
                                markerfacecolor=palette[int(sed)]))

        ax_all.errorbar(freq, flux, yerr=unc, fmt='o', alpha=0.3,
                        color=palette[int(sed)])
        ax_all.set(xscale='log', yscale='log')

        if (sed not in fitter.poly_a1) and (sed not in fitter.poly_a2):
            full_SED_dates.append(date)

            p_fl = res.params['brk_flux{}'.format(sed)]
            p_fr = res.params['brk_freq{}'.format(sed)]
            if not log:
                p_fl_err = p_fl.stderr
                p_fr_err = p_fr.stderr
            else:
                logpfl_std = p_fl.stderr
                logpfr_std = p_fr.stderr
                p_fl = np.power(10, p_fl)
                p_fr = np.power(10, p_fr)
                p_fl_err = [[p_fl - p_fl * np.power(10, -logpfl_std)],
                            [p_fl * np.power(10, logpfl_std) - p_fl]]
                p_fr_err = [[p_fr - p_fr * np.power(10, -logpfr_std)],
                            [p_fr * np.power(10, logpfr_std) - p_fr]]


            if fitter.asym_lim:
                yy = asymp_intercept(flogvals_turn, p_fl, p_fr, fit_alpha1,
                                     fit_alpha2, fit_s)
            else:
                yy = bpl_ratios_notime(flogvals, p_fl, p_fr, fit_alpha1,
                                       fit_alpha2, fit_s, fluxratio, freqratio)

            ax_indv.plot(fvals_turn, np.power(10, yy), alpha=0.3, linestyle=ls,
                         color=palette[int(sed)])
            ax_all.plot(fvals_turn, np.power(10, yy), alpha=0.5, linestyle=ls,
                        color=palette[int(sed)])


            ax_all.errorbar(p_fr, p_fl, yerr=p_fl_err, xerr=p_fr_err,
                            fmt=ms, alpha=0.6, color=palette[int(sed)])
            ax_indv.errorbar(p_fr, p_fl, yerr=p_fl_err, xerr=p_fr_err,
                             fmt=ms, alpha=0.6, color=palette[int(sed)],
                             label=date)

            # NOTE should put plotting of asymptotes on a switch
            #plot_asymptotes(ax_indv, flogvals, fit_alpha1, fit_alpha2, fit_s,
            #                p_fl, p_fr, asym_lim=fitter.asym_lim)
            #plot_asymptotes(ax_all, flogvals_turn, fit_alpha1, fit_alpha2,
            #                fit_s, p_fl, p_fr, asym_lim=fitter.asym_lim)

            ax_indv.legend(title='Days Since Explosion')

        else:
            SPL_slope = fit_alpha1 if sed in fitter.poly_a1 else fit_alpha2
            alp = 'alpha1' if sed in fitter.poly_a1 else 'alpha2'
            bres = res.params['b{}'.format(sed)]

            bres = np.power(10, bres) if log else bres

            yvals = bres * np.power(fvals, SPL_slope)
            ax_indv.plot(fvals, yvals, linestyle='--', alpha=0.2,
                         label=SPL_slope, color=palette[int(sed)])

            ax_all.plot(fvals, yvals, linestyle=ls, alpha=0.2,
                        color=palette[int(sed)])

            # lower and upper uncertainty limits
            y_unc = LinearUnc(res, sed, fvals, alp, log=log)
            y_lo = yvals * np.power(10, -y_unc) if log else yvals - y_unc
            y_hi = yvals * np.power(10, y_unc) if log else yvals + y_unc

            ax_indv.plot(fvals, y_lo, color='blue', alpha=0.3)
            ax_indv.plot(fvals, y_hi, color='blue', alpha=0.3)
            ax_indv.fill_between(fvals, y_lo, y_hi,
                                 facecolor=palette[int(sed)], alpha=0.2,
                                 interpolate=True)

            ax_indv.legend(title='Slope')

        ax_indv.set(title='SED No. {}'.format(sed),
                    xlabel='freq GHz', ylabel='Flux microJy')


        fig_indv.savefig('{}/{}_SED.pdf'.format(plotpath, sed))
        plt.close(fig_indv)

    # NOTE propagating BPL disabled temporarily
    #allpars = comparative_case == "All_Params_Vary"
    #if propagate_days is not None and allpars:
    #    t = float(cp.get('plot', 'desired-time')) * 365
    #    logging.info("Plotting time-evolved SEDs")
    #    ax_all = propagate_bpl_test(ax_all, data, fitter,
    #                                propagate_days, output_dir,
    #                                theta_nom=theta_nom, eE=epsilonE,
    #                                eB=epsilonB, fill_nom=fill_nom,
    #                                comparative_case=comparative_case)
    #    legend1_elements.append(
    #        Line2D([0], [0], marker='*', color='w', alpha=0.3,
    #               label='projected {} days'.format(t),
    #               markerfacecolor='black', markersize=8))

    # add legends to all-SED plot
    legend1 = ax_all.legend(handles=legend1_elements,
                            title='Days Since Explosion', ncol=5,
                            fontsize=4.5, framealpha=0.5,
                            loc='lower left')

    ax_all.set(title='SED Evolution of SN{}'.format(data.name),
               xlabel='Log Frequency (GHz)',
               ylabel= r'Log Flux Density ($\mu$Jy)')

    fig_all.savefig('{}/{}/all.pdf'.format(output_dir, data.name))
    plt.close(fig_all)

    return np.array(full_SED_dates), ax_all, fig_all, legend1_elements


def plot_asymptotes(ax, f_values, fit_alpha1, fit_alpha2, fit_s, p_fl, p_fr,
                    asym_lim=True):
    """Plots asymptotes of the two power laws as a diagnostic for the
    asymptotic-limit analysis.
    """
    if asym_lim:
        uncorrected_F = p_fl
        uncorrected_nu = p_fr
    else:
        uncorrected_F = peak_flux(fit_alpha1, fit_alpha2, fit_s) * p_fl
        uncorrected_F *= 2 ** -fit_s  # comes from F_chev = F_peak * 2^-s
        uncorrected_nu = peak_freq(fit_alpha1, fit_alpha2, fit_s) * p_fr
    l1 = limit(f_values, uncorrected_nu, uncorrected_F, fit_alpha1, fit_s,
               asym_lim=asym_lim)
    l2 = limit(f_values, uncorrected_nu, uncorrected_F, fit_alpha2, fit_s,
               asym_lim=asym_lim)
    flin = np.power(10, f_values)
    ax.plot(flin, np.power(10, l1), ls='--', color='grey', alpha=0.1)
    ax.plot(flin, np.power(10, l2), ls='--', color='grey', alpha=0.1)
    ax.axhline(uncorrected_F, color='grey', alpha=0.2)
    ax.axvline(uncorrected_nu, alpha=0.3, color='grey')


def make_quicklook_plot(data, output_dir):
    seds = data.df['SED Number'].to_numpy()
    dates = data.df['Date'].to_numpy()
    unc = data.df['Uncertainty'].to_numpy()
    freq = data.df['Frequency'].to_numpy()
    flux = data.df['Flux Density'].to_numpy()
    palette = sns.color_palette("icefire", max(seds) * 3)
    fig, ax = plt.subplots()
    for dd in np.arange(len(dates)):
        cidx = round(dates[dd]/max(dates) * 10 + 1)
        ax, fig = baseplot(
            np.log10(freq[dd]), np.log10(flux[dd]),
            'Quick Glance at Data Points', 'Log Frequency (GHz)',
            'Log Flux (ujy)', new=False, axname=ax, figname=fig, alpha=0.5,
            color=palette[cidx], grid=False, marker='o',
            yerror=0.43*unc[dd]/flux[dd], legend_title='Point Dates',
            legend_font=5, legend_cols=2,
            ptlabel='{}, SED {}'.format(round(dates[dd]), seds[dd]),
            xscale='linear', yscale='linear', rtrn=True)
    fig.savefig('{}/{}/quicklook_data.pdf'.format(output_dir, data.name))
    plt.close(fig)
    return


def plot_physical_params(fitter, name, output_dir, epsilonE, epsilonB):
    ls = '-'
    ms = 'D'
    ccc = '#c79b5e'

    df_phys = fitter.result.phys_data
    df_phys_equi = fitter.result.phys_data_equi
    fits = fitter.result.phys_fits
    log = fitter.log
    plotpath = os.path.join(output_dir, name, 'Physical_Params')
    os.makedirs(plotpath, exist_ok=True)

    mask = df_phys['limit'].to_numpy() != 'limit'
    dates = df_phys.loc[mask, 'date'].to_numpy()
    limit_dates = df_phys.loc[~mask, 'date'].to_numpy()

    plotargs = [df_phys, dates, limit_dates, mask, epsilonE, epsilonB, name,
                ccc, ms, fits]
    btfig = plot_b_vs_t(*plotargs, df_equi=df_phys_equi, log=log)
    utfig = plot_u_vs_t(*plotargs, df_equi=df_phys_equi, log=log)
    rtfig = plot_r_vs_t(*plotargs, df_equi=df_phys_equi, log=log)
    brfig = plot_b_vs_r(*plotargs, df_equi=df_phys_equi, log=log)

    if len(dates) <= 1:
        return

    denrfig = plot_den_vs_r(*plotargs, df_equi=df_phys_equi, log=log)
    netfig = plot_ne_vs_t(*plotargs, df_equi=df_phys_equi, log=log)
    mvtfig = plot_mv_vs_t(*plotargs, df_equi=df_phys_equi, log=log)
    vtfig = plot_v_vs_t(*plotargs, df_equi=df_phys_equi, log=log)
    mvrfig = plot_mv_vs_r(*plotargs, df_equi=df_phys_equi, log=log)
    mv2rfig = plot_mv2_vs_r(*plotargs, log=log)

    figs = [btfig, utfig, rtfig, brfig, denrfig, netfig, mvtfig, vtfig,
            mvrfig, mv2rfig]
    fnames = ['b_vs_t', 'u_vs_t', 'r_vs_t', 'b_vs_r', 'den_vs_r', 'ne_vs_t',
              'mv_vs_t', 'v_vs_t', 'mv_vs_r', 'mv2_vs_r']
    for fig, fname in zip(figs, fnames):
        fig.savefig('{}/{}.pdf'.format(plotpath, fname), bbox_inches='tight')
        plt.close(fig)
    return


def plot_b_vs_t(df, dates, limdates, mask, eE, eB, name, color,
                marker, fits, df_equi=None, log=False):
    logging.info("Plotting B(t)")
    b, b_unc, b_lim = plotvals_from_df(df, 'b', mask, log=log)
    ax, fig = baseplot(
        np.log10(dates), b,
        r'{} B vs t $\epsilon_E$={} $\epsilon_B$={}'.format(name, eE, eB),
        'Time Since Explosion (days)', 'Magnetic Field (Gauss)', new=True,
        yerror=b_unc, color=color, marker=marker, rtrn=True)
    ax.scatter(np.log10(limdates), b_lim, marker='$\u2193$',
               alpha=0.5, color=color)
    plot_phys_param_fit(ax, fits['b_vs_t'])
    if df_equi is not None:
        b_e, _, b_lim_e = plotvals_from_df(df_equi, 'b', mask, log=log)
        ax.scatter(np.log10(dates), b_e, color='green', alpha=0.3,
                   label=r'$\epsilon = 1/3$')
        ax.scatter(np.log10(limdates), b_lim_e, marker='$\u2193$',
                   color='green', alpha=0.3)
    ax.legend()
    return fig


def plot_u_vs_t(df, dates, limdates, mask, eE, eB, name, color, marker,
                fits, df_equi=None, log=False):
    logging.info("Plotting U(t)")
    u, u_unc, u_lim = plotvals_from_df(df, 'u', mask, log=log)
    ax, fig = baseplot(
        np.log10(dates), u,
        r'{} U vs t $\epsilon_E$={} $\epsilon_B$={}'.format(name, eE, eB),
        'Time Since Explosion (days)', r'$\rho_{\rm{CSM}} v^2$ (erg)', new=True,
        yerror=u_unc, color=color, marker=marker, rtrn=True)
    logging.info("Plotting limits")
    ax.scatter(np.log10(limdates), u_lim, marker='$\u2191$',
               alpha=0.5, color=color)
    if df_equi is not None:
        u_e, _, u_lim_e = plotvals_from_df(df_equi, 'u', mask, log=log)
        ax.scatter(np.log10(dates), u_e, color='green', alpha=0.3,
                   label=r'$\epsilon = 1/3$')
        ax.scatter(np.log10(limdates), u_lim_e, marker='$\u2191$',
                   color='green', alpha=0.3)
    ax.legend()
    return fig


def plot_r_vs_t(df, dates, limdates, mask, eE, eB, name, color, marker,
                fits, df_equi=None, log=False):
    logging.info("Plotting R(t)")
    r, r_unc, r_lim = plotvals_from_df(df, 'r', mask, log=log)
    ax, fig = baseplot(
        np.log10(dates), r,
        r'{} R vs t $\epsilon_E$={} $\epsilon_B$={}'.format(name, eE, eB),
        'Time Since Explosion (days)', 'Radius of Shock (cm)', new=True,
        yerror=r_unc, color=color, marker=marker, rtrn=True)
    ax.scatter(np.log10(limdates), r_lim, marker='$\u2191$',
               alpha=0.5, color=color)
    plot_phys_param_fit(ax, fits['r_vs_t'])
    if df_equi is not None:
        r_e, _, r_lim_e = plotvals_from_df(df_equi, 'r', mask, log=log)
        ax.scatter(np.log10(dates), r_e, color='green', alpha=0.3,
                   label=r'$\epsilon = 1/3$')
        ax.scatter(np.log10(limdates), r_lim_e, marker='$\u2191$',
                   color='green', alpha=0.3)
    ax.legend(title='Slope Best Fit')
    return fig


def plot_b_vs_r(df, dates, limdates, mask, eE, eB, name, color, marker,
                fits, df_equi=None, log=False):
    logging.info("Plotting B(R)")
    r, r_unc, _ = plotvals_from_df(df, 'r', mask, log=log)
    b, b_unc, _ = plotvals_from_df(df, 'b', mask, log=log)
    ax, fig = baseplot(
        r, b,
        r'{} B vs R $\epsilon_E$={} $\epsilon_B$={}'.format(name, eE, eB),
        'Radius of Shock (cm)', 'Magnetic Field (Gauss)', new=True,
        yerror=b_unc, xerror=r_unc, color=color,
        marker=marker, rtrn=True)
    plot_phys_param_fit(ax, fits['b_vs_r'])
    if df_equi is not None:
        r_e, _, _ = plotvals_from_df(df_equi, 'r', mask, log=log)
        b_e, _, _ = plotvals_from_df(df_equi, 'b', mask, log=log)
        ax.scatter(r_e, b_e, color='green', alpha=0.3,
                   label=r'$\epsilon = 1/3$')
    ax.legend(title="Fit Params")
    return fig


def plot_den_vs_r(df, dates, limdates, mask, eE, eB, name, color, marker,
                  fits, df_equi=None, log=False):
    logging.info("Plotting rho(R)")
    r, r_unc, _ = plotvals_from_df(df, 'r', mask, log=log)
    den, den_unc, _ = plotvals_from_df(df, 'den', mask, log=log)
    ax, fig = baseplot(
        r, den,
        r'$\rho$ of CSM in {} $\epsilon_E$={} $\epsilon_B$={}'.format(
            name, eE, eB),
        'Radius of Shock (cm)', r'$\rho (g/cm^{3}$)', new=True,
        yerror=den_unc, xerror=r_unc, color=color,
        marker=marker, rtrn=True)
    plot_phys_param_fit(ax, fits['den_vs_r'])
    secax = ax.secondary_xaxis('top', functions=(r_to_t, t_to_r))
    secax.set_xlabel(r'Time Before Explosion [d], $v_w=1000 km/s$', fontsize=7)
    # draw lines of constant mass loss
    spaced_r = np.logspace(min(r), max(r))
    tpos = [(16.3890, -18.0649), (16.282, -18.3506), (16.1751, -18.6406),
            (16.07087, -18.93074), (15.9662, -19.7186), (16.0198, -19.3246)]
    pl = [-2, -2.5, -3, -3.5, -4.5, -4]
    for plidx, tp in zip(pl, tpos):
        yv = np.log10(const_m_loss(spaced_r, 10 ** plidx, 1000))
        tlbl = r'$10^{' + str(plidx) + r'}M_{\bigodot}/yr$'
        ax.plot(np.log10(spaced_r), yv, color='grey', alpha=0.4, ls='--')
        ax.text(tp[0], tp[1], tlbl, rotation=334, fontsize=7,
                rotation_mode='anchor', alpha=0.7, zorder=0)
    if df_equi is not None:
        r_e, _, _ = plotvals_from_df(df_equi, 'r', mask, log=log)
        den_e, _, _ = plotvals_from_df(df_equi, 'den', mask, log=log)
        ax.scatter(r_e, den_e, color='green', alpha=0.3,
                   label=r'$\epsilon = 1/3$')
    ax.legend(title="Slope", fontsize=6)
    return fig


def plot_ne_vs_t(df, dates, limdates, mask, eE, eB, name, color, marker,
                 fits, df_equi=None, log=False):
    logging.info("Plotting n_e(t)")
    ne, ne_unc, ne_lim = plotvals_from_df(df, 'n_e', mask, log=log)
    ax, fig = baseplot(
        np.log10(dates), ne,
        r'{} n_e vs t $\epsilon_E$={} $\epsilon_B$={}'.format(name, eE, eB),
        'Time Since Explosion (days)',
        r'No. Density of Electrons ($\rm{cm}^{-3}$)', new=True,
        yerror=ne_unc, color=color, marker=marker, rtrn=True)
    ax.scatter(np.log10(limdates), ne_lim, marker='$\u2193$',
               alpha=0.5, color=color)
    if df_equi is not None:
        ne_e, _, ne_lim_e = plotvals_from_df(df_equi, 'n_e', mask, log=log)
        ax.scatter(np.log10(dates), ne_e, color='green', alpha=0.3,
                   label=r'$\epsilon = 1/3$')
        ax.scatter(np.log10(limdates), ne_lim_e, marker='$\u2193$',
                   color='green', alpha=0.3)
    ax.legend()
    return fig


def plot_mv_vs_t(df, dates, limdates, mask, eE, eB, name, color, marker,
                 fits, df_equi=None, log=False):
    logging.info("Plotting Mv(t)")
    mv, mv_unc, mv_lim = plotvals_from_df(df, 'mv', mask, log=log)
    title = name + r' $\dot{M}/v$ vs t $\epsilon_E$='
    title += str(eE) + r'$\epsilon_B$=' + str(eB)
    ax, fig = baseplot(
        np.log10(dates), mv, title, 'Time Since Explosion (days)',
        r'$\dot{M}(10^{-4}*M_{\bigodot}*yr^{-1}) (v_w= 1000 km/s)$', new=True,
        yerror=mv_unc, color=color, marker=marker, rtrn=True)
    ax.scatter(np.log10(limdates), mv_lim, marker='$\u2193$',
               alpha=0.5, color=color)
    if df_equi is not None:
        mv_e, _, mv_lim_e = plotvals_from_df(df_equi, 'mv', mask, log=log)
        ax.scatter(np.log10(dates), mv_e, color='green', alpha=0.3,
                   label=r'$\epsilon = 1/3$')
        ax.scatter(np.log10(limdates), mv_lim_e, marker='$\u2193$',
                   color='green', alpha=0.3)
    ax.legend()
    return fig


def plot_v_vs_t(df, dates, limdates, mask, eE, eB, name, color, marker,
                fits, df_equi=None, log=False):
    logging.info("Plotting v(t)")
    v, v_unc, v_lim = plotvals_from_df(df, 'v', mask, log=log)
    vplot = np.power(10, v)
    vlimplot = np.power(10, v_lim)
    if log:
        # symmetric-in-log err converted to linear difference arrays
        v_err = abs(vplot - vplot * np.array([np.power(10, -v_unc),
                                              np.power(10, v_unc)]))
    else:
        # symmetric-in-linear err
        v_err = vplot * v_unc / np.log10(np.e)
    title = r'{} Velocity of Shock Front '
    title += r'$\epsilon_E$={} $\epsilon_B$={}'
    ax, fig = baseplot(
        np.log10(dates), vplot/speedLight, title.format(name, eE, eB),
        'Time Since Explosion (days)', 'Velocity (fraction of c)', new=True,
        yerror=v_err/speedLight, color=color, marker=marker, rtrn=True)
    ax.scatter(np.log10(limdates), vlimplot/speedLight, marker='$\u2191$',
               alpha=0.5, color=color)
    if df_equi is not None:
        v_e, _, v_lim_e = plotvals_from_df(df_equi, 'v', mask, log=log)
        v_e_plot = np.power(10, v_e)
        v_lim_e_plot = np.power(10, v_lim_e)
        ax.scatter(np.log10(dates), v_e_plot/speedLight, color='green',
                   alpha=0.3, label=r'$\epsilon = 1/3$')
        ax.scatter(np.log10(limdates), v_lim_e_plot/speedLight,
                   marker='$\u2191$', color='green', alpha=0.3)
    ax.legend()
    return fig


def plot_mv_vs_r(df, dates, limdates, mask, eE, eB, name, color, marker,
                 fits, df_equi=None, log=False):
    logging.info("Plotting Mv(R)")
    r, r_unc, _ = plotvals_from_df(df, 'r', mask, log=log)
    mv, mv_unc, _ = plotvals_from_df(df, 'mv', mask, log=log)
    title = name + r' $\dot{M}$ vs R ($v_w$= 1000 km/s) $\epsilon_E$='
    title += str(eE) + r' $\epsilon_B$=' + str(eB)
    ax, fig = baseplot(
        r, mv, title, 'Radius of Shock (cm)',
        r'$\dot{M}(10^{-4}M_{\bigodot}*yr^{-1}) (v_w= 1000 km/s)$', new=True,
        yerror=mv_unc, xerror=r_unc, color=color,
        marker=marker, rtrn=True)
    secax = ax.secondary_xaxis('top', functions=(r_to_t, t_to_r))
    secax.set_xlabel(r'Time Before Explosion [d], $v_w=1000 km/s$', fontsize=7)
    if df_equi is not None:
        r_e, _, _ = plotvals_from_df(df_equi, 'r', mask, log=log)
        mv_e, _, _ = plotvals_from_df(df_equi, 'mv', mask, log=log)
        ax.scatter(r_e, mv_e, color='green', alpha=0.3,
                   label=r'$\epsilon = 1/3$')
    ax.legend()
    return fig


def plot_mv2_vs_r(df, dates, limdates, mask, eE, eB, name, color, marker,
                  fits, log=False):
    logging.info("Plotting 1/Mv(R)")
    r, r_unc, _ = plotvals_from_df(df, 'r', mask, log=log)
    mv, mv_unc, _ = plotvals_from_df(df, 'mv', mask, log=log)
    logmv2 = -mv if log else np.log10(1. / mv)
    logr = r if log else np.log10(r)
    title = name + r' $v_w$ vs R  $\dot{M} = (10^{-4}*M_{\bigodot}*yr^{-1}) '
    title += r'\epsilon_E$={} $\epsilon_B$={}'.format(eE, eB)
    ax, fig = baseplot(
        logr, logmv2, title, 'Radius of Shock (cm)',
        r'$v_w$ (1000 km/s) $\dot{M} = 10^{-4}*M_{\bigodot}*yr^{-1}$',
        new=True, yerror=mv_unc, xerror=r_unc, color=color,
        marker=marker, rtrn=True)
    max_vw = max(np.power(10, logmv2)) #km/s
    min_vw = min(np.power(10, logmv2)) #km/s
    max_r = max(logr) #cm
    min_r = min(logr) #cm
    grav_const = 6.67e-8 #cgs
    min_mr = min_vw * 1e5 / np.sqrt(2 * grav_const)
    max_mr = max_vw * 1e5 / np.sqrt(2 * grav_const)
    ax.axhline(np.log10(min_vw), label='{:.2f}'.format(np.log10(min_vw)),
               alpha=0.5, linestyle='--')
    ax.axhline(np.log10(max_vw), label='{:.2f}'.format(np.log10(max_vw)),
               alpha=0.5, linestyle='--')
    str1 = r'$v_w =${:.2f} km/s, '.format(max_vw*1000)
    str1 += r'$\sqrt{M/R} =$'+'{:.2f}E7'.format(max_mr/1e7)+r'$(g/cm)^{1/2}$'
    str2 = r'$v_w =${:.2f} km/s, '.format(min_vw*1000)
    str2 += r'$\sqrt{M/R} =$'+'{:.2f}E7'.format(min_mr/1e7)+r'$(g/cm)^{1/2}$'
    ax.text(min_r * 1.01, np.log10(max_vw) * .995, str1, fontsize=7,
            color='#004C99')
    ax.text(max_r * .98, np.log10(min_vw) * .995, str2, fontsize=7,
            color='#004C99')
    ax.legend()
    return fig


def plot_param_vs_t(x, y, yerr, xlim, ylim, title, xlabel, ylabel, fname,
                    color, pfit=None, equi=None):
    ax, fig = baseplot(x, y, title, xlabel, ylabel, new=True, yerror=yerr,
                     color=color, marker=marker, rtrn=True)
    ax.scatter(xlim, ylim, marker='$\u2193$', alpha=0.5, color=color)
    if pfit is not None:
        plot_phys_param_fit(ax, pfit)
    if equi is not None:
        ax.scatter(x, equi['y'], color='green', alpha=0.3,
                   label=r'$\epsilon = 1/3$')
        ax.scatter(xlim, equi['ylim'], marker='$\u2193$',
                   color='green', alpha=0.3)
    ax.legend()
    fig.savefig(fname, bbox_inches='tight')
    plt.close(fig)
    return


def plotvals_from_df(df, par, mask, log=False):
    y_raw = np.array([w.n for w in df.loc[mask, par]])
    y_unc_raw = np.array([w.std_dev for w in df.loc[mask, par]])
    y_lim_raw = np.array([w for w in df.loc[~mask, par]])
    y = y_raw if log else np.log10(y_raw)
    y_unc = y_unc_raw if log else np.log10(np.e) * y_unc_raw / y_raw
    y_lim = y_lim_raw if log else np.log10(y_lim_raw)
    return y, y_unc, y_lim


def plot_phys_param_fit(ax, fitdict):
    c = 'teal' if 'y_unc' in fitdict else 'black'
    alp = 0.5 if 'y_unc' in fitdict else 1.0
    lbl = fitdict['lbl'] if 'lbl' in fitdict else None
    ax.plot(fitdict['x'], fitdict['y'], color=c, alpha=alp, label=lbl)
    if 'y_unc' in fitdict:
        y_hi = fitdict['y'] + fitdict['y_unc']
        y_lo = fitdict['y'] - fitdict['y_unc']
        ax.plot(fitdict['x'], y_hi, color=c, ls='--', alpha=0.3)
        ax.plot(fitdict['x'], y_lo, color=c, ls='--', alpha=0.3)
    return


def propagate_bpl_test(ax, data, fitter, desired_time, output_dir,
                       theta_nom=np.pi/2, eE=0.1, eB=0.01, fill_nom=1.0,
                       comparative_case='All_Params_Vary'):
    """Propagates the broken powerlaw to a given time (in days) and plots
    the result to the supplied axis.
    """
    style = {'All_Params_Vary': ['-', 'D', 'black'],
             'Hold_alpha2_2.5': [':', 's', '#00495d'],
             'Hold_s_fixed': ['-.', 'p', '#c79b5e'],
             'Hold_alpha2_and_s': ['--', '^', 'maroon']}
    ls, ms, ccc = style[comparative_case]

    outpath = os.path.join(output_dir, data.name, 'Projection',
                            comparative_case)
    os.makedirs(outpath, exist_ok=True)

    res = fitter.result
    df_phys = res.phys_data
    alpha1 = ufloat(res.disp.params['alpha1'].value,
                    res.disp.params['alpha1'].stderr)
    alpha2 = res.linked_variables['alpha2']
    s = res.linked_variables['s']
    p = res.linked_variables['alpha1']
    theta = ufloat(theta_nom, 0)
    epsilonE = ufloat(epsilonE, 0)
    epsilonB = ufloat(epsilonB, 0)
    filling = ufloat(fill_nom, 0)
    energyLower = ufloat(energyLower_nom, 0)
    distance = ufloat(data.distance, 0)

    mask = df_phys['limit'].to_numpy() != 'limit'
    b_ufloat = np.array([w for w in df_phys.loc[mask, 'b']])
    r_ufloat = np.array([w for w in df_phys.loc[mask, 'r']])
    dates = df_phys.loc[mask, 'date'].to_numpy()

    prop_b, prop_r = wrapped_prop_b_and_r(b_ufloat, r_ufloat, dates,
                                          desired_time, name, comparative_case)

    prop_flux = wrapped_flux(prop_b, prop_r, theta,  p, epsilonE, epsilonB,
                             filling, energyLower, distance)
    prop_nu = wrapped_nu(prop_b, prop_r, theta,  p, epsilonE, epsilonB, filling,
                         energyLower, distance)
    # scale to display units
    prop_flux *= 1e6
    prop_nu *= 5

    fff = np.log10(np.arange(prop_nu.n/10, prop_nu.n*10, prop_nu.n/10))
    f = [ufloat(ff, 0) for ff in fff]

    prop_bpl = [wrapped_asym(freq, prop_flux, prop_nu, alpha1, alpha2, s) for
                freq in f]

    prop_bpl_num = [num.n for num in prop_bpl]
    prop_bpl_unc = [num.s for num in prop_bpl]

    ax.plot(fff, prop_bpl_num,  alpha=0.3, color=ccc) #marker='*',
    ax.scatter(np.log10(prop_nu.n), np.log10(prop_flux.n), marker=ms,
               alpha=0.3, color=ccc)

    l1 = limit1(fff, alpha1.n, alpha2.n, s.n, prop_flux.n, prop_nu.n)
    l2 = limit2(fff, prop_nu.n, prop_flux.n, alpha1.n, alpha2.n, s.n)
    ax.plot(fff, l1, alpha=0.1, linestyle='--', label='f>>f0', color='grey')
    ax.plot(fff, l2, alpha=0.1, linestyle='--', label='f<<f0', color='grey')

    # write projected values to file
    outfile = '{}/projection_at_{}_days.txt'.format(outpath, desired_time)
    cols = ['projected time (days)', 'projected b (Gauss)', 'projected r (cm)',
            'projected frequency (GHz)', 'projected flux (microJy)']
    with open(outfile, 'a') as fp:
        fp.write(', '.join(cols) + '\n')
        fp.write('{}, {}, {}, {}, {} '.format(
            desired_time, prop_b, prop_r, prop_nu, prop_flux)+'\n')

    return ax


