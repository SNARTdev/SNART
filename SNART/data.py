
import os
import logging
from lmfit import fit_report
import numpy as np
import pandas as pd


class SNData(object):
    def __init__(self, name, filename, explosion_date, percentage,
                 distance):

        self.name = name
        self.distance = distance

        count, df0, df_upper = self.read_data(filename, explosion_date)
        self.count = count
        self.df_upper = df_upper
        self.df = self.group_SEDs(percentage, df0)
        # store obs units (Jy and 5GHz) version of data
        self.df_obs = self.df.copy(deep=True)
        self.df_obs.loc[:, "Flux Density"] *= 1e-6
        self.df_obs.loc[:, "Uncertainty"] *= 1e-6
        self.df_obs.loc[:, "Frequency"] *= 0.2

        self.sed_pre_list = np.arange(max(self.df['SED Number']) + 1)
        self.ribbons = self.cut_dataframe(self.df, self.sed_pre_list)
        self.ribbons_obs = self.cut_dataframe(self.df_obs, self.sed_pre_list)

    def read_data(self, filename, explosion_date):

        df = pd.read_csv(filename)
        if "Name" in df.columns:
            df = df.loc[(df["Name"]==self.name)]
        else:
            logging.info("No 'Name' column found in datafile. "
                         "Will assume all datapoints belong to %s", self.name)
        df.sort_values("Date")

        df.loc[:, "Date"] -= explosion_date
        df_upper = df.loc[(df["Upper Limit"] == '*upper limit')]
        df = df.loc[(df["Upper Limit"] != '*upper limit')]

        count = len(np.array(df["Flux Density"]))

        if len(np.array(df.loc[:,"Date"])) == 0:
            logging.info("WARNING: this supernova data is either empty or "
                         "only has upper limits available.")

        return count, df, df_upper

    def group_SEDs(self, percentage, df):
        # assign label 0 to all SEDs
        df.insert(0, "SED Number", 0)

        # calculate date range of first group
        d0 = df.iloc[0]["Date"]
        first_end = d0 * (1. + percentage)

        df.loc[(df["Date"] <= first_end), "SED Number"] = 0

        # loop through data and assign SED numbers
        prev_end = first_end
        sed_number = 1
        while (np.array(df["SED Number"])[-1] == 0):
            this_start = np.array(df.loc[df["Date"] > prev_end, "Date"])[0]
            this_end = this_start * (1. + percentage)

            group_mask = (df["Date"] > prev_end) & (df["Date"] <= this_end)
            df.loc[group_mask, "SED Number"] = sed_number

            prev_end = this_end
            sed_number += 1

        df = df.sort_values(by=["SED Number", "Frequency"])

        return df

    def cut_dataframe(self, df, sed_pre_list):
        ribbons = {sed: SED(df.loc[df["SED Number"] == sed], sed) for sed
                   in sed_pre_list}
        return ribbons

    @classmethod
    def from_config(cls, cp):
        filename = cp.get("data", "datafile")
        percentage = float(cp.get("data", "percentage"))
        name = cp.get("event", "name")
        explosion_date = float(cp.get("event", "explosion-date"))
        distance = float(cp.get("event", "distance"))
        return cls(name, filename, explosion_date, percentage, distance)


class SED(object):
    def __init__(self, data, number):
        self.number = number
        self.chi_flux = data["Flux Density"].to_numpy()
        self.chi_flux_log = np.log10(self.chi_flux)
        self.chi_freq = data["Frequency"].to_numpy()
        self.chi_freq_log = np.log10(self.chi_freq)
        self.chi_unc = data["Uncertainty"].to_numpy()
        self.chi_unc_log = np.log10(np.e) * self.chi_unc / self.chi_flux


def write_output(data, fit, fit_disp, opts):
    df_phys = fit.result.phys_data
    df_phys_equi = fit.result.phys_data_equi
    outroot = os.path.join(opts.output_dir, data.name)
    os.makedirs(outroot, exist_ok=True)

    # write initial dataframe (obs units)
    data.df_obs.to_csv('{}/initial_dataframe.csv'.format(outroot), header=True,
                       index=None, sep=',', mode='w')

    # write fit results
    with open('{}/fit_result_obs.txt'.format(outroot), 'w') as fp:
        fp.write(fit_report(fit.result.obs))
    #with open('{}/fit_result_disp.txt'.format(outroot), 'w') as fp:
    #    fp.write(fit_report(fit_disp.res.res_lmfit))

    # write physical param datarame
    phys_param_dir = os.path.join(outroot, 'Physical_Params')
    os.makedirs(phys_param_dir, exist_ok=True)
    df_phys.to_csv('{}/dataframe.csv'.format(phys_param_dir), header=True,
                   index=None, sep=',', mode='w', float_format='%.15f')

    if df_phys_equi is not None:
        df_phys_equi.to_csv('{}/dataframe_equi.csv'.format(phys_param_dir),
                            header=True, index=True, sep=',', mode='w')
    return
