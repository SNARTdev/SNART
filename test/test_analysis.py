
import unittest
import numpy as np
import pandas as pd
from uncertainties import ufloat, UFloat
from SNART.data import SNData
from SNART.fitting import SNFit
from SNART.synchrotron import create_phys_param_df


class TestAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(cls):

         event = "2004C"
         datafile = "examples/SN2004C_radio_data.csv"
         explosion_date = 52988.0
         percentage = 0.025
         distance = 35.9

         # read data
         sndata = SNData(
             event,
             datafile,
             explosion_date,
             percentage,
             distance
         )

         # load fit models
         snfit_log = SNFit(
             sndata.df_obs,
             sndata.sed_pre_list,
             sndata.ribbons_obs,
             impostors=[3, 6, 11, 13, 14, 15],
             s_guess=-1.0,
             s_vary=False,
             two_slope=True,
             asym_lim=True,
             log=True
         )
         snfit_lin = SNFit(
             sndata.df_obs,
             sndata.sed_pre_list,
             sndata.ribbons_obs,
             impostors=[3, 6, 11, 13, 14, 15],
             s_guess=-1.0,
             s_vary=False,
             two_slope=True,
             asym_lim=True,
             log=False
         )

         # calculate physical params
         class Dummy(object):
             def __init__(self, **kwargs):
                 for k, v in kwargs.items():
                     setattr(self, k, v)

         nsargs = dict(epsilonE=0.1, epsilonB=0.01, equipartition=False,
                       i_nom=1.0, fill_nom=1.0, theta_nom=np.pi/2)
         opts = Dummy(**nsargs)
         cls.df_phys_log, _ = create_phys_param_df(sndata, snfit_log, opts)
         cls.df_phys_lin, _ = create_phys_param_df(sndata, snfit_lin, opts)

         # reference results
         cls.log_b_ref = [
             ufloat(-0.248104872830893, 0.039153908560385),
             ufloat(-0.266368641534179, 0.032399782643050),
             ufloat(-0.265125599420022, 0.019293837751709),
             ufloat(-0.281113176070864, 0.020755673220840),
             ufloat(-0.316393081045657, 0.020129124674197),
             ufloat(-0.339472344924969, 0.027677318756883),
             ufloat(-0.322092510260937, 0.018548725071132),
             ufloat(-0.324373377016308, 0.019597703774973),
             ufloat(-0.345224247723348, 0.017390688865633),
             ufloat(-0.391984531840485, 0.018397997612203),
             ufloat(-0.400697390394187, 0.024896516990229),
             ufloat(-0.499324694687468, 0.021454761068439),
             ufloat(-0.594449305704003, 0.042883449959653),
             ufloat(-0.512805003265496, 0.060289950528689),
             ufloat(-0.660072983399171, 0.050932017781460),
             ufloat(-0.607178641692782, 0.052828505476379),
             -0.604872578187384,
             -0.607798865935830,
             -0.596559868605961,
             -0.600092046514581,
             -0.600381395784783,
             -0.592732630096400,
             -0.609538937712486,
             -0.580106065901882,
             -0.551380482437553,
             -0.530435526765199,
             -0.524484425646357,
             -0.519678306790397
         ]

         cls.log_mv_ref = [
             ufloat(0.726691367542355, 0.094513841970929),
             ufloat(0.942345532028362, 0.083664422496636),
             ufloat(1.290293876753516, 0.065496264449291),
             ufloat(1.374104948762966, 0.067260269397732),
             ufloat(1.366351339824263, 0.066494199721869),
             ufloat(1.424088845443904, 0.076582555783511),
             ufloat(1.503880886900106, 0.064629503812981),
             ufloat(1.541236298457669, 0.065856145719748),
             ufloat(1.591366383067726, 0.063328478530115),
             ufloat(1.557945788502078, 0.064456940825708),
             ufloat(1.633560596354130, 0.072664284136248),
             ufloat(1.495795847951907, 0.068132070613401),
             ufloat(1.384658637279355, 0.100780519336937),
             ufloat(1.573998355225665, 0.131682438066211),
             ufloat(1.372909721623696, 0.114791261414875),
             ufloat(1.541856417782085, 0.118170029700958),
             1.688147450020054,
             1.740524202135241,
             1.867854160797453,
             2.027256785200578,
             2.226424657525641,
             2.382302209393682,
             2.461791937490307,
             2.649097534964270,
             3.086739710455983,
             3.206784349713267,
             3.443241509964199,
             3.524140179882224
         ]

         cls.log_den_ref = [
             ufloat(-18.354034944800667, 0.150191152776077),
             ufloat(-18.217888788841247, 0.124736323605221),
             ufloat(-18.178876604659642, 0.079268636161330),
             ufloat(-18.207928650810167, 0.083035403485712),
             ufloat(-18.314438031431830, 0.083172251562230),
             ufloat(-18.352802369934913, 0.112178918796387),
             ufloat(-18.220457434645549, 0.079157043359668),
             ufloat(-18.204110360264455, 0.081154064577341),
             ufloat(-18.181255491766088, 0.077598615742096),
             ufloat(-18.334753023737562, 0.080832446868927),
             ufloat(-18.219917959099874, 0.110258910265679),
             ufloat(-18.525482224696891, 0.101229761700026),
             ufloat(-18.876029861311594, 0.179487444472186),
             ufloat(-18.462479917683414, 0.268730407135145),
             ufloat(-18.984185019855602, 0.224703369016982),
             ufloat(-18.673520610367170, 0.228767865281596)
         ]

         cls.lin_b_ref = [
             ufloat(0.564799764954239, 0.050922056810523),
             ufloat(0.541539940547942, 0.040401618611097),
             ufloat(0.543092324976204, 0.024127495123933),
             ufloat(0.523463220505293, 0.025017651181287),
             ufloat(0.482621010600793, 0.022369415354791),
             ufloat(0.457643068825770, 0.029166083013083),
             ufloat(0.476328659142081, 0.020344195866051),
             ufloat(0.473833807392327, 0.021382526831510),
             ufloat(0.451621267470142, 0.018084459673319),
             ufloat(0.405522309445225, 0.017179369010061),
             ufloat(0.397467685084848, 0.022785951613855),
             ufloat(0.316719341078126, 0.015646854326327),
             ufloat(0.254419255901164, 0.025122647755518),
             ufloat(0.307039313510074, 0.042626790655556),
             ufloat(0.218739072381131, 0.025654018834319),
             ufloat(0.247070503890978, 0.030056014065575),
             0.248385866529470,
             0.246717891550205,
             0.253185972645220,
             0.251135121378873,
             0.250967855231882,
             0.255427028667037,
             0.245731337920854,
             0.262962270458553,
             0.280943519771813,
             0.294824748410959,
             0.298892476738158,
             0.302218611434395
         ]

         cls.lin_mv_ref = [
             ufloat(5.329538457695800, 1.159897652627085),
             ufloat(8.756756503360734, 1.686988856963199),
             ufloat(19.511554838944537, 2.942629607200357),
             ufloat(23.664815287626183, 3.665134374070990),
             ufloat(23.246063721120649, 3.559274388557353),
             ufloat(26.551360921699214, 4.682153828925757),
             ufloat(31.906473413393854, 4.748286836814168),
             ufloat(34.772395664744558, 5.273036894966847),
             ufloat(39.026816070073117, 5.690994783163813),
             ufloat(36.136312046473797, 5.363417166508900),
             ufloat(43.008916382839772, 7.196287838570371),
             ufloat(31.317989944694784, 4.913326309640964),
             ufloat(24.246925467034067, 5.626786703879493),
             ufloat(37.496938208355530, 11.370127652044742),
             ufloat(23.599776666384159, 6.238114911982417),
             ufloat(34.822101387408864, 9.475502494728113),
             48.769223358344007,
             55.020267357523615,
             73.765391525279753,
             106.476865402391738,
             168.431423885634217,
             241.157428829008438,
             289.594544361121734,
             445.754792407099046,
             1221.063321924848424,
             1609.840129189099116,
             2774.851852655718631,
             3343.017713241176807
         ]

         cls.lin_den_ref = [
             ufloat(4.425514080980839e-19, 1.530548725307056e-19),
             ufloat(6.054928602994742e-19, 1.739131217425525e-19),
             ufloat(6.624020711434786e-19, 1.209077525068613e-19),
             ufloat(6.195408058966797e-19, 1.184591964604963e-19),
             ufloat(4.847975225922444e-19, 0.928484459964303e-19),
             ufloat(4.438086631559281e-19, 1.146412955489977e-19),
             ufloat(6.019226820712322e-19, 1.097143414250483e-19),
             ufloat(6.250121357161023e-19, 1.167999550502351e-19),
             ufloat(6.587802703517490e-19, 1.177118194434091e-19),
             ufloat(4.626422920698853e-19, 0.861126681165852e-19),
             ufloat(6.026707951501440e-19, 1.530120305770251e-19),
             ufloat(2.982058237638395e-19, 0.695121243056611e-19),
             ufloat(1.330357901320075e-19, 0.549832230153648e-19),
             ufloat(3.447602862997153e-19, 2.133424583179598e-19),
             ufloat(1.037083291637758e-19, 0.536611728241115e-19),
             ufloat(2.120698098457312e-19, 1.117177794870962e-19)
         ]

    def test_log_b(self):
        logb = self.df_phys_log['b'].tolist()
        return self.compare_results(self.log_b_ref, logb)

    def test_log_mv(self):
        logmv = self.df_phys_log['mv'].tolist()
        return self.compare_results(self.log_mv_ref, logmv)

    def test_log_den(self):
        # Trim NaNs at end. Find a better way to handle this
        logden = self.df_phys_log['den'].tolist()[:len(self.log_den_ref)]
        return self.compare_results(self.log_den_ref, logden)

    def test_lin_b(self):
        linb = self.df_phys_lin['b'].tolist()
        return self.compare_results(self.lin_b_ref, linb, delta=5e-4)

    def test_lin_mv(self):
        linmv = self.df_phys_lin['mv'].tolist()
        return self.compare_results(self.lin_mv_ref, linmv, delta=5e-4)

    def test_lin_den(self):
        # Trim NaNs at end. Find a better way to handle this
        linden = self.df_phys_lin['den'].tolist()[:len(self.lin_den_ref)]
        return self.compare_results(self.lin_den_ref, linden, delta=5e-4)

    def compare_results(self, pref, ptest, delta=1e-6):
        compresult = []
        for pr, pt in zip(pref, ptest):
            if isinstance(pr, UFloat):
                compresult.append(self.compare_ufloats(pr, pt, delta=delta))
            else:
                compresult.append(self.assertAlmostEqual(pr, pt, delta=delta))
        return all(compresult)

    def compare_ufloats(self, u1, u2, delta=1e-6):
        assert isinstance(u1, UFloat)
        assert isinstance(u2, UFloat)
        nmatch = self.assertAlmostEqual(u1.n, u2.n, delta=delta)
        smatch = self.assertAlmostEqual(u1.s, u2.s, delta=delta)
        return nmatch and smatch

if __name__ == "__main__":
    unittest.main()
