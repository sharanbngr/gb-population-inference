from matplotlib import pyplot as plt
import numpy as np
import scipy
import pandas as pd
import os, sys
from tqdm import tqdm
from utils import get_rfft
from lisatools.sensitivity import AET1SensitivityMatrix
import lisatools.detector as lisa_models
from noise import generate_noise_realization
from datageneration.datageneration.population_model import (
    PowerLawChirpPowerLawSeperation,
)
from datageneration.datageneration.distributions import TruncatedPowerLaw

from sorting import vector_sorting, iterative_sorting
from dynesty import NestedSampler


def get_dwd_pop_strains(
    Mc_powerlaw_index, r_powerlaw_index, n_dwd, duration, sampling_duration
):

    limits = {
        "chirp_mass": [0.5, 1.1],  # In solar masses
        #           'seperation':          [0.5, 25.0],   # In 1e8 meters
        "seperation": [5.0, 10.0],  # In 1e8 meters
        "luminosity_distance": [1.0, 50.0],  # In kilo parsecs
        "phase": [0, 2 * np.pi],
    }

    ## Instantiate the population distribution object
    dist = PowerLawChirpPowerLawSeperation(
        limits=limits,
        distance_power_law_index=1,  # p(d) ~ d
        N_white_dwarfs=n_dwd,
        duration=duration,
        sample_rate=1 / sampling_duration,
        poisson=True,
    )  # Total Number of sources should be drawn from poisson distribution

    ## alpha and beta are the power-law parameters for the chirpmass and orbital seperation distribution.
    ## there are the key inferred parameters.
    Lambda = {"alpha": Mc_powerlaw_index, "beta": r_powerlaw_index}
    # popdict = dist.waveform.compute_waveform_parameters(dist.generate_samples(Lambda, size=N_dwd))
    poptimes, popstrains = dist.generate_time_series(Lambda, summed=False)

    return dist, poptimes, popstrains


def generate_one_detector_data(
    n_dwd=1e4,
    duration=int(1e5),
    R_lisa=1e-3,
    sampling_duration=4,
    Mc_powerlaw_index=4.0,
    r_powerlaw_index=-2.0,
):

    ## generate strain noise for all three channels
    noise_dict = generate_noise_realization(
        sampling_duration, duration  # dt (in s)  # 30*24*3600 # Tobs (in s)
    )

    dist, poptimes, popstrains = get_dwd_pop_strains(
        Mc_powerlaw_index, r_powerlaw_index, n_dwd, duration, sampling_duration
    )

    injected_gw_parameters = pd.DataFrame(dist.samples_from_population)

    ## hacky but works for the toy model
    ## setting arbitrary (flat) response functions
    popstrains_in_detector = R_lisa * popstrains

    data_dict = {
        "t": poptimes,
        "time_domain_data": noise_dict["timeseries"][0, :]
        + popstrains_in_detector.sum(axis=1),
    }

    data_dict["freqs"], data_dict["fft_data"] = get_rfft(
        data_dict["time_domain_data"], data_dict["t"], 1.0 / sampling_duration
    )

    _, data_dict["fft_noise"] = get_rfft(
        noise_dict["timeseries"][0, :],
        data_dict["t"],
        1.0 / sampling_duration,
    )

    _, data_dict["total_gw_spectrum"] = get_rfft(
        popstrains_in_detector.sum(axis=1),
        data_dict["t"],
        1.0 / sampling_duration,
    )

    fs_welch, psd_data_welch = scipy.signal.welch(
        data_dict["time_domain_data"],
        fs=1 / sampling_duration,
        window="hann",
        noverlap=0.0,
        nperseg=256 * 8,
    )

    sens_mat = AET1SensitivityMatrix(
        np.array(data_dict["freqs"]).astype("float"),
        model=lisa_models.sangria,
    )
    SnA, SnE, SnT = sens_mat.sens_mat
    data_dict["theoretical_noise_PSD"] = SnA

    plt.figure()
    plt.loglog(
        data_dict["freqs"],
        2 * np.abs(data_dict["fft_noise"]) ** 2 / duration,
        label="noise PSD",
        alpha=0.5,
        lw=2.0,
    )

    plt.loglog(fs_welch, psd_data_welch, label="Data Welch PSD", lw=1.0)

    plt.loglog(
        data_dict["freqs"],
        data_dict["theoretical_noise_PSD"],
        label="Theoretical noise PSD data",
        lw=1.0,
    )

    plt.loglog(
        data_dict["freqs"],
        2 * np.abs(data_dict["total_gw_spectrum"] ** 2) / duration,
        label="GW spectrum",
        lw=1.0,
    )
    plt.legend()
    plt.xlim(1e-4, 1e-2)
    plt.ylim(1e-46, 1e-38)
    plt.savefig("./inference_tests/spectra.png", dpi=250)
    plt.close()

    fg, N_res, res_idx, unres_idx = vector_sorting(
        injected_gw_parameters,
        data_dict["freqs"],
        data_dict["theoretical_noise_PSD"],
        duration,
        R_lisa,
        wts=np.ones(injected_gw_parameters.shape[0]),
        snr_thresh=7,
    )

    ## we're not going to sample over resolved binaries in this toy model. Keeping them fixed.
    resolved = {}
    resolved["N_resolved"] = N_res
    _, resolved["ffts_resolved"] = get_rfft(
        popstrains_in_detector[:, res_idx], data_dict["t"], 1.0 / sampling_duration
    )
    resolved["dwd_parameters"] = injected_gw_parameters.iloc[res_idx]

    resolved["scaled_PSD_resolved"] = (
        0.5 * duration * R_lisa * (resolved["dwd_parameters"]["amplitude"]) ** 2
    )

    _, unresolved_fft = get_rfft(
        popstrains_in_detector[:, unres_idx].sum(axis=1),
        data_dict["t"],
        1.0 / sampling_duration,
    )

    data_dict["true_foreground"] = 2 * np.abs(unresolved_fft) ** 2 / duration
    # data_dict["resolved"] = resolved

    plt.loglog(
        data_dict["freqs"],
        data_dict["theoretical_noise_PSD"],
        c="k",
        label="Theoretical Noise PSD",
    )
    plt.loglog(
        data_dict["freqs"],
        data_dict["true_foreground"],
        c="orchid",
        label="Unresolved Foreground",
    )
    plt.scatter(
        resolved["dwd_parameters"]["frequency"],
        resolved["scaled_PSD_resolved"],
        marker=".",
        s=10,
        c="k",
        label="Resolved Binaries (amp)",
        zorder=10,
    )
    # plt.scatter(fs_resolved,psds_resolved,marker='*',s=10,c='maroon',label='Resolved Binaries (fft)')
    plt.ylim(1e-46, 1e-38)
    plt.legend()
    plt.savefig("./inference_tests/resolved_unresolved.png", dpi=250)
    plt.close()

    return data_dict, injected_gw_parameters, resolved


class pop_inference:

    def __init__(self, true_vals, data_dict, resolved, duration, R_lisa):
        self.data_dict = data_dict
        self.resolved = resolved
        self.true_vals = true_vals
        self.duration = duration
        self.R_lisa = R_lisa
        self.sampling_duration = 4

        self.limits = {
            "chirp_mass": [0.5, 1.1],  # In solar masses
            #           'seperation':          [0.5, 25.0],   # In 1e8 meters
            "seperation": [5.0, 10.0],  # In 1e8 meters
            "luminosity_distance": [1.0, 50.0],  # In kilo parsecs
            "phase": [0, 2 * np.pi],
        }

        ## used for vector sorting
        self.calc_fiducial_population()

        print("Setting up the Hierarchical bayesian Inference Framework.")

    def calc_fiducial_population(self):

        self.fiducial_Mc_powerlaw_index = 2.0
        self.fiducial_r_powerlaw_index = -1.1
        n_dwd = 1e4

        fiducial_dist, poptimes, popstrains = get_dwd_pop_strains(
            self.fiducial_Mc_powerlaw_index,
            self.fiducial_r_powerlaw_index,
            n_dwd,
            self.duration,
            self.sampling_duration,
        )

        self.fiducial_population = pd.DataFrame(fiducial_dist.samples_from_population)

        ## hacky but works for the toy model
        ## setting arbitrary (flat) response functions
        self.fiducial_popstrains_in_detector = self.R_lisa * popstrains
        _, self.fiducial_popstrains_ffts = get_rfft(
        self.fiducial_popstrains_in_detector , self.data_dict["t"], 1.0 / self.sampling_duration
    )

        self.fiducial_popstrains_psds = 2 * np.abs(self.fiducial_popstrains_ffts)**2 / self.duration

        self._fiducial_chirpmass_logpdf, self._fiducial_seperation_logpdf = (
            self.calc_population_weights(
                self.fiducial_Mc_powerlaw_index, self.fiducial_r_powerlaw_index
            )
        )

        return

    def calc_population_weights(self, Mc_powerlaw_index, r_powerlaw_index):

        chirp_mass = TruncatedPowerLaw(Mc_powerlaw_index, *self.limits["chirp_mass"])

        seperation = TruncatedPowerLaw(r_powerlaw_index, *self.limits["seperation"])

        chirpmass_logpdf = chirp_mass.logpdf(self.fiducial_population["chirp_mass"])
        seperation_logpdf = seperation.logpdf(self.fiducial_population["seperation"])

        return chirpmass_logpdf, seperation_logpdf

    def unresolved_foreground(self, wts, unres_idx):

        unresolved_psds = self.fiducial_popstrains_psds[:, unres_idx]

        wts_unresolved = wts.to_numpy()[unres_idx]

        S_unresolved = np.mean(unresolved_psds * wts_unresolved, axis=1)

        return S_unresolved
        # import pdb; pdb.set_trace()

    def prior(self, theta):

        theta[0] = 100 + 5e3 * theta[0]  ## n_tot
        theta[1] = -10 + 20 * theta[1]  ## Mc powerlaw
        theta[2] = -10 + 20 * theta[2]  ## seperation powerlaw

        return theta

    def log_likelihood(self, theta):

        N_tot, Mc_powerlaw_index, r_powerlaw_index = theta[0], theta[1], theta[2]
        chirpmass_logpdf, seperation_logpdf = self.calc_population_weights(
            Mc_powerlaw_index, r_powerlaw_index
        )

        chirpmass_log_wts = chirpmass_logpdf - self._fiducial_chirpmass_logpdf
        seperation_log_wts = seperation_logpdf - self._fiducial_seperation_logpdf

        wts = np.exp(chirpmass_log_wts + seperation_log_wts)
        
        #fg, N_res, res_idx, unres_idx = vector_sorting(
        _, _, res_idx, unres_idx = vector_sorting(
            self.fiducial_population,
            self.data_dict["freqs"],
            self.data_dict["theoretical_noise_PSD"],
            self.duration,
            self.R_lisa,
            wts=wts,
            snr_thresh=7,
        )

        f_resolved = np.sum(wts.to_numpy()[res_idx])

        S_unresolved = self.unresolved_foreground(wts, unres_idx)

        ## total noise : astrophysical + instrumental
        C_total = self.data_dict["theoretical_noise_PSD"] + S_unresolved

        residuals = self.data_dict["fft_data"] - np.sum(
            self.resolved["ffts_resolved"], axis=1
        )

        log_poisson_term = - N_tot * f_resolved + self.resolved["N_resolved"] * np.log(N_tot * f_resolved)

        log_likelihood = (-2 / self.duration) * np.sum(
            np.abs(residuals) ** 2 / C_total
        ) - np.sum(np.log(2 * self.duration * C_total))


        return log_likelihood + log_poisson_term

def toy_model_inference():

    true_vals = {
        "n_dwd": int(1e3),
        "Mc_powerlaw_index": 4.0,
        "r_powerlaw_index": -2.0,
    }

    n_dwd_true = int(1e3)
    duration = int(1e5)
    R_lisa = 1e-3

    data_dict, injected_gw_parameters, resolved = generate_one_detector_data(
        n_dwd=true_vals["n_dwd"],
        duration=duration,
        R_lisa=R_lisa,
        Mc_powerlaw_index=true_vals["Mc_powerlaw_index"],
        r_powerlaw_index=true_vals["r_powerlaw_index"],
    )

    dwd_inference = pop_inference(true_vals, data_dict, resolved, duration, R_lisa)

    engine = NestedSampler(
        dwd_inference.log_likelihood,
        dwd_inference.prior,
        ndim=3,
        bound="multi",
        sample="auto",
        nlive=250,
    )


if __name__ == "__main__":

    toy_model_inference()
