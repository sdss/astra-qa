
from IPython.display import display, Markdown
from collections import OrderedDict
from peewee import BitField
from tabulate import tabulate
from astropy.table import Table
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import binned_statistic_2d
from matplotlib.colors import LogNorm
from IPython.display import display, Markdown
from tabulate import tabulate
from peewee import fn, JOIN, Case
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
from astropy.time import Time
from matplotlib.ticker import MaxNLocator
import numpy as np
import warnings
from astra import __version__
from astra.models import (
    ApogeeVisitSpectrum,
    ApogeeVisitSpectrumInApStar,
    ApogeeCoaddedSpectrumInApStar,
    ApogeeMADGICSVisitSpectrum,
    BossVisitSpectrum,
    BossCombinedSpectrum,
    BossRestFrameVisitSpectrum,
    ApogeeCombinedSpectrum,
    ApogeeRestFrameVisitSpectrum,
    Spectrum,
    Source
)
from collections import OrderedDict
from peewee import BitField
import pickle
from astra.utils import expand_path
    
DEFAULT_SPECTRUM_MODELS = (
    ApogeeVisitSpectrum,
    ApogeeRestFrameVisitSpectrum,
    ApogeeCombinedSpectrum,
    ApogeeVisitSpectrumInApStar,
    ApogeeCoaddedSpectrumInApStar, 
    ApogeeMADGICSVisitSpectrum,
    BossVisitSpectrum,
    BossCombinedSpectrum,
    BossRestFrameVisitSpectrum
)    
from matplotlib.colors import LogNorm

import os
from astra.utils import expand_path

def human_size(bytes, units=['B','KB','MB','GB','TB', 'PB', 'EB']):
    """ Returns a human readable string representation of bytes """
    return str(bytes) + " " + units[0] if bytes < 1024 else human_size(bytes>>10, units[1:])

def render_links_to_summary_files(model, version=None):
    
    version = version or __version__

    possible_summary_files = [
        ("results for all co-added (star-level) spectra (`astraAllStar*`)", "astraAllStar"),
        ("results for all visit spectra (`astraAllVisit*`)", "astraAllVisit"),
    ]

    content = ""
    for desc, filetype in possible_summary_files:
        path = expand_path(f"$MWM_ASTRA/{version}/summary/{filetype}{model.__name__}-{version}.fits.gz")
        if os.path.exists(path):
            content += f"* {desc}:\n  <https://data.sdss5.org/sas/sdsswork/mwm/spectro/astra/{version}/summary/{filetype}{model.__name__}-{version}.fits.gz> ({human_size(os.path.getsize(path))})\n\n"

    return Markdown(content)    
    

def render_all_solar_results(model):
    solar = list(
        model
        .select()
        .join(Source)
        .where(Source.sdss4_apogee_id == "VESTA")
        .dicts()
    )

    headers = []
    for n in solar[0].keys():
        if n == "spectrum_pk":
            # Get the kind of spectrum model.
            headers.append(f"Spectrum model")
    
        headers.append(f"`{n}`")
    
    
    rows = []                        
    for row in solar:
        values = []
        for k, v in row.items():
            if k == "spectrum_pk":
                values.append(f"`{Spectrum.get(v).resolve().__class__.__name__}`")                    
            values.append(v)
        rows.append(values)


    return Markdown(tabulate(
        rows, 
        headers=headers
    ))
    
    
def prepare_data_for_gaia_hrd_plots(model, limit=None):
    return Table(rows=list(
        model
        .select(
            model,
            Source.bp_mag,
            Source.rp_mag,
            Source.g_mag,
            Source.plx
        )
        .join(Source)
        .limit(limit)
        .dicts()
    ))
    
    

def plot_gaia_hrd_coloured_by_field(results, field, functions=("min", "median", "max"), bins=200):
    
    warnings.simplefilter("ignore")
    
    fig, axes = plt.subplots(1, 3)
    fig.set_size_inches(10, 4)

    x = (results["bp_mag"].astype(float) - results["rp_mag"].astype(float))
    y = (results["g_mag"].astype(float) - 5 + 5 * np.log10(results["plx"].astype(float)))
    
    kwds = dict(
        bins=(
            np.linspace(-1, 4, bins),
            np.linspace(0, 20, bins)
        ),
        interpolation="None",
        colorbar=True,
        orientation="horizontal",
    )
    for ax, function in zip(axes, functions):
            
        plot_binned_statistic(
            x,
            y,
            results[field].astype(float),
            function=function,
            cmap="inferno",
            zlabel=f"{function}",
            ax=ax,
            **kwds
        )
        #ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xlabel("BP - RP")
        ax.set_ylabel("G")
    axes[1].set_title(field)
    fig.tight_layout()
    return fig




    
def render_all_solar_results_pivot(model):
    solar = list(
        model
        .select()
        .join(Source)
        .where(Source.sdss4_apogee_id == "VESTA")
        .dicts()
    )

    headers = ["Field name / Spectrum model"]
    if len(solar) == 0:
        return Markdown("No results for the Sun from this pipeline.")
    
    for n in solar[0].keys():
        headers.append(f"`{n}`")
    
    rows = [[h] for h in headers]
    for i, r in enumerate(solar):
        
        rows[0].append(f"`{Spectrum.get(r['spectrum_pk']).resolve().__class__.__name__}`")
        
    
        for j, (k, v) in enumerate(r.items(), start=1):
            rows[j].append(v)

    return Markdown(tabulate(rows[1:], headers=rows[0]))
    

def render_data_model(model):
    
    rows = []
    for name, field in model._meta.fields.items():        
        rows.append((f"`{name}`", field.help_text))

    return Markdown(
        tabulate(
            rows,
            headers=["Field", "Description"]
        )
    )
    
def render_flag_definitions(model, total=None):

    if total is None:
        total = model.select().count()

    fields = OrderedDict()
    for name, field in model._meta.fields.items():
        if isinstance(field, BitField):
            fields[name] = field

    flags = OrderedDict()
    for key, value in model.__dict__.items():
        if key.startswith("flag_") and (value.__class__.__name__ == "FlagDescriptor"):
            flags.setdefault(value._field.name, [])
            
            bit = int(np.log2(value._value))
            short_description = key[5:]
            full_description = value.help_text
            
            flags[value._field.name].append((bit, short_description, full_description))

    # Now ensure all the flag definitions are sorted by bit value.
    for name in flags.keys():
        flags[name] = sorted(flags[name], key=lambda item: item[0])

    if len(flags) == 0:
        return Markdown("""
::: {.callout-warning}
## No flag definitions on this data model!

This data model has no flag definitions!
:::
""")

    # Now print them for this Model.
    content = ""
    for name, field in fields.items():
        table = []
        if name in flags:
            for bit, short_description, full_description in flags[name]:

                count = (
                    model
                    .select()
                    .where(getattr(model, f"flag_{short_description}"))
                    .count()
                )
                table.append((bit, f"`{short_description.upper()}`", f"{count:,}", f"({100 * count/total:.1f}%)", full_description))
                
        content += f"### `{name}`\n"
        if len(table) == 0:
            content += \
f"""
::: {{.callout-warning}}
## No flag definitions for field `{name}` on this data model!
:::\n
"""
        else:
            content += tabulate(
                table, 
                headers=["Bit", "Name", "Occurrences", " ", "Description"],
                colalign=("left", "left", "left", "right", "left")
            ) + "\n\n"

    return Markdown(content)        
    
    
def render_row_counts_by_spectrum_model(model, spectrum_models=None):
    if spectrum_models is None:
        spectrum_models = DEFAULT_SPECTRUM_MODELS
        
    total, rows = (0, [])
    for spectrum_model in spectrum_models:
        count =(
            model
            .select()
            .join(spectrum_model, on=(spectrum_model.spectrum_pk == model.spectrum_pk))
            .count()
        )
        rows.append((f"{spectrum_model.filetype.default}", f"`astra.models.{spectrum_model.__name__}`", f"{count:,}"))
        total += count
    
    rows.append(("", "", f"**{total:,}**"))
    
    return Markdown(
        tabulate(
            rows,
            headers=["File type", "Spectrum model", "Number of result rows"],
            colalign=("left", "left", "right")
        )
    )


def prepare_apokasc_comparison(model, spectrum_model):
    warnings.simplefilter("ignore")
    
    apokasc = Table.read(
        expand_path(
            f"$MWM_ASTRA/aux/external-catalogs/APOKASC_cat_v7.0.5.fits"
        )
    )

    q = (
        model
        .select(
            model,
            Source.sdss4_apogee_id
        )
        .distinct(Source)
        .join(Source)
        .switch(model)
        .join(spectrum_model, on=(model.spectrum_pk == spectrum_model.spectrum_pk))
        .where(
            Source.sdss4_apogee_id.in_(list(apokasc["2MASS_ID"]))
        &   model.logg.is_null(False)
        )
        .dicts()
    )
    sdss4_apogee_ids = np.array([r["sdss4_apogee_id"] for r in q])
    _, apokasc_indices, pipeline_indices = np.intersect1d(apokasc["2MASS_ID"], sdss4_apogee_ids, return_indices=True)

    results = Table(rows=list(q))

    return (apokasc, results, apokasc_indices, pipeline_indices)


def plot_apokasc_rgb_rc_logg_comparison(apokasc, results):
        
    is_rgb = (apokasc["APOKASC3_EVSTATES"] == 1)
    is_rc = (apokasc["APOKASC3_EVSTATES"] == 2)

    x = apokasc["APOKASC3P_LOGG"]
    z = results["logg"]

    fig, (ax_rgb, ax_rc) = plt.subplots(1, 2)
    fig.set_size_inches(10, 4)
    
    ax_rgb.scatter(
        x[is_rgb],
        (z - x)[is_rgb],
        s=1, 
        alpha=0.5,
        c='k'
    )
    ax_rc.scatter(
        x[is_rc],
        (z - x)[is_rc],
        s=1,
        alpha=0.5,
        c='k'
    )
    ax_rgb.set_title("Red giant branch stars")
    ax_rgb.set_xlim(0, 4)
    ax_rc.set_xlim(2, 3)

    ax_rc.set_title("Red clump stars")
    for ax in (ax_rgb, ax_rc):
        ax.axhline(0, c="#666666", ls=":", lw=0.5, zorder=-1)
        ax.set_ylim(-1, 1)
        ax.set_xlabel(r"APOKASC logg")
        ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))

        if ax.is_first_col():
            ax.set_ylabel(r"logg - APOKSAC logg")
    #fig.tight_layout()
    return fig


def plot_apokasc_logg_distributions(apokasc, results, model):
    y_seismo = apokasc["APOKASC3P_LOGG"]
    y_spectro = results["logg"]

    is_rgb = (apokasc["APOKASC3_EVSTATES"] == 1)
    is_rc = (apokasc["APOKASC3_EVSTATES"] == 2)

    bins = np.arange(0.5, 4, 0.05)

    fig, (ax_all, ax_rgb, ax_rc) = plt.subplots(3, 1)
    fig.set_size_inches(10, 6)

    ax_rgb.hist(
        y_seismo[is_rgb],
        bins=bins,
        edgecolor="tab:blue",
        facecolor="None",
    )
    ax_rgb.hist(
        y_spectro[is_rgb],
        bins=bins,
        edgecolor="k",
        facecolor="None",   
    )
    ax_rc.hist(
        y_seismo[is_rc],
        bins=bins,
        edgecolor="tab:blue",
        facecolor="None",
    )
    ax_rc.hist(
        y_spectro[is_rc],
        bins=bins,
        edgecolor="k",
        facecolor="None",   
    )

    ax_all.hist(
        y_seismo,
        bins=bins,
        edgecolor="tab:blue",
        facecolor="None",
        label="APOKASC"
    )
    ax_all.hist(
        y_spectro,
        bins=bins,
        edgecolor="k",
        facecolor="None",
        label=model.__name__
    )
    ax_rgb.set_title("RGB (according to APOKASC)")
    ax_rc.set_title("RC (according to APOKASC)")
    ax_all.legend()

    ax_all.set_title("All")
    ax_rc.set_xlabel("logg")
    ax_all.set_ylabel("count")
    fig.tight_layout()    
    return fig


def plot_hrd_of_apokasc_sample(results):
    
    x_spectro = results["teff"]
    y_spectro = results["logg"]
    c_spectro = results["fe_h"]


    fig = plt.figure()
    fig.set_size_inches(10, 4.5)
    ax_hrd = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax_density = plt.subplot2grid((2, 2), (0, 0))
    ax_log_density = plt.subplot2grid((2, 2), (1, 0))
    
    #fig, (ax_density, ax_log_density, ax_hrd) = plt.subplots(3, 1)
    #fig.set_size_inches(10, 0)

    n_bins = 100
    min_entries_per_bin = 2
    bins = (
        np.linspace(3500, 6000, n_bins),
        np.linspace(0.5, 4, n_bins)
    )
    H, xedges, yedges, binnumber = binned_statistic_2d(
        x_spectro, y_spectro, y_spectro,
        statistic="count", 
        bins=bins
    )
    H[H < min_entries_per_bin] = np.nan
    imshow_kwds = dict(
        vmin=None, vmax=None,
        aspect=np.ptp(xedges)/np.ptp(yedges), 
        extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]),
        cmap="inferno",
        interpolation="none",
    )
    im = ax_density.imshow(H.T, **imshow_kwds)
    log_im = ax_log_density.imshow(H.T, norm=LogNorm(), **imshow_kwds)

    for ax in (ax_density, ax_log_density):
        ax.set_xlabel("Teff")
        ax.set_ylabel("logg")
        ax.set_xlim(np.max(bins[0]), np.min(bins[0]))
        ax.set_ylim(np.max(bins[1]), np.min(bins[1]))    
        ax.set_aspect(np.ptp(ax.get_xlim())/np.ptp(ax.get_ylim()))    

    cbar_im = plt.colorbar(im, ax=ax_density)
    cbar_im.set_label("Counts")
    
    cbar_log_im = plt.colorbar(log_im, ax=ax_log_density)
    cbar_log_im.set_label("Counts")

    scat = ax_hrd.scatter(
        x_spectro,
        y_spectro,
        c=c_spectro,
        vmin=-1,
        vmax=0.5,
        s=1,
    )
    ax_hrd.set_xlabel("Teff")
    ax_hrd.set_ylabel("logg")
    ax_hrd.set_xlim(np.max(bins[0]), np.min(bins[0]))
    ax_hrd.set_ylim(np.max(bins[1]), np.min(bins[1]))    
    ax_hrd.set_aspect(np.ptp(ax_hrd.get_xlim())/np.ptp(ax_hrd.get_ylim()))    
    cbar = plt.colorbar(scat, ax=ax_hrd)
    cbar.set_label("[Fe/H]")
    fig.tight_layout()    

    return fig


def plot_binned_statistic(
    x, y, z,
    bins=100,
    function=np.nanmedian,
    xlabel=None,
    ylabel=None,
    zlabel=None,
    ax=None,
    colorbar=False,
    figsize=(8, 8),
    vmin=None,
    vmax=None,
    min_entries_per_bin=None,
    subsample=None,
    mask=None,
    orientation="vertical",
    full_output=False,
    **kwargs
):
    
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=figsize)
    else:
        fig = ax.figure
    
    finite = np.isfinite(x * y * z)
    if mask is not None:
        finite *= mask
    if subsample is not None:
        idx = np.where(finite)[0]
        if subsample < 1:
            subsample *= idx.size
        if int(subsample) > idx.size:
            finite = idx
        else:
            finite = np.random.choice(idx, int(subsample), replace=False)
    
    H, xedges, yedges, binnumber = binned_statistic_2d(
        x[finite], y[finite], z[finite],
        statistic=function, bins=bins)

    if min_entries_per_bin is not None:
        if function != "count":
            H_count, xedges, yedges, binnumber = binned_statistic_2d(
                x[finite], y[finite], z[finite],
                statistic="count", bins=bins)

        else:
            H_count = H

        H[H_count < min_entries_per_bin] = np.nan
        if not np.any(np.isfinite(H)) or np.max(H) == 0:
            return (fig, None) if full_output else fig


    if (vmin is None or vmax is None) and "norm" not in kwargs:
        vmin_default, med, vmax_default = np.nanpercentile(H, kwargs.pop("norm_percentiles", [5, 50, 95]))
        if vmin is None:
            vmin = vmin_default
        if vmax is None:
            vmax = vmax_default
    
    imshow_kwds = dict(
        vmin=vmin, vmax=vmax,
        aspect=np.ptp(xedges)/np.ptp(yedges), 
        extent=(xedges[0], xedges[-1], yedges[-1], yedges[0]),
        cmap="inferno",
        interpolation="bilinear")
    imshow_kwds.update(kwargs)
    
    image = ax.imshow(H.T, **imshow_kwds)
    if colorbar:
        if orientation == "vertical":
            colorbar_kwds = dict(fraction=0.046, pad=0.04)
        else:
            colorbar_kwds = dict(fraction=0.026, pad=0.24)
        cbar = plt.colorbar(image, ax=ax, orientation=orientation, **colorbar_kwds)
        if zlabel is not None:
            cbar.set_label(zlabel)

    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)

    return (fig, image) if full_output else fig




def plot_formal_error_wrt_snr(model, spectrum_models, unflagged=True, x_min=1, x_max=100, n_bins=200, y_max=None, default_y_max=0.5, label_names=("teff", "logg", "fe_h")):
        
    fields = []
    for ln in label_names:
        fields.extend([getattr(model, ln), getattr(model, f"e_{ln}")])
        
    rows = []
    for spectrum_model in spectrum_models:
        q = (
            model
            .select(
                spectrum_model.snr,
                *fields
            )
            .join(
                spectrum_model, 
                on=(spectrum_model.spectrum_pk == model.spectrum_pk)
            )
        )
        if unflagged:
            q = q.where(model.result_flags == 0)
            
        rows.extend(list(q.dicts()))
    precision_data = Table(rows=rows)

    if y_max is None:
        y_max = {}
    
    default_y_max_values = dict(
        teff=500,
        logg=0.5,
        fe_h=0.5
    )
    for k, v in default_y_max_values.items():
        y_max.setdefault(k, v)
    
    L = len(label_names)
    K = int(np.ceil(L / 3))
    fig, axes = plt.subplots(K, 3)
    fig.set_size_inches(10, 10/3. * K)
    
    for ax, label_name in zip(axes.flat, label_names):
        plot_binned_statistic(
            precision_data["snr"].astype(float),
            precision_data[f"e_{label_name}"].astype(float),
            precision_data[f"e_{label_name}"].astype(float),
            function="count",
            norm=LogNorm(),
            bins=(
                np.linspace(x_min, x_max, n_bins),
                np.linspace(0, y_max.get(label_name, default_y_max), n_bins)
            ),
            interpolation="none",
            ax=ax
        )
        ax.set_ylim(ax.get_ylim()[::-1])
        ax.set_xlabel("S/N")
        ax.set_ylabel(f"e_{label_name}")
    for ax in axes.flat[L:]:
        ax.set_visible(False)
        
    fig.tight_layout()
    return fig




def plot_visits_by_mjd(spectrum_visit_model, pipeline_model):

    second_where = pipeline_model.spectrum_pk.is_null(False)
    if hasattr(pipeline_model, "result_flags"):
        second_where &= (pipeline_model.result_flags == 0)
    
    q = (
        spectrum_visit_model
        .select(
            spectrum_visit_model.mjd,
            fn.count(spectrum_visit_model.mjd),
            fn.count(Case(None, ((pipeline_model.spectrum_pk.is_null(False), 1), ))),
            fn.count(Case(None, ((second_where, 1), )))
        )
        .join(pipeline_model, JOIN.LEFT_OUTER, on=(pipeline_model.spectrum_pk == spectrum_visit_model.spectrum_pk))
        .group_by(spectrum_visit_model.mjd)
        .tuples()
    )

    data = np.array(list(q))
    data = data[np.argsort(data[:, 0])]
    mjd, n_spectra, n_results, n_good_results = data.T

    dates = Time(mjd, format="mjd")

    n_future = 30
    fig, (ax_count, ax_fail) = plt.subplots(2, 1)
    fig.set_size_inches(10, 5)
    ax_count.plot(dates.datetime, n_spectra + 1, c="k", drawstyle="steps-mid", label=f"{spectrum_visit_model.__name__}")
    ax_count.plot(dates.datetime, n_results + 1, c="tab:blue", drawstyle="steps-mid", label=f"{pipeline_model.__name__} results")
    ax_count.plot(dates.datetime, n_good_results + 1, c="tab:red", drawstyle="steps-mid", label=f"{pipeline_model.__name__} results without flags")

    ax_fail.plot(mjd, 100 * (1 - n_results / n_spectra), c="tab:blue", drawstyle="steps-mid")

    ax_count.legend(loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.25), frameon=False)

    ax_count.semilogy()
    ax_fail.semilogy()

    ax_count.set_xlim(dates[0].datetime, Time(mjd[-1] + n_future, format="mjd").datetime)
    show_years = np.unique(dates.byear.astype(int))[1:]

    ax_count.set_xticks([datetime(y, 1, 1) for y in show_years])
    ax_count.set_xticklabels(map(str, show_years))
    ax_count.set_ylabel("Count + 1")

    ax_fail.set_xlim(mjd[0], mjd[-1] + n_future)
    ax_fail.set_xlabel("MJD")
    ax_fail.set_ylabel(f"Failures [%]")
    fig.tight_layout()    
    return fig


def plot_kiel_density(
    teff, logg, n_bins=200, x_lims=(3000, 6500), y_lims=(-0.5, 6),
    min_entries_per_bin=5, log=True,
):
    
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(10, 10)

    args = (teff, logg, logg)
    kwds = dict(
        bins=(
            np.linspace(*x_lims, n_bins),
            np.linspace(*y_lims, n_bins)
        ),    
        interpolation="None",
        colorbar=True
    )
    if log:
        kwds["norm"] = LogNorm()
    plot_binned_statistic(
        *args,
        function="count",
        cmap="inferno",
        zlabel="Count",
        ax=ax,
        min_entries_per_bin=min_entries_per_bin,
        **kwds
    )

    ax.set_xlim(ax.get_xlim()[::-1])
    ax.set_xlabel("Teff [K]")
    ax.set_ylabel("log(g)")
    fig.tight_layout()
    return fig


def plot_z_scores(model_name):
    with open(expand_path(f"$MWM_ASTRA/{__version__}/aux/{model_name}.pkl"), "rb") as fp:
        content = pickle.load(fp)

    with open(expand_path(f"$MWM_ASTRA/{__version__}/aux/{model_name}_corrections.pkl"), "rb") as fp:
        corrections, reference = pickle.load(fp)
        
    field_names = [ea[2:-2] for ea in content.keys() if ea.startswith("e_") and ea.endswith("_0")]
    N = len(field_names)
    fig, axes = plt.subplots(N, 2)
    
    fig.set_size_inches(10, N * 5)
    bins = np.linspace(-5, 5, 100)
    for i, (ax_before, ax_after) in enumerate(axes):
        name = field_names[i]
        dz = (content[f"{name}_0"] - content[f"{name}_1"])
        var_0, var_1 = (content[f"e_{name}_0"]**2, content[f"e_{name}_1"]**2)
        z = dz / np.sqrt(var_0 + var_1)
        
        H, bin_edges = np.histogram(z, bins=bins, density=True)
        ax_before.plot(
            bin_edges[:-1] + np.diff(bin_edges)[0]/2,
            H,
            c="k",
            label="raw (formal)",
            drawstyle="steps-mid"
        )

        ax_before.plot(
            reference["bin_edges"][:-1] + np.diff(reference["bin_edges"])[0]/2,
            reference["reference_pdf"],
            c="tab:blue",
            label="N(0, 1)",
            drawstyle="steps-mid",
        )
        ax_before.legend()
                        
        ax_after.plot(
            reference["bin_edges"][:-1] + np.diff(reference["bin_edges"])[0]/2,
            reference["reference_pdf"],
            c="tab:blue",
            drawstyle="steps-mid",
            label="N(0, 1)"
        )        
        ax_after.plot(
            reference["bin_edges"][:-1] + np.diff(reference["bin_edges"])[0]/2,
            corrections[name]['best_z_pdf'],
            c="k",
            drawstyle="steps-mid",
            label="after"
        )
        ax_after.legend()
        offset, scale = (corrections[name]["offset"], corrections[name]["scale"])
        ax_after.text(
            0.05, 0.95,
            f"scale={scale:.2e}; offset={offset:.2e}",
            transform=ax_after.transAxes
        )

        if i == 0:
            ax_before.set_title("Raw (formal)")
            ax_after.set_title("Corrected")
        ax_before.set_xlabel(f"z_{name} (before)")
        ax_after.set_xlabel(f"z_{name} (after)")

    fig.tight_layout()
    return fig        
        
        
    

def plot_kiel_two_panel(teff, logg, fe_h, n_bins=200, x_lims=(3000, 6500), y_lims=(-0.5, 6), log=True):
    
    fig, axes = plt.subplots(1, 2)
    fig.set_size_inches(10, 5)

    args = (teff, logg, fe_h)
    kwds = dict(
        bins=(
            np.linspace(*x_lims, n_bins),
            np.linspace(*y_lims, n_bins)
        ),    
        interpolation="None",
        colorbar=True,
    )
    plot_binned_statistic(
        *args,
        function="median",
        cmap="viridis",
        zlabel="[Fe/H]",
        ax=axes[1],
        **kwds
    )    
    if log:
        kwds["norm"] = LogNorm()
    plot_binned_statistic(
        *args,
        function="count",
        cmap="inferno",
        zlabel="Count",
        ax=axes[0],
        **kwds
    )

    for ax in axes:
        ax.set_xlim(ax.get_xlim()[::-1])
        ax.set_xlabel("Teff [K]")
        ax.set_ylabel("log(g)")
    fig.tight_layout()
    return fig


import matplotlib.pyplot as plt
from astropy.table import Table


def plot_cluster_view(
    where,
    model, 
    spectrum_model, 
    metallicity_field_name="fe_h",
    teff_limits=(3000, 7000),
    logg_limits=(0, 5.5),
    
):

    fields = [
        model.teff,
        model.logg,
        getattr(model, metallicity_field_name).alias("fe_h"),
        model.e_teff,
        model.e_logg,
        getattr(model, f"e_{metallicity_field_name}").alias("e_fe_h"),
    ]

    data = Table(rows=list(
        model
        .select(*fields)
        .join(spectrum_model, on=(model.spectrum_pk == spectrum_model.spectrum_pk))
        .switch(model)
        .join(Source)
        .where(where) 
        .dicts()
    ))
    if len(data) == 0:
        return None

    fig = plt.figure()
    fig.set_size_inches(10, 4.5)
    ax = plt.subplot2grid((2, 2), (0, 1), rowspan=2)
    ax_teff = plt.subplot2grid((2, 2), (0, 0))
    ax_logg = plt.subplot2grid((2, 2), (1, 0))

    ax.errorbar(
        data["teff"].astype(float),
        data["logg"].astype(float),
        xerr=data["e_teff"].astype(float),
        yerr=data["e_logg"].astype(float),
        fmt='none',
        elinewidth=0.5,
        ecolor="#666666",
    )
    scat = ax.scatter(
        data["teff"].astype(float),
        data["logg"].astype(float),
        c=data["fe_h"].astype(float),
        s=2,
        zorder=10
    )

    ax.set_xlim(max(teff_limits), min(teff_limits))
    ax.set_ylim(max(logg_limits), min(logg_limits))
    ax.set_xlabel("Teff [K]")
    ax.set_ylabel("log(g)")

    cbar = plt.colorbar(scat, ax=ax)
    cbar.set_label("[Fe/H]")

    ax_teff.scatter(data["teff"], data["fe_h"], s=1, c='k')
    ax_teff.errorbar(
        data["teff"].astype(float),
        data["fe_h"].astype(float),
        xerr=data["e_teff"].astype(float),
        yerr=data["e_fe_h"].astype(float),
        fmt='none',
        elinewidth=0.5,
        ecolor="#666666",
        zorder=-1
    )
    ax_teff.set_xlabel("Teff [K]")
    ax_teff.set_xlim(teff_limits)
    
    ax_logg.scatter(data["logg"], data["fe_h"], s=1, c='k')
    ax_logg.errorbar(
        data["logg"].astype(float),
        data["fe_h"].astype(float),
        xerr=data["e_logg"].astype(float),
        yerr=data["e_fe_h"].astype(float),
        fmt='none',
        elinewidth=0.5,
        ecolor="#666666",
        zorder=-1
    )
    ax_logg.set_xlabel("log(g)")
    ax_logg.set_xlim(logg_limits)
    for ax in (ax_teff, ax_logg):
        ax.set_ylabel("[Fe/H]")

    fig.tight_layout()
    return fig
    

