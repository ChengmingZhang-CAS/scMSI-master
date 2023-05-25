import inspect
import logging
import warnings
from typing import Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch
from scipy.sparse import issparse
from sklearn.mixture import GaussianMixture

from scvi import _CONSTANTS
from scvi._compat import Literal
from scvi.data import get_from_registry

logger = logging.getLogger(__name__)

Number = Union[int, float]


class DifferentialComputation:
    """
    Unified class for differential computation.

    This class takes a function from a model like `SCVI` or `TOTALVI` and takes outputs
    from this function with respect to the adata input and computed Bayes factors as
    described in [Lopez18]_, [Xu21]_, or [Boyeau19]_.

    Parameters
    ----------
    model_fn
        Function in model API to get values from.
    adata
        AnnData setup with scvi
    """

    def __init__(self, model_fn, adata):
        self.adata = adata
        self.model_fn = model_fn

    def get_bayes_factors(
        self,
        idx1: Union[List[bool], np.ndarray],
        idx2: Union[List[bool], np.ndarray],
        mode: Literal["vanilla", "change"] = "vanilla",
        batchid1: Optional[Sequence[Union[Number, str]]] = None,
        batchid2: Optional[Sequence[Union[Number, str]]] = None,
        use_observed_batches: Optional[bool] = False,
        n_samples: int = 5000,
        use_permutation: bool = False,
        m_permutation: int = 10000,
        change_fn: Optional[Union[str, Callable]] = None,
        m1_domain_fn: Optional[Callable] = None,
        delta: Optional[float] = 0.5,
        pseudocounts: Union[float, None] = 0.0,
        cred_interval_lvls: Optional[Union[List[float], np.ndarray]] = None,
    ) -> Dict[str, np.ndarray]:
        r"""
        A unified method for differential expression inference.

        Two modes coexist:

        - the `"vanilla"` mode follows protocol described in [Lopez18]_ and [Xu21]_
        In this case, we perform hypothesis testing based on the hypotheses

        .. math::
            M_1: h_1 > h_2 ~\text{and}~ M_2: h_1 \leq h_2.

        DE can then be based on the study of the Bayes factors

        .. math::
            \log p(M_1 | x_1, x_2) / p(M_2 | x_1, x_2).

        - the `"change"` mode (described in [Boyeau19]_)
        This mode consists of estimating an effect size random variable (e.g., log fold-change) and
        performing Bayesian hypothesis testing on this variable.
        The `change_fn` function computes the effect size variable :math:`r` based on two inputs
        corresponding to the posterior quantities (e.g., normalized expression) in both populations.

        Hypotheses:

        .. math::
            M_1: r \in R_1 ~\text{(effect size r in region inducing differential expression)}

        .. math::
            M_2: r  \notin R_1 ~\text{(no differential expression)}

        To characterize the region :math:`R_1`, which induces DE, the user has two choices.

        1. A common case is when the region :math:`[-\delta, \delta]` does not induce differential
           expression. If the user specifies a threshold delta, we suppose that :math:`R_1 = \mathbb{R} \setminus [-\delta, \delta]`
        2. Specify an specific indicator function:

        .. math::
            f: \mathbb{R} \mapsto \{0, 1\} ~\text{s.t.}~ r \in R_1 ~\text{iff.}~ f(r) = 1.

        Decision-making can then be based on the estimates of

        .. math::
            p(M_1 \mid x_1, x_2).

        Both modes require to sample the posterior distributions.
        To that purpose, we sample the posterior in the following way:

        1. The posterior is sampled `n_samples` times for each subpopulation.
        2. For computational efficiency (posterior sampling is quite expensive), instead of
           comparing the obtained samples element-wise, we can permute posterior samples.
           Remember that computing the Bayes Factor requires sampling :math:`q(z_A \mid x_A)` and :math:`q(z_B \mid x_B)`.

        Currently, the code covers several batch handling configurations:

        1. If ``use_observed_batches=True``, then batch are considered as observations
           and cells' normalized means are conditioned on real batch observations.
        2. If case (cell group 1) and control (cell group 2) are conditioned on the same
           batch ids. This requires ``set(batchid1) == set(batchid2)`` or ``batchid1 == batchid2 === None``.
        3. If case and control are conditioned on different batch ids that do not intersect
           i.e., ``set(batchid1) != set(batchid2)`` and ``len(set(batchid1).intersection(set(batchid2))) == 0``.

        This function does not cover other cases yet and will warn users in such cases.

        Parameters
        ----------
        mode
            one of ["vanilla", "change"]
        idx1
            bool array masking subpopulation cells 1. Should be True where cell is
            from associated population
        idx2
            bool array masking subpopulation cells 2. Should be True where cell is
            from associated population
        batchid1
            List of batch ids for which you want to perform DE Analysis for
            subpopulation 1. By default, all ids are taken into account
        batchid2
            List of batch ids for which you want to perform DE Analysis for
            subpopulation 2. By default, all ids are taken into account
        use_observed_batches
            Whether posterior values are conditioned on observed
            batches
        n_samples
            Number of posterior samples
        use_permutation
            Activates step 2 described above.
            Simply formulated, pairs obtained from posterior sampling
            will be randomly permuted so that the number of pairs used
            to compute Bayes Factors becomes `m_permutation`.
        m_permutation
            Number of times we will "mix" posterior samples in step 2.
            Only makes sense when `use_permutation=True`
        change_fn
            function computing effect size based on both posterior values
        m1_domain_fn
            custom indicator function of effect size regions
            inducing differential expression
        delta
            specific case of region inducing differential expression.
            In this case, we suppose that :math:`R \setminus [-\delta, \delta]` does not induce differential expression
            (LFC case). If the provided value is `None`, then a proper threshold is determined
            from the distribution of LFCs accross genes.
        pseudocounts
            pseudocount offset used for the mode `change`.
            When None, observations from non-expressed genes are used to estimate its value.
        cred_interval_lvls
            List of credible interval levels to compute for the posterior
            LFC distribution

        Returns
        -------
        Differential expression properties

        """
        # if not np.array_equal(self.indices, np.arange(len(self.dataset))):
        #     warnings.warn(
        #         "Differential expression requires a Posterior object created with all indices."
        #     )
        eps = 1e-8
        # Normalized means sampling for both populations
        scales_batches_1 = self.scale_sampler(
            selection=idx1,
            batchid=batchid1,
            use_observed_batches=use_observed_batches,
            n_samples=n_samples,
        )
        scales_batches_2 = self.scale_sampler(
            selection=idx2,
            batchid=batchid2,
            use_observed_batches=use_observed_batches,
            n_samples=n_samples,
        )

        px_scale_mean1 = scales_batches_1["scale"].mean(axis=0)
        px_scale_mean2 = scales_batches_2["scale"].mean(axis=0)

        # Sampling pairs
        # The objective of code section below is to ensure than the samples of normalized
        # means we consider are conditioned on the same batch id
        batchid1_vals = np.unique(scales_batches_1["batch"])
        batchid2_vals = np.unique(scales_batches_2["batch"])

        create_pairs_from_same_batches = (
            set(batchid1_vals) == set(batchid2_vals)
        ) and not use_observed_batches
        if create_pairs_from_same_batches:
            # First case: same batch normalization in two groups
            logger.debug("Same batches in both cell groups")
            n_batches = len(set(batchid1_vals))
            n_samples_per_batch = (
                m_permutation // n_batches if m_permutation is not None else None
            )
            logger.debug(
                "Using {} samples per batch for pair matching".format(
                    n_samples_per_batch
                )
            )
            scales_1 = []
            scales_2 = []
            for batch_val in set(batchid1_vals):
                # Select scale samples that originate from the same batch id
                scales_1_batch = scales_batches_1["scale"][
                    scales_batches_1["batch"] == batch_val
                ]
                scales_2_batch = scales_batches_2["scale"][
                    scales_batches_2["batch"] == batch_val
                ]

                # Create more pairs
                scales_1_local, scales_2_local = pairs_sampler(
                    scales_1_batch,
                    scales_2_batch,
                    use_permutation=use_permutation,
                    m_permutation=n_samples_per_batch,
                )
                scales_1.append(scales_1_local)
                scales_2.append(scales_2_local)
            scales_1 = np.concatenate(scales_1, axis=0)
            scales_2 = np.concatenate(scales_2, axis=0)
        else:
            logger.debug("Ignoring batch conditionings to compare means")
            if len(set(batchid1_vals).intersection(set(batchid2_vals))) >= 1:
                warnings.warn(
                    "Batchids of cells groups 1 and 2 are different but have an non-null "
                    "intersection. Specific handling of such situations is not implemented "
                    "yet and batch correction is not trustworthy."
                )
            scales_1, scales_2 = pairs_sampler(
                scales_batches_1["scale"],
                scales_batches_2["scale"],
                use_permutation=use_permutation,
                m_permutation=m_permutation,
            )

        # Adding pseudocounts to the scales
        if pseudocounts is None:
            logger.debug("Estimating pseudocounts offet from the data")
            x = get_from_registry(self.adata, _CONSTANTS.X_KEY)
            where_zero_a = densify(np.max(x[idx1], 0)) == 0
            where_zero_b = densify(np.max(x[idx2], 0)) == 0
            pseudocounts = estimate_pseudocounts_offset(
                scales_a=scales_1,
                scales_b=scales_2,
                where_zero_a=where_zero_a,
                where_zero_b=where_zero_b,
            )
        logger.debug("Using pseudocounts ~ {}".format(pseudocounts))
        # Core of function: hypotheses testing based on the posterior samples we obtained above
        if mode == "vanilla":
            logger.debug("Differential expression using vanilla mode")
            proba_m1 = np.mean(scales_1 > scales_2, 0)
            proba_m2 = 1.0 - proba_m1
            res = dict(
                proba_m1=proba_m1,
                proba_m2=proba_m2,
                bayes_factor=np.log(proba_m1 + eps) - np.log(proba_m2 + eps),
                scale1=px_scale_mean1,
                scale2=px_scale_mean2,
            )

        elif mode == "change":
            logger.debug("Differential expression using change mode")

            # step 1: Construct the change function
            def lfc(x, y):
                return np.log2(x + pseudocounts) - np.log2(y + pseudocounts)

            if change_fn == "log-fold" or change_fn is None:
                change_fn = lfc
            elif not callable(change_fn):
                raise ValueError("'change_fn' attribute not understood")

            # step2: Construct the DE area function
            if m1_domain_fn is None:

                def m1_domain_fn(samples):
                    delta_ = (
                        delta
                        if delta is not None
                        else estimate_delta(lfc_means=samples.mean(0))
                    )
                    logger.debug("Using delta ~ {:.2f}".format(delta_))
                    return np.abs(samples) >= delta_

            change_fn_specs = inspect.getfullargspec(change_fn)
            domain_fn_specs = inspect.getfullargspec(m1_domain_fn)
            if (len(change_fn_specs.args) != 2) | (len(domain_fn_specs.args) != 1):
                raise ValueError(
                    "change_fn should take exactly two parameters as inputs; m1_domain_fn one parameter."
                )
            try:
                change_distribution = change_fn(scales_1, scales_2)
                is_de = m1_domain_fn(change_distribution)
                delta_ = (
                    estimate_delta(lfc_means=change_distribution.mean(0))
                    if delta is None
                    else delta
                )
            except TypeError:
                raise TypeError(
                    "change_fn or m1_domain_fn have has wrong properties."
                    "Please ensure that these functions have the right signatures and"
                    "outputs and that they can process numpy arrays"
                )
            proba_m1 = np.mean(is_de, 0)
            change_distribution_props = describe_continuous_distrib(
                samples=change_distribution,
                credible_intervals_levels=cred_interval_lvls,
            )
            change_distribution_props = {
                "lfc_" + key: val for (key, val) in change_distribution_props.items()
            }

            res = dict(
                proba_de=proba_m1,
                proba_not_de=1.0 - proba_m1,
                bayes_factor=np.log(proba_m1 + eps) - np.log(1.0 - proba_m1 + eps),
                scale1=px_scale_mean1,
                scale2=px_scale_mean2,
                pseudocounts=pseudocounts,
                delta=delta_,
                **change_distribution_props,
            )
        else:
            raise NotImplementedError("Mode {mode} not recognized".format(mode=mode))

        return res

    @torch.no_grad()
    def scale_sampler(
        self,
        selection: Union[List[bool], np.ndarray],
        n_samples: Optional[int] = 5000,
        n_samples_per_cell: Optional[int] = None,
        batchid: Optional[Sequence[Union[Number, str]]] = None,
        use_observed_batches: Optional[bool] = False,
        give_mean: Optional[bool] = False,
    ) -> dict:
        """
        Samples the posterior scale using the variational posterior distribution.

        Parameters
        ----------
        selection
            Mask or list of cell ids to select
        n_samples
            Number of samples in total per batch (fill either `n_samples_total`
            or `n_samples_per_cell`)
        n_samples_per_cell
            Number of time we sample from each observation per batch
            (fill either `n_samples_total` or `n_samples_per_cell`)
        batchid
            Biological batch for which to sample from.
            Default (None) sample from all batches
        use_observed_batches
            Whether normalized means are conditioned on observed
            batches or if observed batches are to be used
        give_mean
            Return mean of values


        Returns
        -------
        type
            Dictionary containing:
            `scale`
            Posterior aggregated scale samples of shape (n_samples, n_vars)
            where n_samples correspond to either:
            - n_bio_batches * n_cells * n_samples_per_cell
            or
            - n_samples_total
            `batch`
            associated batch ids

        """
        # Get overall number of desired samples and desired batches
        if batchid is None and not use_observed_batches:
            categorical_mappings = self.adata.uns["_scvi"]["categorical_mappings"]
            batchid = categorical_mappings["_scvi_batch"]["mapping"]
        if use_observed_batches:
            if batchid is not None:
                raise ValueError("Unconsistent batch policy")
            batchid = [None]
        if n_samples is None and n_samples_per_cell is None:
            n_samples = 5000
        elif n_samples_per_cell is not None and n_samples is None:
            n_samples = n_samples_per_cell * len(selection)
        if (n_samples_per_cell is not None) and (n_samples is not None):
            warnings.warn(
                "n_samples and n_samples_per_cell were provided. Ignoring n_samples_per_cell"
            )
        n_samples = int(n_samples / len(batchid))
        if n_samples == 0:
            warnings.warn(
                "very small sample size, please consider increasing `n_samples`"
            )
            n_samples = 2

        # Selection of desired cells for sampling
        if selection is None:
            raise ValueError("selections should be a list of cell subsets indices")
        selection = np.asarray(selection)
        if selection.dtype is np.dtype("bool"):
            if len(selection) < self.adata.shape[0]:
                raise ValueError("Mask must be same length as adata.")
            selection = np.asarray(np.where(selection)[0].ravel())

        # Sampling loop
        px_scales = []
        batch_ids = []
        for batch_idx in batchid:
            idx_selected = np.arange(self.adata.shape[0])[selection]
            px_scales.append(
                self.model_fn(
                    self.adata,
                    indices=idx_selected,
                    transform_batch=batch_idx,
                    n_samples_overall=n_samples,
                )
            )
            batch_idx = batch_idx if batch_idx is not None else np.nan
            batch_ids.append([batch_idx] * px_scales[-1].shape[0])
        px_scales = np.concatenate(px_scales)
        batch_ids = np.concatenate(batch_ids).reshape(-1)
        if px_scales.shape[0] != batch_ids.shape[0]:
            raise ValueError("sampled scales and batches have inconsistent shapes")
        if give_mean:
            px_scales = px_scales.mean(0)
        return dict(scale=px_scales, batch=batch_ids)


def estimate_delta(lfc_means: List[np.ndarray], coef=0.6, min_thres=0.3):
    """
    Computes a threshold LFC value based on means of LFCs.

    Parameters
    ----------
    lfc_means
        LFC means for each gene, should be 1d.
    coef
        Tunable hyperparameter to choose the threshold based on estimated modes, defaults to 0.6
    min_thres
        Minimum returned threshold value, defaults to 0.3
    """
    logger.debug("Estimating delta from effect size samples")
    if lfc_means.ndim >= 2:
        raise ValueError("lfc_means should be 1-dimensional of shape: (n_genes,).")
    gmm = GaussianMixture(n_components=3)
    gmm.fit(lfc_means[:, None])
    vals = np.sort(gmm.means_.squeeze())
    res = coef * np.abs(vals[[0, -1]]).mean()
    res = np.maximum(min_thres, res)
    return res


def estimate_pseudocounts_offset(
    scales_a: List[np.ndarray],
    scales_b: List[np.ndarray],
    where_zero_a: List[np.ndarray],
    where_zero_b: List[np.ndarray],
    percentile: Optional[float] = 0.9,
):
    """
    Determines pseudocount offset.

    This shrinks LFCs asssociated with non-expressed genes to zero.

    Parameters
    ----------
    scales_a
        Scales in first population
    scales_b
        Scales in second population
    where_zero_a
        mask where no observed counts
    where_zero_b
        mask where no observed counts
    """
    max_scales_a = np.max(scales_a, 0)
    max_scales_b = np.max(scales_b, 0)
    asserts = (
        (max_scales_a.shape == where_zero_a.shape)
        and (max_scales_b.shape == where_zero_b.shape)
    ) and (where_zero_a.shape == where_zero_b.shape)
    if not asserts:
        raise ValueError(
            "Dimension mismatch between scales and/or masks to compute the pseudocounts offset."
        )
    if where_zero_a.sum() >= 1:
        artefact_scales_a = max_scales_a[where_zero_a]
        eps_a = np.percentile(artefact_scales_a, q=percentile)
    else:
        eps_a = 1e-10

    if where_zero_b.sum() >= 1:
        artefact_scales_b = max_scales_b[where_zero_b]
        eps_b = np.percentile(artefact_scales_b, q=percentile)
    else:
        eps_b = 1e-10
    res = np.maximum(eps_a, eps_b)
    return res


def pairs_sampler(
    arr1: Union[List[float], np.ndarray, torch.Tensor],
    arr2: Union[List[float], np.ndarray, torch.Tensor],
    use_permutation: bool = True,
    m_permutation: int = None,
    sanity_check_perm: bool = False,
    weights1: Union[List[float], np.ndarray, torch.Tensor] = None,
    weights2: Union[List[float], np.ndarray, torch.Tensor] = None,
) -> tuple:
    """
    Creates more pairs.

    In a context where we want to estimate a double sum, virtually increases the number
    of samples by considering more pairs so as to better estimate the double summation operation

    Parameters
    ----------
    arr1
        samples from population 1
    arr2
        samples from population 2
    use_permutation
        Whether to mix samples from both populations
    m_permutation
        param sanity_check_perm: If True, resulting mixed arrays arr1 and arr2 are mixed together
        In most cases, this parameter should remain False
    sanity_check_perm
        TODO
    weights1
        probabilities associated to array 1 for random sampling
    weights2
        probabilities associated to array 2 for random sampling

    Returns
    -------
    type
        new_arr1, new_arr2
    """
    if use_permutation is True:
        # prepare the pairs for sampling
        n_arr1 = arr1.shape[0]
        n_arr2 = arr2.shape[0]
        if not sanity_check_perm:
            # case1: no permutation, sample from A and then from B
            u, v = (
                np.random.choice(n_arr1, size=m_permutation, p=weights1),
                np.random.choice(n_arr2, size=m_permutation, p=weights2),
            )
            first_set = arr1[u]
            second_set = arr2[v]
        else:
            # case2: permutation, sample from A+B twice (sanity check)
            u, v = (
                np.random.choice(n_arr1 + n_arr2, size=m_permutation),
                np.random.choice(n_arr1 + n_arr2, size=m_permutation),
            )
            concat_arr = np.concatenate((arr1, arr2))
            first_set = concat_arr[u]
            second_set = concat_arr[v]
    else:
        first_set = arr1
        second_set = arr2
    return first_set, second_set


def credible_intervals(
    ary: np.ndarray, confidence_level: Union[float, List[float], np.ndarray] = 0.94
) -> np.ndarray:
    """
    Calculate highest posterior density (HPD) of array for given credible_interval.

    Taken from the arviz package
    The HPD is the minimum width Bayesian credible interval (BCI). This implementation works only
    for unimodal distributions.

    Parameters
    ----------
    ary
        posterior samples
    confidence_level
        confidence level

    Returns
    -------
    type
        intervals minima, intervals maxima
    """
    if ary.ndim > 1:
        hpd = np.array(
            [
                credible_intervals(row, confidence_level=confidence_level)
                for row in ary.T
            ]
        )
        return hpd
    # Make a copy of trace
    ary = ary.copy()
    n = len(ary)
    ary = np.sort(ary)
    interval_idx_inc = int(np.floor(confidence_level * n))
    n_intervals = n - interval_idx_inc
    interval_width = ary[interval_idx_inc:] - ary[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError(
            "Too few elements for interval calculation. "
            "Check that credible_interval meets condition 0 =< credible_interval < 1"
        )
    min_idx = np.argmin(interval_width)
    hdi_min = ary[min_idx]
    hdi_max = ary[min_idx + interval_idx_inc]
    return np.array([hdi_min, hdi_max])


def describe_continuous_distrib(
    samples: Union[np.ndarray, torch.Tensor],
    credible_intervals_levels: Optional[Union[List[float], np.ndarray]] = None,
) -> dict:
    """
    Computes properties of distribution based on its samples.

    Parameters
    ----------
    samples
        samples of shape (n_samples, n_features)
    credible_intervals_levels
        Confidence in (0, 1)
        of credible intervals to be computed

    Returns
    -------
    type
        properties of distribution
    """
    dist_props = dict(
        mean=samples.mean(0),
        median=np.median(samples, 0),
        std=samples.std(0),
        min=samples.min(0),
        max=samples.max(0),
    )
    credible_intervals_levels = (
        [] if credible_intervals_levels is None else credible_intervals_levels
    )
    for confidence in credible_intervals_levels:
        intervals = credible_intervals(samples, confidence_level=confidence)
        interval_min, interval_max = intervals[:, 0], intervals[:, 1]
        conf_str = str(confidence)[:5]
        dist_props["confidence_interval_{}_min".format(conf_str)] = interval_min
        dist_props["confidence_interval_{}_max".format(conf_str)] = interval_max

    return dist_props


def save_cluster_xlsx(
    filepath: str, de_results: List[pd.DataFrame], cluster_names: List
):
    """
    Saves multi-clusters DE in an xlsx sheet.

    Parameters
    ----------
    filepath
        xslx save path
    de_results
        list of pandas Dataframes for each cluster
    cluster_names
        list of cluster names

    """
    writer = pd.ExcelWriter(filepath, engine="xlsxwriter")
    for i, x in enumerate(cluster_names):
        de_results[i].to_excel(writer, sheet_name=str(x))
    writer.close()


def densify(arr):
    if issparse(arr):
        return np.asarray(arr.todense()).squeeze()
    return arr


def _prepare_obs(
    idx1: Union[List[bool], np.ndarray, str],
    idx2: Union[List[bool], np.ndarray, str],
    adata: AnnData,
):
    """
    Construct an array used for masking.

    Given population identifiers `idx1` and potentially `idx2`,
    this function creates an array `obs_col` that identifies both populations
    for observations contained in `adata`.
    In particular, `obs_col` will take values `group1` (resp. `group2`)
    for `idx1` (resp `idx2`).

    Parameters
    ----------
    idx1
        Can be of three types. First, it can corresponds to a boolean mask that
        has the same shape as adata. It can also corresponds to a list of indices.
        Last, it can correspond to string query of adata.obs columns.
    idx2
        Same as above
    adata
        Anndata
    """

    def ravel_idx(my_idx, obs_df):
        return (
            obs_df.index.isin(obs_df.query(my_idx).index)
            if isinstance(my_idx, str)
            else np.asarray(my_idx).ravel()
        )

    obs_df = adata.obs
    idx1 = ravel_idx(idx1, obs_df)
    g1_key = "one"
    obs_col = np.array(["None"] * adata.shape[0], dtype=str)
    obs_col[idx1] = g1_key
    group1 = [g1_key]
    group2 = None if idx2 is None else "two"
    if idx2 is not None:
        idx2 = ravel_idx(idx2, obs_df)
        obs_col[idx2] = group2
    if (obs_col[idx1].shape[0] == 0) or (obs_col[idx2].shape[0] == 0):
        raise ValueError("One of idx1 or idx2 has size zero.")
    return obs_col, group1, group2


def _de_core(
    adata,
    model_fn,
    groupby,
    group1,
    group2,
    idx1,
    idx2,
    all_stats,
    all_stats_fn,
    col_names,
    mode,
    batchid1,
    batchid2,
    delta,
    batch_correction,
    fdr,
    silent,
    **kwargs
):
    """Internal function for DE interface."""
    if group1 is None and idx1 is None:
        group1 = adata.obs[groupby].astype("category").cat.categories.tolist()
        if len(group1) == 1:
            raise ValueError(
                "Only a single group in the data. Can't run DE on a single group."
            )

    if not isinstance(group1, IterableClass) or isinstance(group1, str):
        group1 = [group1]

    # make a temp obs key using indices
    temp_key = None
    if idx1 is not None:
        obs_col, group1, group2 = _prepare_obs(idx1, idx2, adata)
        temp_key = "_scvi_temp_de"
        adata.obs[temp_key] = obs_col
        groupby = temp_key

    df_results = []
    dc = DifferentialComputation(model_fn, adata)
    for g1 in track(
        group1,
        description="DE...",
        disable=silent,
    ):
        cell_idx1 = (adata.obs[groupby] == g1).to_numpy().ravel()
        if group2 is None:
            cell_idx2 = ~cell_idx1
        else:
            cell_idx2 = (adata.obs[groupby] == group2).to_numpy().ravel()

        all_info = dc.get_bayes_factors(
            cell_idx1,
            cell_idx2,
            mode=mode,
            delta=delta,
            batchid1=batchid1,
            batchid2=batchid2,
            use_observed_batches=not batch_correction,
            **kwargs,
        )

        if all_stats is True:
            genes_properties_dict = all_stats_fn(adata, cell_idx1, cell_idx2)
            all_info = {**all_info, **genes_properties_dict}

        res = pd.DataFrame(all_info, index=col_names)
        sort_key = "proba_de" if mode == "change" else "bayes_factor"
        res = res.sort_values(by=sort_key, ascending=False)
        if mode == "change":
            res["is_de_fdr_{}".format(fdr)] = _fdr_de_prediction(
                res["proba_de"], fdr=fdr
            )
        if idx1 is None:
            g2 = "Rest" if group2 is None else group2
            res["comparison"] = "{} vs {}".format(g1, g2)
            res["group1"] = g1
            res["group2"] = g2
        df_results.append(res)

    if temp_key is not None:
        del adata.obs[temp_key]

    result = pd.concat(df_results, axis=0)

    return result


def _fdr_de_prediction(posterior_probas: np.ndarray, fdr: float = 0.05):
    """Compute posterior expected FDR and tag features as DE."""
    if not posterior_probas.ndim == 1:
        raise ValueError("posterior_probas should be 1-dimensional")
    sorted_genes = np.argsort(-posterior_probas)
    sorted_pgs = posterior_probas[sorted_genes]
    cumulative_fdr = (1.0 - sorted_pgs).cumsum() / (1.0 + np.arange(len(sorted_pgs)))
    d = (cumulative_fdr <= fdr).sum()
    pred_de_genes = sorted_genes[:d]
    is_pred_de = np.zeros_like(cumulative_fdr).astype(bool)
    is_pred_de[pred_de_genes] = True
    return is_pred_de