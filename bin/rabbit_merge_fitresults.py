#!/usr/bin/env python3
"""Merge several rabbit fitresults.hdf5 files into one.

rabbit_fit writes one fitresults.hdf5 per invocation, while the consumers
(rabbit_plot_hists, rabbit_print_*) read a single file. Whenever the results
of one analysis end up spread over several invocations, this tool recombines
them; the merge is content-agnostic:

  * results groups with the SAME name are recursively unioned; entries present
    in several inputs must agree (equal values pass silently and double as
    validation; floats within --rtol count as equal, and NaN/None entries —
    rabbit's "not computed", e.g. parms variances from a --noHessian run —
    are filled from the input that computed them), genuinely differing
    entries are conflicts resolved by --prefer;
  * same-name groups must come from the same postfit point (their stored parms
    are compared; --force to skip);
  * results groups unique to one input are copied through;
  * the merged meta is the first input's plus a merged_inputs provenance list.

Typical uses: recombining partial post-processing runs (the --externalPostfit
--noFit pattern splits expensive covariance / impacts / hist errors /
saturated tests across runs); merging runs that each computed a different
mapping or projection, e.g. in parallel; packaging fits of different datasets
(data + asimov, results vs results_asimov) into one multi-result file.

Example — banded postfit plot with the saturated p-value from two partial runs:

    rabbit_merge_fitresults.py cov/fitresults.hdf5 saturated/fitresults.hdf5 \\
        -o merged/fitresults.hdf5
"""

import argparse
import os
import sys

from rabbit import io_tools


def make_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="fitresults.hdf5 files to merge, in precedence order",
    )
    parser.add_argument(
        "-o",
        "--output",
        required=True,
        help="merged output file (must not exist unless --override)",
    )
    parser.add_argument(
        "--prefer",
        choices=["error", "first", "last"],
        default="error",
        help="conflict policy for entries that differ between inputs within a "
        "same-name results group (default: error out listing them)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="skip the guard that same-name results groups share the same "
        "postfit parameter values",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=io_tools.MERGE_RTOL,
        help="relative tolerance for float comparisons between inputs "
        "(cross-run numerical noise; NaN/None entries are treated as "
        "'not computed' and filled from the other input)",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="overwrite the output file if it exists",
    )
    return parser


def main():
    args = make_parser().parse_args()

    if len(args.inputs) < 2:
        sys.exit("need at least two input files to merge")
    if os.path.exists(args.output) and not args.override:
        sys.exit(f"{args.output} exists; use --override to overwrite")
    outdir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(outdir, exist_ok=True)

    merged_groups = io_tools.merge_fitresults(
        args.inputs,
        args.output,
        prefer=args.prefer,
        force=args.force,
        rtol=args.rtol,
    )

    print(f"Wrote {args.output}:")
    for gname, merged in merged_groups.items():
        mappings = list(merged.get("mappings", {}).keys())
        print(
            f"  {gname}: {len(merged)} entries"
            + (f", mappings: {mappings}" if mappings else "")
        )


if __name__ == "__main__":
    main()
