import argparse
import numpy as np
import effex as fx


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # ARGUMENT PARSING
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description='Pull data from effex-generated .csv file and post-process it. Shows a plot.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('filename',
        type=str,
        help='(str) output visibilities.csv file from effex.'
    )

    args = parser.parse_args()

    metadata = {}
    for meta in np.loadtxt(args.filename, dtype=str, max_rows=1, delimiter=','):
        key, val = meta.split(':')
        metadata[key] = val

    if metadata['mode'].lower() in ['continuum', 'test']:
        skiprows = 1
    else:
        skiprows = 2

    # Magic number: (1/f) / 10 is the spacing the Correlator class uses in
    # to step through delay-space in test mode to ensure the delay pattern is
    # adequately sampled
    if 'test' == metadata['mode'].lower():
        test_delay_sweep_step = (1 / float(metadata['frequency'])) / 10.
    else:
        test_delay_sweep_step = 0

    output = np.loadtxt(args.filename, dtype=np.complex128, delimiter=',', skiprows=skiprows)

    fx.post_process(output,
        float(metadata['bandwidth']),
        float(metadata['frequency']),
        int(metadata['resolution']),
        int(metadata['num_samp']),
        metadata['mode'].lower(),
        False, # never omit plot, that is the point
        test_delay_sweep_step=test_delay_sweep_step
    )

