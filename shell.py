import argparse
import logging
import os

from defaults import __version__, DEFAULT_THETA_RES, DEFAULT_TMAX, DEFAULT_DT
from double import do_the_thing


def main():
    # Setup command line option parser
    parser = argparse.ArgumentParser(
        description='Parametric modeling of buckling and free vibration in '\
                    'prismatic shell structures, performed by solving the '\
                    'eigenvalue problem in HCFSM.'
    )
    #parser.add_argument(
    #    'data_file',
    #    help="Data file describing the parametric model, please see "\
    #         "'examples/data-files/barbero-viscoelastic.yaml' for an example"
    #)
    parser.add_argument(
        '--theta_resolution',
        type = int,
        default=DEFAULT_THETA_RES
    )
    parser.add_argument(
        '--tmax',
        type = int,
        default=DEFAULT_TMAX
    )
    parser.add_argument(
        '--dt',
        type = int,
        default=DEFAULT_DT
    )


    '''
    parser.add_argument(
        '-d',
        '--purge-integral-db-cache',
        action='store_true',
        help='Purge the integral db cache, forcing it to redownload'
    )
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_const',
        const=logging.WARN,
        dest='verbosity',
        help='Be quiet, show only warnings and errors'
    )
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_const',
        const=logging.DEBUG,
        dest='verbosity',
        help='Be very verbose, show debug information'
    )
    '''
    parser.add_argument(
        '--version',
        action='version',
        version="%(prog)s " + __version__
    )

    args = parser.parse_args()

    # Configure logging
    #log_level = args.verbosity or logging.INFO
    #logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")

    #if not args.results_file:
    #    args.results_file = os.path.splitext(args.data_file)[0] + '.hdf5'

    do_the_thing(
        theta_resolution=args.theta_resolution,
        tmax=args.tmax,
        dt=args.dt
    )

if __name__ == '__main__':
    main()
