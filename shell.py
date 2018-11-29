import argparse
import logging
import os

from defaults import __version__, DEFAULT_THETA_RES, DEFAULT_TMAX, DEFAULT_DT, DEFAULT_FILENAME
from double import do_the_thing


def main():
    # Setup command line option parser
    parser = argparse.ArgumentParser(
        description='Double pendulum simulation.'
    )
    parser.add_argument(
		'-r',
        '--theta_resolution',
        type = int,
        default=DEFAULT_THETA_RES
    )
    parser.add_argument(
        '--tmax',
        type = int,
        default=DEFAULT_TMAX,
		help='end time'
    )
    parser.add_argument(
        '--dt',
        type = int,
        default=DEFAULT_DT,
		help='delta time'
    )
    parser.add_argument(
        '-o',
		'--output_filename',
        default=DEFAULT_FILENAME,
		help='output csv file filename'
    )
    parser.add_argument(
        '-g',
        '--graph',
        action='store_true',
        help='Draw the graph too.'
    )
    parser.add_argument(
        '-p',
        '--parallel',
        action='store_true',
        help='Use multiprocessing to parallelize the code.'
    )
    '''
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
        dt=args.dt,
		filename=args.output_filename,
		graph=args.graph,
		parallel=args.parallel
    )

if __name__ == '__main__':
    main()
