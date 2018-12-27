import json
import argparse


def parse_arg():
    parser = argparse.ArgumentParser(description='Convert log into CSV')
    parser.add_argument('-c', '--columns', type=str,
                        help='list columns seperated by comma.')
    parser.add_argument('-o', '--out_file', type=str,
                        help='specify output file.')
    parser.add_argument('log_file', type=str, nargs=1,
                        help='log filename.')
    return parser.parse_args()


def main():
    args = parse_arg()

    result = []

    col_names = None
    if args.columns:
        col_names = args.columns.split(',')
        result.append(','.join(col_names))

    with open(args.log_file[0]) as f:
        json_data = json.load(f)
        for row in json_data:
            if col_names is None:
                col_names = list(row.keys())
                result.append(','.join(col_names))
            result.append(','.join([str(row[c]) for c in col_names]))

    if args.out_file:
        with open(args.out_file, 'w') as f:
            f.write('\n'.join(result))
    else:
        print('\n'.join(result))


if __name__ == '__main__':
    main()
