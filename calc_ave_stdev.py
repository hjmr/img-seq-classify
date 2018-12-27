import json
import argparse
import numpy as np


def parse_arg():
    parser = argparse.ArgumentParser(description='Calc average and standard deviation from logs.')
    parser.add_argument('-c', '--columns', type=str, nargs=1,
                        help='list columns seperated by comma')
    parser.add_argument('-o', '--out_file', type=str,
                        help='output resulting json to the specified file.')
    parser.add_argument('log_files', type=str, nargs='+',
                        help='log filenames.')
    return parser.parse_args()


def main():
    args = parse_arg()

    col_names = args.columns[0].split(',')

    json_data = []
    for log_file in args.log_files:
        with open(log_file) as f:
            json_data.append(json.load(f))

    tmp = []
    for row in json_data[0]:
        r_dict = {}
        for c in col_names:
            if c == 'epoch':
                r_dict[c] = 0
            else:
                r_dict[c] = []
        tmp.append(r_dict)

    for j in json_data:
        for idx, row in enumerate(j):
            r_dict = tmp[idx]
            for c in col_names:
                if c == 'epoch':
                    r_dict[c] = row[c]
                else:
                    r_dict[c].append(row[c])

    result = []
    for row in tmp:
        r_dict = {}
        for c in col_names:
            if c == 'epoch':
                r_dict[c] = row[c]
            else:
                v = np.array(row[c], dtype=np.float32)
                r_dict['{}/average'.format(c)] = np.average(v).astype(float)
                r_dict['{}/stdev'.format(c)] = np.std(v).astype(float)
        result.append(r_dict)

    if args.out_file:
        with open(args.out_file, 'w') as f:
            json.dump(result, f, ensure_ascii=False, indent=2, sort_keys=False, separators=(',', ': '))
    else:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=False, separators=(',', ': ')))


if __name__ == '__main__':
    main()
