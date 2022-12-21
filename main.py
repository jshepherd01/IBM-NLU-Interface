from dotenv import load_dotenv
import os
import json
import argparse
from datetime import datetime
from ibm_watson import NaturalLanguageUnderstandingV1
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator
import ibm_watson.natural_language_understanding_v1 as nlu1

load_dotenv()

EMOTIONS = ['joy', 'sadness', 'anger', 'fear', 'disgust']

def init_model(api_key=None, api_url=None):
    if api_key is None: api_key = os.getenv('IBM_NLU_API_KEY')
    if api_url is None: api_url = os.getenv('IBM_NLU_API_URL')
    auth = IAMAuthenticator(api_key)
    model = NaturalLanguageUnderstandingV1(version='2022-04-07', authenticator=auth)
    model.set_service_url(api_url)

    return model

def main():
    parser = argparse.ArgumentParser(description="Interface with IBM Watson Natural Language Understanding.")
    inp = parser.add_mutually_exclusive_group(required=True)
    tar = parser.add_mutually_exclusive_group(required=True)
    ver = parser.add_mutually_exclusive_group()
    inp.add_argument("-i", "--input", help="The text to parse.", dest="text")
    inp.add_argument("-f", "--file", help="The input file to read. Ignored if -i is specified. Expects inputs separated by newlines.")
    tar.add_argument("-t", "--targets", nargs='+', help="Targets to analyse specifically.")
    tar.add_argument("-s", "--targets-file", help="The targets file to read. Expects comma-separated targets, aligned line-wise with the inputs in the -f file.")
    parser.add_argument("-n", "--no-save", action="store_true", help="Do not save the results to the out.json file.")
    ver.add_argument("-v", "--verbose", action="store_true", help="Print additional output text.")
    ver.add_argument("-q", "--quiet", action="store_true", help="No command line output.")
    args = parser.parse_args()

    if (args.text and args.targets_file):
        parser.print_usage()
        print(f"{parser.prog}: error: cannot use -s/--targets-file with -i/--input. Use -t/--targets instead.")
        raise SystemExit()
    if (args.file and args.targets):
        parser.print_usage()
        print(f"{parser.prog}: error: cannot use -t/--targets with -f/--file. Use -s/--targets-file instead.")
        raise SystemExit()

    text_inp = []
    targ_inp = []

    if args.text:
        text_inp.append(args.text)

        if args.targets:
            targ_inp.append(args.targets)
    
    elif args.file:
        with open(args.file) as f:
            r_text_inp = [l for l in f.read().split('\n') if len(l.strip()) > 0 and l[0] != '#']

        if args.targets_file:
            with open(args.targets_file) as f:
                r_targ_inp = [[t.strip() for t in ts.split(',') if len(t.strip()) > 0] for ts in f.read().split('\n') if len(ts.strip()) > 0 and ts[0] != '#']
        
            if len(r_text_inp) != len(r_targ_inp):
                print("The target file and the input file must have the same number of lines.")
                raise SystemError()
            
            for i in range(len(r_text_inp)):
                if len(r_text_inp[i].strip()) >= 1:
                    text_inp.append(r_text_inp[i])
                    targ_inp.append(r_targ_inp[i])

    model = init_model()

    results = {}
    results['date'] = datetime.now()
    results['batch'] = []

    if not args.quiet:
        print()
    for text, targets in zip(text_inp, targ_inp):
        res = analyse(text, targets, model, args.verbose, args.quiet)
        results['batch'].append(res)
    
    if not args.no_save:
        if not args.quiet:
            print("Saving...")
        with open('out.json', 'r') as f:
            past = json.load(f)
        past.append(results)
        with open('out.json', 'w') as f:
            json.dump(past, f, default=str)
        if not args.quiet:
            print("Done.")

def analyse(text, targets, model, verbose=False, quiet=False):
    if not quiet:
        print(f"Analysing: {text}")
        print()

    opts = nlu1.EmotionOptions(targets=targets)
    res = model.analyze(text=text, features=nlu1.Features(emotion=opts)).get_result()
    
    col_0_width = max(len(t) for t in targets+['document'])

    if verbose:
        print(f"Usage: {res['usage']['text_units']} units; {res['usage']['text_characters']} characters.")
        print()
        print(f"Language: {res['language']}")
        print()
    
    if not quiet:
        print("Results")
        print()
        print(' | '.join(['item'.ljust(col_0_width)] + [e.ljust(7) for e in EMOTIONS]))
        print('-|-'.join(['-'*col_0_width] + ['-'*7 for _ in EMOTIONS]))
        print(' | '.join(['document'.ljust(col_0_width)] + [f"{res['emotion']['document']['emotion'][e]:.3f}".ljust(7) for e in EMOTIONS]))
        for t in res['emotion']['targets']:
            print(' | '.join([t['text'].ljust(col_0_width)] + [f"{t['emotion'][e]:.3f}".ljust(7) for e in EMOTIONS]))
        print()


    res['text'] = text

    return res


if __name__ == "__main__":
    main()
