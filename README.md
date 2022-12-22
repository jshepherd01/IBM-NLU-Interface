# IBM-NLU-Interface
Interface with the IBM Natural Language Understanding API

## How to use

1. Follow [these instructions](https://cloud.ibm.com/docs/natural-language-understanding?topic=natural-language-understanding-getting-started) under "Before you begin" up to copying the API Key and URl values.
2. Create a file in this directory called `.env` containing:
```
IBM_NLU_API_URL={the url}
IBM_NLU_API_KEY={the key}
```
3. Execute main.py by providing one line of text such as:
```
python3 main.py -i "I love apples! I don't like oranges." -t apples oranges
```
or using multiple lines in a file such as:
```
python3 main.py -f data/inp.txt -s data/targets.txt
```
Make sure you specify both text and at least one target, otherwise the API won't work.

## Input format

The input file containing text should have a series of sentences separated by newlines. The targets file should have a series of words or phrases separated by newlines. If you want to specify multiple targets for the same sentence, you can write them on the same line separated by commas. In either file, you can start a line with `#` to mark it as a comment, and then it will be ignored. See the [data](/data) directory for examples.
