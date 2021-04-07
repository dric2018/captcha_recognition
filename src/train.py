from tokenizer import Tokenizer

if __name__ == '__main__':
    tok = Tokenizer()

    print(tok.vocab)
    ttc = 'abraca dabra 7<> 9 86'

    print(ttc)
    code = tok.encode(text=ttc, padding=True, max_length=30)

    print(code)

    decoded_txt = tok.decode(ids=code['input_ids'])

    print(decoded_txt)