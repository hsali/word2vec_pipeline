from word2vec_pipeline.preprocessing import token_replacement 

def test_replacement():
    P = token_replacement()

    # We should test the split string, as extra spcaes could be put in.
    # It doesn't make a difference if they are there or not so we ignore them
    
    assert(P("Hi & Low").split() == "Hi and Low".split())
    assert(P("20% and 8%").split() == "20 percent and 8 percent".split())
