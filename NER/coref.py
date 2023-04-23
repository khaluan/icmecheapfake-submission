import spacy
import time
from wasabi import msg  

def resolve_references(inp_str: str, verbose = False) -> str:
    """Function for resolving references with the coref ouput
    inp_str (str): The paragraph that needs to be resolved
    RETURNS (str): The Doc string with resolved references
    """
    global nlp
    doc = nlp(inp_str)
    if verbose:
        print(doc.spans)

    # token.idx : token.text
    token_mention_mapper = {}
    output_string = ""
    clusters = [
        val for key, val in doc.spans.items() if key.startswith("coref_cluster")
    ]

    # Iterate through every found cluster
    for cluster in clusters:
        first_mention = cluster[0]
        # Iterate through every other span in the cluster
        for mention_span in list(cluster)[1:]:
            # Set first_mention as value for the first token in mention_span in the token_mention_mapper
            token_mention_mapper[mention_span[0].idx] = first_mention.text + mention_span[0].whitespace_
            
            for token in mention_span[1:]:
                # Set empty string for all the other tokens in mention_span
                token_mention_mapper[token.idx] = ""

    # Iterate through every token in the Doc
    for token in doc:
        # Check if token exists in token_mention_mapper
        if token.idx in token_mention_mapper:
            output_string += token_mention_mapper[token.idx]
        # Else add original token text
        else:
            output_string += token.text + token.whitespace_

    return output_string

nlp = spacy.load('en_coreference_web_trf')
msg.info("Pipeline construct complete")

# print(resolve_references('Philips plays the guitar because he loves it.', True))
