from refined.inference.processor import Refined
import time
start = time.time()
refined = Refined.from_pretrained(model_name='wikipedia_model_with_numbers',
                                  entity_set="wikipedia")

def entity_linking(doc: str):
    '''
        Input: doc(str) input string for entity linking
        Returns: List[Tuple(entity_text: str, entity: Entity, entity_type: str)]
                The id of the entity could be accessed via entity.wikidata_entity_id (if entity is not None)
    '''
    global refined
    spans = refined.process_text(doc)
    return [(span.text, 
             span.predicted_entity.wikidata_entity_id if span.predicted_entity else None, 
             span.coarse_mention_type) for span in spans]

# end = time.time()
# print(f'Init: {end-start}')
# start = time.time()
# spans = refined.process_text('Julian Castro at his announcement in San Antonio, Tex., on Saturday. Mr. Castro, the former secretary of housing and urban development, would be one of the youngest presidents if elected.')
# end = time.time()
# print(f"Process: {end-start}")
# for span in spans:
#     print(span.text)
#     print(span.predicted_entity)
#     entity = span.predicted_entity
#     print(entity.wikidata_entity_id, entity.wikipedia_entity_title, entity.human_readable_name, entity.parsed_string)
#     print(span.coarse_mention_type)
#     print("")
    