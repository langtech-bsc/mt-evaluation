from lm_eval.api.registry import register_task
from lm_eval.api.mt_task import MTask

class MUST_SHE(MTask):
    VERSION = 1
    DATASET_PATH = "must_she"
    DATASET_NAME = None
    OUTPUT_TYPE = "generate_until"

    def doc_to_text(self, doc):
        return doc['SRC']
    
    def doc_to_target(self, doc):
        return doc['REF']

    def get_row_values(self, doc):
        return  {
                'ID': doc['ID'], 
                'LANG': doc['LANG'], 
                'TALK': doc['TALK'], 
                'WRONG-REF': doc['WRONG-REF'], 
                'SPEAKER': doc['SPEAKER'], 
                'GENDER': doc['GENDER'], 
                'CATEGORY': doc['CATEGORY'],
                'TEXT-CATEGORY': doc['TEXT-CATEGORY'], 
                'GENDERTERMS': doc['GENDERTERMS']
                }

    # Redefine preprocess_results, aggregation, higher_is_better
    # Add mt_gender_evaluation metrics for must_she

    def process_results(self, doc, results):

        # load yaml config
        if self.metric_configs is None:
            self.load_yaml_config()

        source = self.doc_to_text(doc)
        target = self.doc_to_target(doc)
        result = results[0]
        row_value = self.get_row_values(doc)

        self.create_dicts(source, target, result)
        self.res['must_she_scores'] = ( result, row_value )
        return self.res

    def aggregation(self):
        """
        Returns a dictionary of aggregation functions for metrics.
        Returns:
            dict: A dictionary where keys are metric names and values are functions that aggregate metric scores.
        """ 

        self.dict_aggregated['must_she_scores'] = self.must_she_scores
        return self.dict_aggregated

    def must_she_scores(self, arr):

        results = [i[0] for i in arr]
        row_values = [i[1] for i in arr]

        self.sl_scores, self.found_categories = self.sentence_level_scores( results, row_values )
        self.overall_scores = self.global_scores( self.sl_scores, row_values )

        return [ self.overall_scores ]

    def sentence_level_scores( self, results, row_values ):
        sentences, found_right, found_wrong, not_found, gender_terms = [], [], [], [], []
        char_remov=["'", ".", ",", "?", ";", "!", ":", "  "]

        for (i_line, terms_f) in zip(results, row_values):
            sentence_correct, sentence_wrong, sentence_found = 0, 0, 0
            
            gender_marked_terms = terms_f['GENDERTERMS'].strip().lower().split(";")
            gender_terms.extend(gender_marked_terms)

            for char in char_remov:
                i_line = i_line.replace(char, " ")

            generated_terms = i_line.lower().strip().split()
                        
            for t in gender_marked_terms:
                pos_found = -1
                term = t.split(" ")
                found = False
                correct_term = term[0].strip()

                try:
                    wrong_term = term[1].strip()
                except:
                    pass
                
                if correct_term in generated_terms:
                    pos_found = generated_terms.index(correct_term)
                    del generated_terms[pos_found]
                    found_right.extend([correct_term])
                    sentence_correct += 1
                    found = True
                
                elif not found and wrong_term in generated_terms:
                    pos_found = generated_terms.index(wrong_term)
                    del generated_terms[pos_found]
                    found_wrong.extend([correct_term])
                    sentence_wrong += 1
                    found = True
                
                if found:
                    sentence_found += 1
                else:
                    not_found.extend([correct_term])

            gender_cat = terms_f['TEXT-CATEGORY']
            sentences.append({
                "num_terms": len(gender_marked_terms),
                "num_terms_found": sentence_found,
                "num_correct": sentence_correct,
                "num_wrong": sentence_wrong,
                "gender_cat": gender_cat})

        #print('Found right: ',len(found_right), 'Found wrong: ', len(found_wrong), 'Not Found:', len(not_found))
        found_categories = [found_right, found_wrong, not_found]
        #print('Number of gender marked terms: ', len(gender_terms))
        #print('stored tokens to parse: ', len(found_right)+len(found_wrong)+len(not_found))
        return sentences, found_categories
    
    def global_scores(self, sentence_scores, row_values):
        i = 0
        category_buffers = {}
        for line in row_values:
            category = line["TEXT-CATEGORY"]
            if category not in category_buffers:
                category_buffers[category] = {"num_terms": 0, "num_correct": 0, "num_wrong": 0, "num_terms_found": 0}
            try:
                category_buffers[category]["num_terms"] += sentence_scores[i]["num_terms"]
            except:
                pass

            category_buffers[category]["num_terms_found"] += sentence_scores[i]["num_terms_found"]
            category_buffers[category]["num_correct"] += sentence_scores[i]["num_correct"]
            category_buffers[category]["num_wrong"] += sentence_scores[i]["num_wrong"]
            i += 1

        overall_scores = {}
        tot_terms, tot_found, tot_correct, tot_wrong = 0, 0, 0, 0
        for c in category_buffers:
            term_cov = float(category_buffers[c]["num_terms_found"]) / category_buffers[c]["num_terms"]
            if category_buffers[c]["num_terms_found"] > 0:
                gender_acc = float(category_buffers[c]["num_correct"]) / (category_buffers[c]["num_correct"] + category_buffers[c]["num_wrong"])
            else:
                gender_acc = 0.0
            overall_scores[c] = {"term_coverage": term_cov, "gender_accuracy": gender_acc}

            tot_terms += category_buffers[c]["num_terms"]
            tot_found += category_buffers[c]["num_terms_found"]
            tot_correct += category_buffers[c]["num_correct"]
            tot_wrong += category_buffers[c]["num_wrong"]
        
        overall_scores["Global"] = {
            "term_coverage": tot_found / tot_terms,
            "gender_accuracy": tot_correct / (tot_correct + tot_wrong)
            }

        return overall_scores

@register_task("en_ca_must_she")
class MUST_SHE_EN_CA(MUST_SHE):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'en_ca'})

    def get_target(self):
        return 'cat_Latn'

@register_task("en_es_must_she")
class MUST_SHE_EN_ES(MUST_SHE):
    def __init__(self, config=None):
        super().__init__(config={'target_delimiter': '', 'validation_split':'en_es'})

    def get_target(self):
        return 'spa_Latn'