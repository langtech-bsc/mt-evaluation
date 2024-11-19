import streamlit as st
import pandas as pd
import os
import json

DATA_PATH_csv = '../results_summary/results_summary.csv'
DATA_PATH = '../results'

INVALID_LOW = -1000
INVALID_HIGH = 1000

def main():

    st.set_page_config(page_title="Gender Bias", page_icon="🔎", layout="wide")

    if os.path.exists(DATA_PATH_csv):

        # Tabs
        tab1, tab2 = st.tabs([
            "Must-she", 
            "MMHB"
        ])

        df = pd.read_csv(DATA_PATH_csv)

        with tab1:
            
            df_must_she = df.loc[df['dataset'] == 'must_she']

            target_col, models_selected = st.columns(2)

            with target_col:
                target_selected = st.selectbox('Select Target Language', sorted( df_must_she['target'].unique() ) )

            df_must_she_filtered = df_must_she.loc[df_must_she['target'] == target_selected]

            with models_selected:
                selected_models = st.multiselect('Select models to compare', options = df_must_she_filtered['model_name'].unique())

            st.dataframe(df_must_she_filtered)
            JSONS = []
            NAMES = []
            if not df_must_she_filtered.empty:    
                for index, row in df_must_she_filtered.iterrows():
                    # Get the model_name and file_name
                    model_name = row['model_name']
                    file_name = row['file_name']
                    
                    file_path = os.path.join(DATA_PATH, model_name, file_name)
                    JSONS.append(file_path)
                    NAMES.append(model_name)
            else:
                st.write('No data available for the selected combination.')


            # Load the JSON data
            def load_data(json_path):
                with open(json_path, 'r', encoding='utf8') as f:
                    data = json.load(f)
                first_key = next(iter(data['results']))
                return data['results'][first_key]


            # load models data
            DATA_MODELS = {}

            for modelname, json_path in zip(NAMES, JSONS):
                if modelname in selected_models:
                    data = load_data(json_path)
                    # Extract corpus-level metrics, if they are not computed, 
                    # then assign the corresponding default value, -1000 for 
                    # high is better, +1000 for lower is better metrics

                    bleu_corpus = data.get('bleu,none', -1000)
                    ter_corpus = data.get('ter,none', 1000)
                    chrf_corpus = data.get('chrf,none', -1000)
                    comet_corpus = data.get('comet,none', -1000)
                    cometkiwi_corpus = data.get('comet_kiwi,none', -1000)

                    DATA_MODELS[modelname] = {  
                                                # metrics
                                                'bleu':bleu_corpus, 'ter':ter_corpus, 'chrf':chrf_corpus, 
                                                'comet': comet_corpus, 'comet-kiwi': cometkiwi_corpus
                                            }

            if DATA_MODELS:
                st.divider()

                # List of available metrics
                AVAILABLE_METRICS = ['bleu', 'ter', 'chrf', 'comet', 'comet-kiwi']

                # Precompute the best metrics
                best_metrics = {
                    "bleu": max(data_model['bleu'] for data_model in DATA_MODELS.values()),
                    "ter": min(data_model['ter'] for data_model in DATA_MODELS.values()),
                    "chrf": max(data_model['chrf'] for data_model in DATA_MODELS.values()),
                    "comet": max(data_model['comet'] for data_model in DATA_MODELS.values()),
                    "comet-kiwi": max(data_model['comet-kiwi'] for data_model in DATA_MODELS.values()),
                }


                # Allow the user to select which metrics to display
                selected_metrics = st.multiselect(
                    'Select metrics to display:', 
                    options=AVAILABLE_METRICS, 
                    default=['bleu', 'ter', 'chrf', 'comet', 'comet-kiwi']
                )

                # Helper function to calculate delta and decide display logic
                def display_metric(column, name, value, best_value, inverse=False):
                    if value == INVALID_LOW or value == INVALID_HIGH:
                        column.metric(name, None, delta=None, delta_color='off')
                        return

                    delta = round(value - best_value, 3)
                    is_best = delta == 0

                    delta_value = delta if not is_best else 'BEST'
                    if delta < 0 and not inverse:
                        delta_color = 'normal'
                    elif delta > 0 and inverse:
                        delta_color = 'inverse'
                    else:
                        delta_color = 'off'

                    column.metric(name, round(value, 2), delta=delta_value, delta_color=delta_color)

                st.divider()
                # Loop through each model and display the selected metrics
                for modelname, data_model in DATA_MODELS.items():
                    st.markdown(f'###### **{modelname}**')
                    cols = st.columns(max(len(selected_metrics), 1))
                    
                    # Conditional display of selected metrics
                    for idx, metric in enumerate(selected_metrics):
                        if metric == 'bleu':
                            display_metric(cols[idx], "BLEU", data_model['bleu'], best_metrics['bleu'])
                        elif metric == 'ter':
                            display_metric(cols[idx], "TER", data_model['ter'], best_metrics['ter'], inverse=True)
                        elif metric == 'chrf':
                            display_metric(cols[idx], "ChrF", data_model['chrf'], best_metrics['chrf'])
                        elif metric == 'comet':
                            display_metric(cols[idx], "COMET", data_model['comet'], best_metrics['comet'])
                        elif metric == 'comet-kiwi':
                            display_metric(cols[idx], "COMET-KIWI", data_model['comet-kiwi'], best_metrics['comet-kiwi'])
                    st.divider()

            else:
                st.warning("Corpus-Level Metrics require data from at least one model. Please select a model to proceed.")

        with tab2:
            pass
    else:
        st.error(f"File not found: {DATA_PATH_csv}. Please ensure the file exists.")

if __name__ == "__main__":
    main()